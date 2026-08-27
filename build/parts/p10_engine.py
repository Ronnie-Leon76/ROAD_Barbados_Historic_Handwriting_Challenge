# %% [markdown]
# ## 10. Transcription engine
#
# Batched beam search that returns an **N-best list with sequence log-probabilities**,
# not just the top string. Those scores are what makes the fusion in `Section 12`
# possible: rescoring and ROVER both need to know how confident each hypothesis was.
#
# Practical details that matter on a rented GPU:
#
# * left padding for generation (right padding silently corrupts decoder output),
# * width-sorted batching so a 3 200 px line is not padded against a 600 px one,
# * automatic batch-size halving on CUDA OOM instead of losing the run,
# * multi-scale test-time augmentation: the same line is read at several
#   resolutions and the hypotheses are pooled.

# %%
def photometric_variant(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return img
    if mode == "dark":     # thicker, higher-contrast ink
        f = np.clip(255.0 * np.power(img / 255.0, 1.35), 0, 255).astype(np.uint8)
        return cv2.erode(f, np.ones((2, 2), np.uint8))
    if mode == "light":    # thinner strokes, flatter paper
        f = np.clip(255.0 * np.power(img / 255.0, 0.78), 0, 255).astype(np.uint8)
        return cv2.dilate(f, np.ones((2, 2), np.uint8))
    if mode == "sharp":
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        return np.clip(cv2.addWeighted(img, 1.6, blur, -0.6, 0), 0, 255).astype(np.uint8)
    raise ValueError(mode)


class TranscriptionEngine:
    def __init__(self, model, processor, cfg: Config):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.tokenizer = processor.tokenizer
        self.prompt = processor.apply_chat_template(build_messages(), tokenize=False,
                                                    add_generation_prompt=True)

    # ------------------------------------------------------------------ io --
    def _images(self, ids: Sequence[str], scale: float, mode: str) -> list:
        out = []
        for i in ids:
            img = load_line(i, self.cfg, scale=scale, use_cache=abs(scale - 1.0) < 1e-6)
            if mode != "identity":
                img = photometric_variant(img, mode)
            out.append(to_pil(img))
        return out

    def _clean(self, text: str) -> str:
        text = text.split("<|im_end|>")[0]
        for tag in ("assistant\n", "assistant:", "<|endoftext|>"):
            if text.startswith(tag):
                text = text[len(tag):]
        return normalize_text(text)

    # -------------------------------------------------------------- decode --
    def _greedy_scores(self, out, seqs) -> list:
        """Length-normalised log-probability for a greedy decode."""
        try:
            trans = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True)
            pad = self.tokenizer.pad_token_id
            valid = (seqs != pad) if pad is not None else torch.ones_like(seqs, dtype=torch.bool)
            valid = valid[:, :trans.shape[1]]
            trans = torch.nan_to_num(trans[:, :valid.shape[1]], neginf=-20.0, posinf=0.0)
            per_token = (trans * valid).sum(-1) / valid.sum(-1).clamp(min=1)
            return per_token.float().cpu().tolist()
        except Exception:
            return [0.0] * seqs.shape[0]

    @torch.inference_mode()
    def _generate_batch(self, images: list, num_beams: int, n_best: int,
                        length_penalty: float, max_new_tokens: int) -> list:
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.processor(text=[self.prompt] * len(images), images=images,
                                    return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = side
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            # `sequences_scores` is silently dropped unless output_scores is on, and
            # the whole rescoring stage depends on having those scores.
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if num_beams > 1:
            gen_kwargs.update(num_return_sequences=min(n_best, num_beams),
                              length_penalty=length_penalty,
                              early_stopping=True)
        out = self.model.generate(**inputs, **gen_kwargs)

        seqs = out.sequences[:, in_len:]
        texts = self.processor.batch_decode(seqs, skip_special_tokens=True)
        if num_beams > 1 and getattr(out, "sequences_scores", None) is not None:
            scores = out.sequences_scores.float().cpu().tolist()
        else:
            scores = self._greedy_scores(out, seqs)
        del out

        k = min(n_best, num_beams) if num_beams > 1 else 1
        grouped = []
        for b in range(len(images)):
            cand = [(self._clean(texts[b * k + j]), float(scores[b * k + j])) for j in range(k)]
            grouped.append(cand)
        return grouped

    def predict(self, ids: Sequence[str], num_beams: int | None = None,
                n_best: int | None = None, scale: float = 1.0, mode: str = "identity",
                length_penalty: float | None = None, batch_size: int | None = None,
                max_new_tokens: int | None = None, desc: str = "decode",
                leave: bool = True) -> dict:
        cfg = self.cfg
        num_beams = num_beams if num_beams is not None else cfg.num_beams
        n_best = n_best if n_best is not None else cfg.n_best
        length_penalty = length_penalty if length_penalty is not None else cfg.length_penalty
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        batch_size = batch_size or max(1, cfg.infer_batch_size // max(1, num_beams // 2))

        ids = list(ids)
        widths = []
        for i in ids:
            p = CACHE_DIR / f"{i}.png"
            try:
                with Image.open(p) as im:
                    widths.append(im.size[0])
            except Exception:
                widths.append(10 ** 6)
        order = np.argsort(-np.asarray(widths))

        results: dict[str, list] = {}
        pbar = progress(total=len(ids), desc=desc, unit="img", leave=leave)
        pos = 0
        bs = batch_size
        while pos < len(order):
            chunk = [ids[j] for j in order[pos:pos + bs]]
            try:
                images = self._images(chunk, scale, mode)
                grouped = self._generate_batch(images, num_beams, n_best,
                                               length_penalty, max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                free_gpu()
                if bs == 1:
                    print(f"!! OOM at batch size 1 on {chunk[0]} - emitting empty string")
                    results[chunk[0]] = [("", -50.0)]
                    pos += 1
                    pbar.update(1)
                    continue
                bs = max(1, bs // 2)
                print(f"!! CUDA OOM - retrying with batch size {bs}")
                pbar.set_postfix_str(f"bs={bs}", refresh=False)
                continue
            for image_id, cand in zip(chunk, grouped):
                results[image_id] = cand
            pos += len(chunk)
            pbar.update(len(chunk))
        pbar.close()
        return {i: results.get(i, [("", -50.0)]) for i in ids}

    def predict_tta(self, ids: Sequence[str], scales: Sequence[float] | None = None,
                    photometric: bool | None = None, **kw) -> dict:
        """Pool N-best lists across resolutions and photometric variants."""
        cfg = self.cfg
        scales = list(scales if scales is not None else cfg.tta_scales)
        photometric = cfg.tta_photometric if photometric is None else photometric
        views = [(s, "identity") for s in scales]
        if photometric:
            views.append((scales[0], "dark"))
            views.append((scales[0], "light"))

        pooled: dict[str, list] = defaultdict(list)
        for k, (scale, mode) in enumerate(views):
            res = self.predict(ids, scale=scale, mode=mode,
                               desc=f"decode {k+1}/{len(views)} (x{scale:g},{mode})", **kw)
            for image_id, cand in res.items():
                for rank, (text, score) in enumerate(cand):
                    pooled[image_id].append({"text": text, "score": score,
                                             "view": f"{scale:g}:{mode}", "rank": rank})
        return dict(pooled)


def top1(nbest: dict) -> dict:
    """Best hypothesis per ID from a pooled N-best structure."""
    out = {}
    for image_id, cands in nbest.items():
        if not cands:
            out[image_id] = ""
            continue
        if isinstance(cands[0], dict):
            best = max(cands, key=lambda c: (c["score"], -c["rank"]))["text"]
        else:
            best = max(cands, key=lambda c: c[1])[0]
        out[image_id] = best
    return out


def nbest_to_frame(nbest: dict) -> pd.DataFrame:
    rows = []
    for image_id, cands in nbest.items():
        for c in cands:
            if isinstance(c, dict):
                rows.append({"ID": image_id, **c})
            else:
                rows.append({"ID": image_id, "text": c[0], "score": c[1],
                             "view": "1:identity", "rank": 0})
    return pd.DataFrame(rows)
