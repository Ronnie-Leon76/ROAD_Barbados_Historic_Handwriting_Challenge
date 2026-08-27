# %% [markdown]
# ## 8. Prompt, dataset and collator
#
# ### Prompt
#
# The prompt encodes three facts the model cannot infer from a single example: the
# label covers **only the centre line** (crops carry clipped neighbours), the original
# early-modern spelling must be preserved rather than modernised, and `^` is a
# meaningful superscript marker (`y^e`, `15^th`, `w^tsoever`). The identical string is
# used at training and at inference — a mismatch there is one of the most common
# silent regressions in VLM fine-tuning.
#
# ### Loss masking
#
# Only the assistant's transcription contributes to the loss. Rather than
# string-matching the answer inside the token stream (which breaks whenever the
# tokenizer merges a boundary character), the assistant header is derived from the
# chat template itself and located as a token subsequence, so masking is exact for any
# Qwen chat template.

# %%
OCR_PROMPT = (
    "This is a cropped line from a 17th-19th century handwritten Barbadian legal record "
    "(deed, will, inventory or census page).\n"
    "Transcribe the single complete line of handwriting that runs across the middle of the image. "
    "Partial lines clipped at the top and bottom edges are context only - do not transcribe them.\n"
    "Reproduce the original early-modern spelling, capitalisation and punctuation exactly as "
    "written; do not modernise or correct it. Keep the caret '^' that marks a superscript "
    "(for example y^e, 15^th, w^tsoever).\n"
    "Reply with the transcription of that line and nothing else."
)


def build_messages(with_answer: str | None = None) -> list:
    msgs = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}],
    }]
    if with_answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": with_answer}]})
    return msgs


def find_last_subsequence(hay: list, needle: list) -> int:
    """Index one past the last occurrence of `needle` in `hay`, or -1."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return -1
    for start in range(n - m, -1, -1):
        if hay[start:start + m] == needle:
            return start + m
    return -1


class LineDataset(torch.utils.data.Dataset):
    """Real training lines plus, optionally, freshly rendered StackMix lines."""

    def __init__(self, df: pd.DataFrame, cfg: Config, train: bool,
                 renderer: "StackMixRenderer | None" = None, seed: int = 0):
        self.ids = df["ID"].tolist()
        self.texts = df["Target"].tolist()
        self.cfg = cfg
        self.train = train
        self.renderer = renderer if train else None
        self.seed = seed
        self.epoch = 0
        self.n_real = len(self.ids)
        self.n_synth = int(self.n_real * cfg.stackmix_ratio) if self.renderer else 0
        real_len = [self._estimate_tokens(i, t) for i, t in zip(self.ids, self.texts)]
        med = int(np.median(real_len)) if real_len else 400
        self.lengths = real_len + [med] * self.n_synth

    def _estimate_tokens(self, image_id: str, text: str) -> int:
        p = CACHE_DIR / f"{image_id}.png"
        if p.exists():
            try:
                with Image.open(p) as im:
                    w, h = im.size
                return (w // self.cfg.patch_grid) * (h // self.cfg.patch_grid) + len(text) // 3
            except Exception:
                pass
        return 400 + len(text) // 3

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_real + self.n_synth

    def __getitem__(self, idx: int) -> dict:
        if idx < self.n_real:
            img = load_line(self.ids[idx], self.cfg)
            text = self.texts[idx]
            if self.train:
                rng = np.random.default_rng((self.seed, self.epoch, idx))
                img = augment_line(img, rng, self.cfg)
        else:
            rng = np.random.default_rng((self.seed, self.epoch, idx, 7919))
            out = None
            for _ in range(8):
                out = self.renderer.render(rng)
                if out is not None:
                    break
            if out is None:  # fall back to a real line rather than skipping a slot
                j = int(rng.integers(self.n_real))
                img, text = load_line(self.ids[j], self.cfg), self.texts[j]
            else:
                img, text = out
            img = augment_line(img, rng, self.cfg)
        return {"image": to_pil(img), "text": text}


class OCRCollator:
    """Chat-template batching with exact assistant-only loss masking."""

    def __init__(self, processor, train: bool = True, max_length: int = 4096):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.train = train
        self.max_length = max_length
        probe_nogen = processor.apply_chat_template(build_messages(), tokenize=False,
                                                    add_generation_prompt=False)
        probe_gen = processor.apply_chat_template(build_messages(), tokenize=False,
                                                  add_generation_prompt=True)
        suffix = probe_gen[len(probe_nogen):] if probe_gen.startswith(probe_nogen) else "assistant\n"
        self.assistant_header = suffix
        self.header_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        if not self.header_ids:
            raise RuntimeError("Could not derive the assistant header from the chat template")

    def __call__(self, examples: list) -> dict:
        images = [ex["image"] for ex in examples]
        texts = [
            self.processor.apply_chat_template(
                build_messages(ex["text"] if self.train else None),
                tokenize=False, add_generation_prompt=not self.train,
            )
            for ex in examples
        ]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        if not self.train:
            return batch

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id
        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()
            cut = find_last_subsequence(seq, self.header_ids)
            if cut < 0:
                raise RuntimeError("Assistant header not found - chat template mismatch")
            labels[i, :cut] = -100
        if pad_id is not None:
            labels[input_ids == pad_id] = -100
        for name in ("image_token_id", "video_token_id", "vision_start_token_id"):
            tok = getattr(self.processor, name, None) or getattr(
                getattr(self.processor, "tokenizer", None), name, None)
            if isinstance(tok, int):
                labels[input_ids == tok] = -100
        batch["labels"] = labels
        return batch
