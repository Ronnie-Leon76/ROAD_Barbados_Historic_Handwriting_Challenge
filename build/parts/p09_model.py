# %% [markdown]
# ## 9. Model
#
# `AutoModelForImageTextToText` is used rather than a family-specific class so the
# same code runs on Qwen3-VL, Qwen2.5-VL and Qwen2-VL. Attention falls back
# gracefully: FlashAttention-2 if the wheel imports, otherwise PyTorch SDPA.
#
# LoRA targets are **discovered from the graph** instead of hard-coded suffix lists.
# That matters because the projector and vision blocks are named differently in every
# release, and for handwriting the vision tower is exactly where the adaptation needs
# to happen — the language prior already knows 17th-century legalese, the encoder is
# what has never seen secretary hand.
#
# `build_model` serves both fine-tuning modes and both directions of the checkpoint
# round-trip. `CFG.finetune_mode = "full"` skips PEFT entirely and unfreezes the
# network, honouring `full_freeze_vision` / `full_freeze_llm`; on reload the
# checkpoint directory is inspected rather than assumed, so a full checkpoint
# replaces the backbone while an adapter is layered on top of it. That is what lets
# the same decoding and GRPO code run unchanged whichever mode produced the weights.

# %%
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def resolve_attn(cfg: Config) -> str:
    if cfg.attn_implementation != "auto":
        return cfg.attn_implementation
    try:
        import flash_attn  # noqa: F401

        major = torch.cuda.get_device_properties(0).major if torch.cuda.is_available() else 0
        if major >= 8:
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"


def load_processor(cfg: Config, padding_side: str = "right"):
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        # Our own resizing already snapped to the patch grid; widen the processor's
        # limits so its smart_resize becomes a no-op and the token count is exactly
        # what Section 6 predicted. Note that for Qwen VL processors the entries of
        # `size` are pixel *counts*, not edge lengths.
        max_px = cfg.line_height * 2 * cfg.max_line_width
        min_px = cfg.min_line_height * cfg.min_line_width
        for attr, val in (("min_pixels", min_px), ("max_pixels", max_px)):
            if hasattr(ip, attr):
                setattr(ip, attr, val)
        if isinstance(getattr(ip, "size", None), dict):
            ip.size = {**ip.size, "shortest_edge": min_px, "longest_edge": max_px}
        grid = int(getattr(ip, "patch_size", 16)) * int(getattr(ip, "merge_size", 2))
        if grid != cfg.patch_grid:
            print(f"!! processor tokenises on a {grid}px grid but the image cache was "
                  f"built for {cfg.patch_grid}px. Set CFG.patch_grid = {grid}, delete "
                  f"{CACHE_DIR}, and re-run Section 6.")
    processor.tokenizer.padding_side = padding_side
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def _from_pretrained(cfg: Config, quant, attn, device_map, source: str | None = None):
    source = source or cfg.model_id
    kwargs = dict(attn_implementation=attn, quantization_config=quant,
                  device_map=device_map, trust_remote_code=True)
    try:                       # transformers >= 4.56
        return AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, **kwargs)
    except TypeError:          # older signature
        return AutoModelForImageTextToText.from_pretrained(source, torch_dtype=torch.bfloat16, **kwargs)


def is_adapter_dir(path: str | Path) -> bool:
    """A PEFT adapter ships a config; a full checkpoint ships model weights."""
    return (Path(path) / "adapter_config.json").exists()


def set_trainable_towers(model, cfg: Config) -> tuple[int, int]:
    """Full fine-tune: unfreeze everything, then re-freeze the towers we opted out of."""
    frozen = 0
    for name, param in model.named_parameters():
        low = name.lower()
        is_merger = "merger" in low or "projector" in low
        is_vision = ("visual" in low or "vision" in low) and not is_merger
        keep = not ((is_vision and cfg.full_freeze_vision)
                    or (not is_vision and not is_merger and cfg.full_freeze_llm))
        param.requires_grad_(keep)
        if not keep:
            frozen += param.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, frozen


def find_lora_targets(model, cfg: Config) -> list:
    """Every adaptable Linear, filtered by which tower we want to train."""
    import torch.nn as nn

    try:
        import bitsandbytes as bnb

        linear_types = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except Exception:
        linear_types = (nn.Linear,)

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        low = name.lower()
        if "lm_head" in low or "embed" in low or "head.out" in low:
            continue
        is_merger = "merger" in low or "projector" in low or "mm_projector" in low
        is_vision = ("visual" in low or "vision" in low) and not is_merger
        if is_vision and not cfg.train_vision_tower:
            continue
        if is_merger and not cfg.train_merger:
            continue
        targets.append(name)
    return targets


def build_model(cfg: Config, adapter_path: str | Path | None = None,
                for_training: bool = True, verbose: bool = True):
    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    attn = resolve_attn(cfg)
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"

    # A full fine-tune saves whole weights, so a checkpoint replaces the backbone
    # rather than being layered on top of it.
    full_ckpt = (adapter_path is not None and not is_adapter_dir(adapter_path))
    t0 = time.time()
    model = _from_pretrained(cfg, quant, attn, device_map,
                             source=str(adapter_path) if full_ckpt else None)
    if verbose:
        src = f"{adapter_path} (full weights)" if full_ckpt else cfg.model_id
        print(f"[model] {src} | attn={attn} | 4bit={cfg.load_in_4bit} "
              f"| loaded in {time.time()-t0:.0f}s")

    if adapter_path is not None and not full_ckpt:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=for_training)
        if verbose:
            print(f"[model] adapter <- {adapter_path}")
    elif for_training and cfg.finetune_mode == "full":
        trainable, frozen = set_trainable_towers(model, cfg)
        if verbose:
            note = [n for n, on in (("vision", cfg.full_freeze_vision),
                                    ("llm", cfg.full_freeze_llm)) if on]
            print(f"[model] full fine-tune: {trainable/1e9:.2f}B trainable, "
                  f"{frozen/1e9:.2f}B frozen"
                  + (f" ({', '.join(note)} frozen)" if note else ""))
    elif for_training:
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        targets = find_lora_targets(model, cfg)
        if verbose:
            towers = Counter("vision" if ("visual" in t or "vision" in t) else "language"
                             for t in targets)
            print(f"[model] LoRA on {len(targets)} Linear modules {dict(towers)}")
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            use_rslora=cfg.use_rslora,
            use_dora=cfg.use_dora,
        )
        model = get_peft_model(model, lora)

    if for_training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if verbose and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        model.config.use_cache = True
        model.eval()
    return model


def merged_generation_config(model, processor):
    gc_ = model.generation_config
    gc_.pad_token_id = processor.tokenizer.pad_token_id
    eos = processor.tokenizer.eos_token_id
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = [i for i in {eos, im_end} if isinstance(i, int) and i >= 0]
    gc_.eos_token_id = ids if len(ids) > 1 else (ids[0] if ids else eos)
    return gc_


PROCESSOR = load_processor(CFG)
print(f"[model] processor {type(PROCESSOR).__name__} | attn plan: {resolve_attn(CFG)}")

# %% [markdown]
# ### 9.1 Collator smoke test
#
# Builds one real batch and prints exactly which tokens carry loss. If the decoded
# "supervised span" is not precisely the transcription plus the end-of-turn token,
# stop and fix it here — every downstream number depends on this being right.

# %%
_smoke_ds = LineDataset(train_df.head(4), CFG, train=True, renderer=None, seed=0)
_smoke_collator = OCRCollator(PROCESSOR, train=True)
_smoke_batch = _smoke_collator([_smoke_ds[i] for i in range(2)])

print("batch keys:", {k: tuple(v.shape) for k, v in _smoke_batch.items() if hasattr(v, "shape")})
_ids = _smoke_batch["input_ids"][0]
_labels = _smoke_batch["labels"][0]
_sup = _ids[_labels != -100]
print(f"\nassistant header repr: {_smoke_collator.assistant_header!r}")
print(f"tokens total {len(_ids)} | supervised {len(_sup)} "
      f"({100*len(_sup)/len(_ids):.1f}%)")
print(f"supervised span -> {PROCESSOR.tokenizer.decode(_sup)!r}")
print(f"ground truth    -> {_smoke_ds.texts[0]!r}")
assert 0 < len(_sup) < len(_ids), "loss mask covers everything or nothing"
assert _smoke_ds.texts[0][:20] in PROCESSOR.tokenizer.decode(_sup), "masking is misaligned"
print("\n[collator] masking verified")
del _smoke_batch, _smoke_ds
