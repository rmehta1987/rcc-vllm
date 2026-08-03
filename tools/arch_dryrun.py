#!/usr/bin/env python
"""Offline architecture-resolution dry-run for a staged model.

Proves — WITHOUT reserving a GPU — that the installed vLLM + transformers can:
  1. see the model's architecture in vLLM's registry,
  2. import the vLLM model class that implements it (catches broken impls),
  3. parse the model's config.json (catches transformers version / config-class
     mismatches, e.g. the qwen3_5_moe class rename in transformers 5.x),
  4. load the tokenizer.

This is the check that would have caught the GLM-5.2 / Qwen3.5-122B "architecture
not supported" failure before an H200 node was ever reserved. It does NOT load
weights and does NOT initialize CUDA, so it runs on a CPU-only build node.

Usage:  python arch_dryrun.py /path/to/model_dir
Exit 0 only if all four checks pass.
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # local dir only; no network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def main(model_dir: str) -> int:
    cfg_path = os.path.join(model_dir, "config.json")
    cfg = json.load(open(cfg_path))
    archs = cfg.get("architectures") or []
    print(f"model dir     : {model_dir}")
    print(f"model_type    : {cfg.get('model_type')}")
    print(f"architectures : {archs}")

    import vllm
    import transformers
    print(f"vllm          : {vllm.__version__}")
    print(f"transformers  : {transformers.__version__}")

    ok = True

    # (1) registry membership
    from vllm import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    for a in archs:
        hit = a in supported
        ok = ok and hit
        print(f"[1] registry  : {a} -> {'SUPPORTED' if hit else 'NOT SUPPORTED'}")

    # (2) import the vLLM model class impl (surfaces broken impls / import errors).
    # 0.26.0 API: ModelRegistry.models maps arch -> _LazyRegisteredModel with
    # load_model_cls(); resolve_model_cls() now needs a full ModelConfig, which we
    # avoid so no CUDA/engine init happens on this CPU-only node.
    try:
        loaded = [a for a in archs if a in ModelRegistry.models]
        target = loaded[0] if loaded else archs[0]
        cls = ModelRegistry.models[target].load_model_cls()
        print(f"[2] import    : imported {getattr(cls, '__name__', cls)} (arch={target})")
    except Exception as e:
        ok = False
        print(f"[2] import    : FAILED -> {type(e).__name__}: {e}")

    # (3) parse config via transformers (catches config-class / version mismatch)
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        sub = getattr(c, "text_config", None)
        print(f"[3] config    : parsed {type(c).__name__}"
              + (f" (text_config={type(sub).__name__})" if sub is not None else ""))
    except Exception as e:
        ok = False
        print(f"[3] config    : FAILED -> {type(e).__name__}: {e}")

    # (4) tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"[4] tokenizer : loaded {type(tok).__name__} (vocab~{tok.vocab_size})")
    except Exception as e:
        ok = False
        print(f"[4] tokenizer : FAILED -> {type(e).__name__}: {e}")

    print("RESULT        :", "PASS — loadable offline (GPU load still to be tested)"
          if ok else "FAIL — see failed checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: arch_dryrun.py /path/to/model_dir", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
