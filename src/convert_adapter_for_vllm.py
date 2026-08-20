"""Rewrite a PEFT adapter's keys so vLLM can bind it to Qwen3.5.

    ./.venv/bin/python src/convert_adapter_for_vllm.py \
        /workspace/runs/smoke/adapter /workspace/runs/smoke/adapter-vllm

WHY THIS IS NEEDED, established on the smoke rental 2026-08-19/20:

  training  AutoModelForCausalLM resolves Qwen3_5ForCausalLM, whose .model is a
            Qwen3_5TextModel holding .layers directly. PEFT writes
            base_model.model.model.layers.N.…
  serving   vLLM resolves Qwen3_5ForConditionalGeneration, whose .model is a
            Qwen3_5Model holding .visual and .language_model, and whose weights
            on disk are named model.language_model.…

The adapter is missing one path segment. vLLM loads it, matches nothing, binds
nothing, and reports success — the endpoint then serves BASE answers under the
adapter's id, which is indistinguishable from a working adapter unless you
compare logprobs.

Overriding the served architecture instead does NOT work: --hf-overrides swaps
the class but not the weight-name mapping, and the base model then fails to
load. So the adapter is what gets rewritten.

This changes NAMES ONLY. Every tensor is copied byte-identically, and the copy
is verified before the output is written.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Everything under the text model needs the segment; nothing else is touched.
NEEDLE = "base_model.model.model."
INSERT = "base_model.model.model.language_model."


def convert_key(key: str) -> str:
    if not key.startswith(NEEDLE):
        return key
    rest = key[len(NEEDLE):]
    if rest.startswith("language_model."):
        return key                       # already converted; idempotent
    return INSERT + rest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--force", action="store_true",
                   help="overwrite dst if it exists")
    args = p.parse_args()

    src, dst = args.src, args.dst
    if not (src / "adapter_config.json").is_file():
        sys.exit(f"{src} is not a PEFT adapter (no adapter_config.json)")
    weights = src / "adapter_model.safetensors"
    if not weights.is_file():
        sys.exit(f"{src} has no adapter_model.safetensors "
                 "(this converter only handles safetensors)")
    if dst.exists() and not args.force:
        sys.exit(f"{dst} exists; pass --force to overwrite")

    from safetensors.torch import load_file, save_file
    tensors = load_file(str(weights))

    converted, untouched = {}, 0
    for key, value in tensors.items():
        new = convert_key(key)
        if new == key:
            untouched += 1
        converted[new] = value

    changed = len(tensors) - untouched
    print(f"=== KEY CONVERSION ===")
    print(f"  tensors    {len(tensors)}")
    print(f"  rewritten  {changed}")
    print(f"  untouched  {untouched}")
    if changed == 0:
        sys.exit("nothing was rewritten. Either this adapter is already "
                 "converted, or its keys do not start with "
                 f"{NEEDLE!r} and this converter does not understand it.")
    if len(converted) != len(tensors):
        sys.exit("key collision during conversion — refusing to write a "
                 "checkpoint that silently lost tensors")

    sample = sorted(converted)[0]
    print(f"  example    {sample}")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    save_file(converted, str(dst / "adapter_model.safetensors"))

    # Names changed; VALUES must not have. Verify rather than assert.
    check = load_file(str(dst / "adapter_model.safetensors"))
    if set(check) != set(converted):
        sys.exit("the written file does not have the keys we wrote")
    for key, value in converted.items():
        if not check[key].equal(value):
            sys.exit(f"tensor {key} changed on write — refusing to vouch for it")
    print(f"  verified   all {len(check)} tensors byte-identical after write")

    cfg = json.loads((dst / "adapter_config.json").read_text())
    cfg["_converted_for_vllm"] = {
        "from": str(src),
        "reason": "inserted language_model. segment; see "
                  "calibration/SMOKE_RESULT.md",
    }
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"\nwrote {dst}")
    print("Serve THIS directory, and confirm with src/check_adapter_serving.py "
          "that the\nlogprobs differ from the base's. Being served is not being "
          "applied.")


if __name__ == "__main__":
    main()
