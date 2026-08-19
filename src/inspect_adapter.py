"""Is this adapter trained, and can vLLM even apply it?

    ./.venv/bin/python src/inspect_adapter.py /workspace/runs/smoke/adapter

Written because the smoke adapter produced logprobs bit-identical to the base's
across every token, which has two very different explanations:

  the adapter is untrained   LoRA B matrices initialise to ZERO, so an adapter
                             that never received an update has exactly zero
                             effect. Bit-identical output is what that looks like.
  vLLM ignored it            the id is served, requests succeed, and the LoRA is
                             never applied. Same symptom, completely different fix.

Reading lora_B decides it. Non-zero B means the checkpoint carries real training
and the serving side is at fault; all-zero B means the training side is.

Also prints the target modules, because a LoRA whose targets vLLM does not
support for this architecture is applied to nothing while still loading cleanly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: inspect_adapter.py <adapter-dir>")
    d = Path(sys.argv[1])
    cfg_path = d / "adapter_config.json"
    if not cfg_path.is_file():
        sys.exit(f"no adapter_config.json in {d}")
    cfg = json.loads(cfg_path.read_text())
    print("=== ADAPTER CONFIG ===")
    for k in ("peft_type", "r", "lora_alpha", "base_model_name_or_path",
              "target_modules", "task_type"):
        print(f"  {k:<26} {cfg.get(k)}")

    weights = next((d / n for n in ("adapter_model.safetensors", "adapter_model.bin")
                    if (d / n).is_file()), None)
    if weights is None:
        sys.exit("no adapter weights in the directory")

    from safetensors.torch import load_file
    tensors = load_file(str(weights))
    a_keys = [k for k in tensors if "lora_A" in k]
    b_keys = [k for k in tensors if "lora_B" in k]
    print(f"\n=== WEIGHTS ({weights.name}) ===")
    print(f"  tensors {len(tensors)}   lora_A {len(a_keys)}   lora_B {len(b_keys)}")

    def summarise(keys, label):
        if not keys:
            print(f"  {label}: NONE FOUND")
            return 0.0
        peak = 0.0
        nonzero = 0
        for k in keys:
            m = tensors[k].abs().max().item()
            peak = max(peak, m)
            nonzero += int(m > 0)
        print(f"  {label}: max|w| {peak:.6g}   non-zero tensors {nonzero}/{len(keys)}")
        return peak

    summarise(a_keys, "lora_A")
    peak_b = summarise(b_keys, "lora_B")

    # The names matter as much as the values. vLLM binds a LoRA by module path;
    # on a multimodal architecture the language model sits under a prefix, and a
    # checkpoint whose paths do not match what vLLM patches LOADS CLEANLY AND
    # BINDS TO NOTHING. That is indistinguishable from an untrained adapter at
    # the API, which is how this went unnoticed until logprobs were compared.
    print("\n=== MODULE PATHS (first 5) ===")
    for k in sorted(tensors)[:5]:
        print(f"  {k}")
    prefixes = sorted({k.split(".layers.")[0] for k in tensors if ".layers." in k})
    print(f"\n  distinct prefixes before '.layers.': {prefixes}")

    print("\n=== VERDICT ===")
    if peak_b == 0.0:
        print("  lora_B is ALL ZERO — this adapter is UNTRAINED.")
        print("  B initialises to zero and stays there until an optimizer step")
        print("  reaches it, so this checkpoint provably has no effect on any")
        print("  output. The fault is on the TRAINING side: the weights that were")
        print("  updated are not the weights that were saved.")
        sys.exit(1)
    print(f"  lora_B carries real values (max |w| {peak_b:.6g}).")
    print("  The checkpoint IS trained, so bit-identical serving output means")
    print("  vLLM is not applying it. The fault is on the SERVING side —")
    print("  most likely LoRA is unsupported for this architecture, or the")
    print("  target modules are not ones vLLM patches.")


if __name__ == "__main__":
    main()
