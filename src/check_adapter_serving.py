"""Step 6 of the smoke test: prove the adapter id actually generates.

    ./.venv/bin/python src/check_adapter_serving.py --adapter-id tantular-smoke

Exists as a file rather than a shell one-liner because pasting long commands
into a pod terminal mangles them, and a mangled proof is worse than none.

WHAT IT PROVES, and why each part is separate:

  the id is served     /v1/models lists it. Necessary, nowhere near sufficient:
                       an id can be listed and still answer with nothing.
  it generates         non-empty, non-degenerate text. This is the check that
                       matters. The FP8 arm's endpoint listed its model, came up
                       healthy, and emitted only a control token — for two
                       rentals.
  it is not the base    the adapter's output differs from the base's. Reported,
                       not gated: a one-step adapter over four examples may well
                       decode identically under greedy sampling. Identical text
                       is a caveat to record, not a failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

PROMPT = ("Ringkas kalimat berikut menjadi satu kalimat: Rapat anggaran "
          "ditunda ke pekan depan karena data realisasi belum lengkap.")
DEGENERATE = re.compile(r"^(?:(.{1,20}?)\1{4,})$", re.DOTALL)


def post(base: str, model: str, max_tokens: int) -> dict:
    body = json.dumps({"model": model, "prompt": PROMPT,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{base}/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()[:400]}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def text_of(payload: dict) -> str | None:
    if "choices" not in payload:
        return None
    return payload["choices"][0].get("text")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8020/v1")
    p.add_argument("--adapter-id", default="tantular-smoke")
    p.add_argument("--base-id", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    print("=== SERVED MODELS ===")
    try:
        with urllib.request.urlopen(f"{args.base_url}/models", timeout=60) as r:
            served = [m.get("id") for m in json.load(r).get("data", [])]
    except Exception as e:
        sys.exit(f"cannot reach {args.base_url}: {e}")
    for s in served:
        print(f"  {s}")
    if args.adapter_id not in served:
        sys.exit(f"\nFAIL: {args.adapter_id!r} is not served. Serve the base with "
                 "--enable-lora and --lora-modules <id>=<path>.")
    if args.base_id not in served:
        print(f"\n  NOTE: base id {args.base_id!r} not in the list; "
              "the comparison below will be skipped.")

    print(f"\n=== GENERATION [{args.adapter_id}] ===")
    a = post(args.base_url, args.adapter_id, args.max_tokens)
    ta = text_of(a)
    if ta is None:
        sys.exit(f"FAIL: no completion returned.\n  {json.dumps(a)[:500]}")
    print(f"  {ta!r}")

    if not ta.strip():
        sys.exit("\nFAIL: the completion is EMPTY. The endpoint answers and "
                 "produces no text — the failure mode that survived two rentals "
                 "on the FP8 arm. Do not proceed to v1.")
    if DEGENERATE.match(ta.strip()):
        sys.exit(f"\nFAIL: the completion is a repeating loop: {ta[:120]!r}")

    tb = None
    if args.base_id in served:
        print(f"\n=== GENERATION [{args.base_id}] ===")
        tb = text_of(post(args.base_url, args.base_id, args.max_tokens))
        print(f"  {tb!r}")

    print("\n=== RESULT ===")
    print(f"  adapter id served     : yes")
    print(f"  adapter produced text : yes ({len(ta.strip())} chars)")
    if tb is not None:
        differs = ta != tb
        print(f"  differs from base     : {'yes' if differs else 'NO'}")
        if not differs:
            print("    Identical output. Recorded, not failed: a one-step "
                  "adapter over four\n    examples can decode identically under "
                  "greedy sampling. It does mean this\n    run did NOT prove the "
                  "two ids address different weights.")
    print("\nSTEP 6 PASSED. Stop the pod (step 7).")
    print("This authorises NO training run. That is a separate decision.")


if __name__ == "__main__":
    main()
