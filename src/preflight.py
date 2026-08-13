"""Verify a serving endpoint before a study run, and pin its signature.

    python3 src/preflight.py --teacher muse-glimmer --host ai19-ollama \
        --record data/calibration/int4/signature.json
    python3 src/preflight.py --teacher muse-glimmer --host ai19-ollama \
        --verify data/calibration/int4/signature.json

A model disappeared out from under this study once already: the gateway kept
advertising `ollama/muse-glimmer-30b` in /v1/models while returning HTTP 400
for it, and 52 calls failed. Worse than failing loudly would be a model being
silently *replaced* — the run completes, the numbers look fine, and they
describe different weights than the ones being compared.

So each arm records a signature once, and every later run verifies against it.
A mismatch fails the run rather than producing results nobody can interpret.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install -r requirements.txt")

from config import base_url, resolve


def probe(resolved: dict, api_key: str) -> dict:
    # A cold model load can take minutes: an 18GB 30B evicted from VRAM has to
    # come back off disk before it answers. Use the host's own request budget
    # rather than a short default, or preflight fails on a healthy endpoint.
    timeout = float(resolved.get("HOST_REQUEST_TIMEOUT_S") or 600)
    url = resolved["HOST_BASE_URL"] or base_url(resolved)
    model = resolved["TEACHER_REPO"] if resolved["HOST_BASE_URL"] \
        else resolved["TEACHER_SERVED_MODEL_NAME"]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with httpx.Client(timeout=timeout) as client:
        try:
            listing = client.get(f"{url}/models", headers=headers)
        except httpx.TransportError as error:
            sys.exit(f"PREFLIGHT FAIL — cannot reach {url}: {error!r}")
        if listing.status_code != 200:
            sys.exit(f"PREFLIGHT FAIL — {url}/models returned {listing.status_code}: "
                     f"{listing.text[:200]}")
        available = [m.get("id") for m in (listing.json().get("data") or [])]
        if model not in available:
            sys.exit(
                f"PREFLIGHT FAIL — model {model!r} is not served at {url}.\n"
                f"available ({len(available)}): {', '.join(map(str, available[:8]))}"
            )

        # Listing a model is not the same as serving it — the gateway proved
        # exactly that. Actually call it.
        probe_response = client.post(
            f"{url}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 16, "temperature": 0.0},
        )
    if probe_response.status_code != 200:
        sys.exit(f"PREFLIGHT FAIL — {model} listed but not callable "
                 f"({probe_response.status_code}): {probe_response.text[:200]}")

    payload = probe_response.json()
    return {
        "endpoint": url,
        "requested_model": model,
        "reported_model": payload.get("model"),
        "quantization": resolved["HOST_QUANTIZATION"],
        "host": resolved["HOST_NAME"],
        "teacher": resolved["TEACHER_NAME"],
        "models_available": sorted(str(m) for m in available),
    }


# Fields that must not drift between recording and a later run. `endpoint` is
# excluded deliberately: a tunnel may bind a different local port, which does
# not change what is being measured.
PINNED = ("requested_model", "reported_model", "quantization", "host", "teacher")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--record", type=Path, help="write the signature here")
    parser.add_argument("--verify", type=Path, help="fail if it differs from here")
    args = parser.parse_args()

    resolved = resolve(args.teacher, args.host)
    api_key = ""
    if resolved["HOST_API_KEY_ENV"]:
        api_key = os.environ.get(resolved["HOST_API_KEY_ENV"], "")

    signature = probe(resolved, api_key)

    if args.verify:
        if not args.verify.exists():
            sys.exit(f"PREFLIGHT FAIL — no recorded signature at {args.verify}")
        recorded = json.loads(args.verify.read_text(encoding="utf-8"))
        drift = {k: (recorded.get(k), signature.get(k))
                 for k in PINNED if recorded.get(k) != signature.get(k)}
        if drift:
            print("PREFLIGHT FAIL — serving signature changed since it was recorded:")
            for field, (was, now) in drift.items():
                print(f"  {field}: recorded {was!r} -> now {now!r}")
            sys.exit("Results from this endpoint are not comparable to the recorded arm.")
        print(f"PREFLIGHT OK — signature matches {args.verify}")

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"recorded signature -> {args.record}")

    for field in PINNED:
        print(f"  {field:<18} {signature[field]}")


if __name__ == "__main__":
    main()
