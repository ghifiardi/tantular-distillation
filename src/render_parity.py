"""Establish prompt-rendering parity between arms before the FP8 comparison.

    python3 src/render_parity.py probe   --host ai19-ollama
    python3 src/render_parity.py verify  --host ai19-ollama --canonical <name>

WHY. The int4 arm runs on Ollama, whose model card carries
`TEMPLATE {{ .Prompt }}` — a passthrough — and whose GGUF has no embedded
chat_template. So Ollama serializes `messages` into a prompt by an internal
default we cannot read. A vLLM FP8 arm would apply the chat template from
tokenizer_config.json instead. Comparing those two directly measures prompt
serialization AND quantization at once, and no verdict could separate them.

APPROACH. Rather than detect the difference and hope it is small, remove it:
render the prompt ourselves and send the identical string to both arms through
a completion path (`/api/generate` with raw=true on Ollama, `/v1/completions`
on vLLM). Parity then holds by construction, and the hash proves it.

`probe` first recovers what Ollama's chat endpoint actually does, by running
candidate renderings through the raw path at temperature 0 and comparing each
output against the chat endpoint's. A candidate that reproduces the chat output
exactly is Ollama's serialization — which makes the normalized protocol
behaviourally equivalent to the deployment baseline, not a new configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install -r requirements.txt")

from config import resolve

ROOT = Path(__file__).resolve().parent.parent
PARITY_DIR = ROOT / "calibration" / "parity"

# A deterministic, short-answer probe: the router task constrains output to one
# token from a closed set, so two renderings agreeing is signal rather than
# coincidence of a long free-text generation.
PROBE_SYSTEM = (
    "Klasifikasikan permintaan pengguna ke salah satu intent: TANYA_DOKUMEN, "
    "EDIT_TEKS, DRAFT_TEKS, TERJEMAH, RINGKAS, UBAH_NADA, CEK_AMAN, UMUM. "
    "Jawab HANYA nama intent."
)
PROBE_USER = "Terjemahkan bagian executive summary ini ke bahasa Inggris."

# Candidate serializations. Named so a result can be recorded and reproduced.
CANDIDATES = {
    "sys_blank_user":   lambda s, u: f"{s}\n\n{u}",
    "sys_nl_user":      lambda s, u: f"{s}\n{u}",
    "labelled_turns":   lambda s, u: f"System: {s}\nUser: {u}\nAssistant:",
    "user_only":        lambda s, u: u,
    # Harmony-style control tokens. The first probe showed the model emitting
    # `<|start|>assistant to=self<|message|>` unprompted, which is the tell that
    # its native format is harmony — and that Ollama renders it with a built-in
    # architecture renderer rather than the Modelfile template.
    "harmony":          lambda s, u: (
        f"<|start|>system<|message|>{s}<|end|>"
        f"<|start|>user<|message|>{u}<|end|>"
        f"<|start|>assistant"),
    "harmony_message":  lambda s, u: (
        f"<|start|>system<|message|>{s}<|end|>"
        f"<|start|>user<|message|>{u}<|end|>"
        f"<|start|>assistant<|message|>"),
    "harmony_nosys":    lambda s, u: (
        f"<|start|>user<|message|>{s}\n\n{u}<|end|>"
        f"<|start|>assistant<|message|>"),
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ollama_raw(base: str, model: str, prompt: str, timeout: float) -> str:
    """Send an exact string, bypassing all templating."""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "raw": True, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 600, "seed": 0}},
        )
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


def chat(base_v1: str, model: str, system: str, user: str, timeout: float) -> str:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base_v1}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0.0, "max_tokens": 600, "seed": 0},
        )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--host", required=True)
        p.add_argument("--teacher", default="muse-glimmer")
        if name == "verify":
            p.add_argument("--canonical", required=True, choices=sorted(CANDIDATES))
    args = parser.parse_args()

    resolved = resolve(args.teacher, args.host)
    base_v1 = resolved["HOST_BASE_URL"]
    if not base_v1:
        sys.exit(f"host '{args.host}' has no base_url; parity probing needs an HTTP endpoint")
    base = base_v1.rsplit("/v1", 1)[0]
    model = resolved["TEACHER_REPO"]
    timeout = float(resolved["HOST_REQUEST_TIMEOUT_S"])

    if args.command == "verify":
        rendered = CANDIDATES[args.canonical](PROBE_SYSTEM, PROBE_USER)
        print(f"canonical   {args.canonical}")
        print(f"sha256[:16] {digest(rendered)}")
        print(f"chars       {len(rendered)}")
        print("\nSend this exact string to BOTH arms:")
        print("  ollama : POST /api/generate  {raw: true, prompt: <string>}")
        print("  vllm   : POST /v1/completions {prompt: <string>}")
        print("Parity then holds by construction; the hash above is the proof.")
        return

    print(f"probing {model} at {base}\n")
    reference = chat(base_v1, model, PROBE_SYSTEM, PROBE_USER, timeout)
    print(f"chat endpoint output: {reference[:80]!r}\n")

    results = {}
    for name, render in CANDIDATES.items():
        rendered = render(PROBE_SYSTEM, PROBE_USER)
        try:
            output = ollama_raw(base, model, rendered, timeout)
        except httpx.HTTPStatusError as error:
            print(f"  {name:<18} ERROR {error.response.status_code}")
            continue
        match = output == reference
        results[name] = {"sha256": digest(rendered), "output": output, "matches_chat": match}
        print(f"  {name:<18} sha={digest(rendered)}  match={'YES' if match else 'no '}  "
              f"out={output[:44]!r}")

    matching = [n for n, r in results.items() if r["matches_chat"]]
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARITY_DIR / f"{args.host}.probe.json"
    out_path.write_text(json.dumps(
        {"host": args.host, "model": model, "chat_reference": reference,
         "candidates": results, "matching": matching}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nrecorded -> {out_path}")

    if matching:
        print(f"\nOllama's chat serialization matches: {matching}")
        print("Use that as the canonical rendering. The normalized protocol is then "
              "behaviourally equivalent to the deployment baseline, so the existing "
              "52-prompt int4 result carries over rather than needing a re-run.")
    else:
        print("\nNo candidate reproduced the chat endpoint's output.")
        print("Ollama's serialization is something else. Under the normalized "
              "protocol both arms still receive identical bytes, but the int4 arm "
              "MUST be re-run: its existing traces came from a different rendering. "
              "Keep the existing result as the deployment baseline; use the "
              "normalized re-run for the int4-vs-FP8 verdict.")


if __name__ == "__main__":
    main()
