"""Pin a harness's system-prompt identity from a real file, or refuse.

    # report only
    ./.venv/bin/python src/verify_harness_identity.py tantular-office-current

    # measure and pin
    ./.venv/bin/python src/verify_harness_identity.py tantular-office-current --write

`validate_harness` reports "system prompt identity is unverified" for every
harness whose prompts.system carries no digest, and both shipped harnesses are
in that state. This is the only thing allowed to clear it.

WHY THE PROMPT IS PART OF THE HARNESS. The harness digest is supposed to answer
"which scaffolding produced this trace?" — tools, verification, memory,
execution limits. The system prompt belongs in that list: two runs of "the same"
harness with different prompts are different experiments, and a digest that
ignores the prompt cannot tell them apart. Pinning it is what makes the digest
mean what it claims.

WHAT IT WILL NOT DO. It does not guess. A declared path that is absent — which
is the normal state here, because the add-in lives in a sibling repository that
is often not checked out — exits non-zero and leaves the config untouched. A
prompt that changed after being pinned is a MISMATCH and is refused, never
silently re-pinned: re-pinning would let a harness change identity without
anyone deciding that it should.

A directory path is digested as a TREE (sorted relative paths and bytes),
because the shipped harnesses point at ../tantular_office_addin/src rather than
a single file, and the prompt there is assembled from several modules.

No network, no credential, no model.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = ROOT / "configs" / "harnesses"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def die(msg: str, code: int = 2) -> None:
    print(f"\nHARNESS IDENTITY UNVERIFIED: {msg}", file=sys.stderr)
    sys.exit(code)


def digest_path(path: Path) -> str:
    """sha256 of a file, or of a directory tree's names and bytes."""
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    sha = hashlib.sha256()
    for entry in sorted(p for p in path.rglob("*") if p.is_file()):
        sha.update(str(entry.relative_to(path)).encode())
        sha.update(b"\x00")
        sha.update(hashlib.sha256(entry.read_bytes()).digest())
    return sha.hexdigest()


def set_scalar(lines: list[str], path: tuple[str, ...], value: str) -> list[str]:
    """Replace one scalar in place, keeping comments and layout.

    Rewriting through yaml.dump would delete the comments, and in these configs
    the comments carry the reasoning about what the harness may do.
    """
    out = list(lines)
    depth = 0
    index = 0
    while depth < len(path) and index < len(out):
        indent = "  " * depth
        head = f"{indent}{path[depth]}:"
        if out[index].startswith(head):
            if depth == len(path) - 1:
                _, _, rest = out[index].partition(":")
                comment = ""
                stripped = rest.strip()
                if "#" in rest and not stripped.startswith("#"):
                    comment = "   #" + rest.split("#", 1)[1]
                elif stripped.startswith("#"):
                    comment = "   " + stripped
                out[index] = f"{indent}{path[depth]}: {value}{comment}".rstrip()
                return out
            depth += 1
        index += 1
    die(f"could not find {'.'.join(path)} in the harness file; fix it by hand")
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("harness", help="name under configs/harnesses/ (no .yaml)")
    parser.add_argument("--write", action="store_true",
                        help="pin the measured digest and set verified: true, "
                             "ONLY on a clean measurement")
    args = parser.parse_args()

    path = HARNESS_DIR / f"{args.harness}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in HARNESS_DIR.glob("*.yaml"))
        die(f"no such harness: {args.harness!r} (have: {', '.join(available) or 'none'})")

    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    system = (spec.get("prompts") or {}).get("system") or {}
    declared = system.get("path")
    if not declared:
        die(f"{args.harness!r} declares no prompts.system.path, so there is "
            "nothing to measure. A harness with no prompt identity cannot be "
            "verified; give it a path or leave it unverified honestly.")

    prompt_path = Path(str(declared)).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = ROOT / prompt_path
    if not prompt_path.exists():
        die(f"no prompt at {prompt_path}.\n"
            f"{args.harness!r} declares prompts.system.path: {declared}\n"
            "That path is usually a sibling checkout of the Office add-in. "
            "Obtain it and re-run; nothing is guessed in its absence.")

    measured = digest_path(prompt_path)
    pinned = system.get("sha256")

    print(f"harness   {args.harness}")
    print(f"prompt    {prompt_path}  ({'directory tree' if prompt_path.is_dir() else 'file'})")
    print(f"measured  {measured}")
    print(f"pinned    {pinned or '(none)'}")

    if isinstance(pinned, str) and SHA256_RE.match(pinned.strip()):
        if pinned.strip() != measured:
            die("prompt MISMATCH: the harness pins a different digest.\n"
                f"  pinned   {pinned.strip()}\n"
                f"  measured {measured}\n"
                "Either the prompt changed or the pin is wrong. Decide which, "
                "deliberately; this will not re-pin on its own.")
        if system.get("verified") is not True and not args.write:
            die("the digest matches but verified is not true; re-run with "
                "--write to record it.", code=1)
        print("\nverified: the prompt matches the pinned digest.")
        if system.get("verified") is True:
            return

    if not args.write:
        die("prompts.system.sha256 is not pinned. Re-run with --write to "
            "record the measured digest.", code=1)

    lines = path.read_text(encoding="utf-8").splitlines()
    lines = set_scalar(lines, ("prompts", "system", "sha256"), measured)
    lines = set_scalar(lines, ("prompts", "system", "verified"), "true")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT) if ROOT in path.parents else path}: "
          "prompts.system.verified: true")


if __name__ == "__main__":
    main()
