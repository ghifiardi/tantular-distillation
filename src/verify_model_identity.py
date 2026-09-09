"""Fill the model registry's placeholder digests from a REAL local snapshot.

    # report only — says what the digests are and whether they match
    ./.venv/bin/python src/verify_model_identity.py qwen35-9b-instruct --offline

    # fill configs/models/<name>.yaml and set digests_verified: true
    ./.venv/bin/python src/verify_model_identity.py qwen35-9b-instruct --write

configs/models/*.yaml ship with placeholders (TOKENIZER_DIGEST_QWEN35_9B) and
`digests_verified: false`. That flag is load-bearing: src/distill_plan.py treats
the tokenizer sha256 as the COMPATIBILITY KEY that decides whether token-level
KL (Mode C) is even possible. A guessed digest would make that decision on
fiction, so this tool only ever copies a value it measured, and refuses
otherwise.

WHAT IT DIGESTS, and why the two digests are kept independent:

  tokenizer.sha256      every tokenizer file present in the snapshot, name and
                        bytes. tokenizer_config.json is digested with its
                        `chat_template` key REMOVED — otherwise editing the
                        prompt template would move the compatibility key and a
                        Mode C decision would flip for a reason that has nothing
                        to do with the vocabulary.

  chat_template.sha256  the EFFECTIVE template: chat_template.jinja when the
                        snapshot has one (it wins in current HF layouts),
                        otherwise tokenizer_config.json's `chat_template`.

NO NETWORK, NO CREDENTIAL. The snapshot must already be in the local Hugging
Face cache (or named with --snapshot). Nothing here reads HF_TOKEN or the
stored login, and --offline pins HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE for any
library that might later be imported here.

WHY THE CACHE PATH AND NOT AN ARBITRARY DIRECTORY. A digest is only evidence if
you can say which commit it came from. The cache layout records that
(models--org--name/snapshots/<commit>), a loose directory does not — so
--snapshot on a plain directory reports the digests and REFUSES to mark the spec
verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_ids                                            # noqa: E402

# Every file that can change how text becomes tokens. Absent ones are skipped;
# their absence is itself part of the digest because the name is only mixed in
# when the file exists.
TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
CHAT_TEMPLATE_FILE = "chat_template.jinja"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def die(msg: str, code: int = 2) -> None:
    print(f"\nIDENTITY UNVERIFIED: {msg}", file=sys.stderr)
    sys.exit(code)


# --- snapshot resolution ----------------------------------------------------

def hf_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cache_repo_id(snapshot: Path) -> str | None:
    """The repo a cache directory belongs to, e.g. models--Qwen--Qwen3.5-9B."""
    repo_dir = snapshot.parent.parent
    if snapshot.parent.name == "snapshots" and repo_dir.name.startswith("models--"):
        return repo_dir.name[len("models--"):].replace("--", "/")
    return None


def cache_commit(snapshot: Path) -> str | None:
    """The commit a directory stands for, or None if it does not say.

    Only the cache layout .../snapshots/<40-hex>/ answers this. Everything else
    is a directory of files with no provenance.
    """
    parent = snapshot.parent
    if parent.name == "snapshots" and COMMIT_RE.match(snapshot.name):
        return snapshot.name
    return None


def resolve_from_cache(model_id: str, revision: str | None) -> tuple[Path, str]:
    repo_dir = hf_cache_root() / ("models--" + str(model_id).replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        die(f"no local snapshot of {model_id!r} under {hf_cache_root()}.\n"
            f"Obtain it first, then re-run:\n"
            f"    hf download {model_id} --revision <commit>\n"
            "or point at an existing copy with --snapshot. This tool will not "
            "download anything itself.")
    available = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
    if revision and COMMIT_RE.match(revision):
        target = snapshots / revision
        if not target.is_dir():
            die(f"the registry pins {model_id!r} at {revision}, which is not in "
                f"the local cache (have: {', '.join(available) or 'none'}). "
                "Fetch that exact commit; do not verify against a different one.")
        return target, revision
    if len(available) != 1:
        die(f"{model_id!r} has {len(available)} cached snapshots "
            f"({', '.join(available) or 'none'}) and the registry revision is "
            "not a pinned commit. Pin tokenizer.revision, or pass --snapshot.")
    return snapshots / available[0], available[0]


# --- digests ----------------------------------------------------------------

def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def read_tokenizer_config(snapshot: Path) -> dict:
    path = snapshot / "tokenizer_config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"tokenizer_config.json is not readable JSON: {path}: {exc}")
    if not isinstance(payload, dict):
        die(f"tokenizer_config.json must be a JSON object: {path}")
    return payload


def tokenizer_digest(snapshot: Path) -> tuple[str, list[str]]:
    """sha256 over the tokenizer files, excluding the chat template."""
    present = [name for name in TOKENIZER_FILES if (snapshot / name).is_file()]
    if not ({"tokenizer.json", "tokenizer.model"} & set(present)):
        die(f"{snapshot} holds no tokenizer.json or tokenizer.model; this is not "
            "a tokenizer snapshot.")
    sha = hashlib.sha256()
    for name in present:                      # TOKENIZER_FILES is sorted
        if name == "tokenizer_config.json":
            config = dict(read_tokenizer_config(snapshot))
            config.pop("chat_template", None)
            payload = canonical_json(config)
        else:
            payload = (snapshot / name).read_bytes()
        sha.update(name.encode())
        sha.update(b"\x00")
        sha.update(hashlib.sha256(payload).digest())
    return sha.hexdigest(), present


def chat_template_digest(snapshot: Path) -> tuple[str, str]:
    """(digest, where it came from). Refuses when there is no template."""
    jinja = snapshot / CHAT_TEMPLATE_FILE
    if jinja.is_file():
        return hashlib.sha256(jinja.read_bytes()).hexdigest(), CHAT_TEMPLATE_FILE
    template = read_tokenizer_config(snapshot).get("chat_template")
    if isinstance(template, str) and template.strip():
        return (hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "tokenizer_config.json:chat_template")
    if isinstance(template, list) and template:
        # Multiple named templates (e.g. default + tool use): digest all of them.
        return (hashlib.sha256(canonical_json(template)).hexdigest(),
                "tokenizer_config.json:chat_template[]")
    die(f"{snapshot} has no chat template (no {CHAT_TEMPLATE_FILE}, no "
        "chat_template in tokenizer_config.json). A base checkpoint without a "
        "template cannot serve the product's chat path; see "
        "train/BASE_VS_INSTRUCT.md.")
    raise AssertionError("unreachable")      # die() exits


# --- the registry spec ------------------------------------------------------

def spec_path(name: str) -> Path:
    path = ROOT / "configs" / "models" / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (ROOT / "configs" / "models").glob("*.yaml"))
        die(f"no such registry model: {name!r} (have: {', '.join(available) or 'none'})")
    return path


def is_pinned_digest(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.match(value.strip()))


def set_scalar(lines: list[str], path: tuple[str, ...], value: str) -> list[str]:
    """Replace one scalar in place, keeping every comment and blank line.

    Rewriting the file through yaml.dump would delete the comments, and in these
    configs the comments carry the reasoning (why bf16 only, why FP8 is
    impossible on Ampere). So the edit is textual and narrow.
    """
    depth = 0
    out = list(lines)
    index = 0
    while depth < len(path) and index < len(out):
        want = path[depth]
        indent = "  " * depth
        head = f"{indent}{want}:"
        if out[index].startswith(head):
            if depth == len(path) - 1:
                _, _, rest = out[index].partition(":")
                comment = ""
                stripped = rest.strip()
                if "#" in rest and not stripped.startswith("#"):
                    comment = "   #" + rest.split("#", 1)[1]
                elif stripped.startswith("#"):
                    comment = "   " + stripped
                # An all-digit commit or digest would otherwise be read back
                # as a YAML 1.1 integer.
                written = f'"{value}"' if value.isdigit() else value
                out[index] = f"{indent}{want}: {written}{comment}".rstrip()
                return out
            depth += 1
        index += 1
    die(f"could not find {'.'.join(path)} in the spec to update; fix it by hand")
    raise AssertionError("unreachable")


# --- main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="registry name under configs/models/ (no .yaml)")
    parser.add_argument("--snapshot", default=None,
                        help="explicit local snapshot directory; only a Hugging "
                             "Face cache path names its commit, and only a named "
                             "commit can mark the spec verified")
    parser.add_argument("--offline", action="store_true",
                        help="pin HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE; no network "
                             "call is made either way")
    parser.add_argument("--write", action="store_true",
                        help="fill the digests and set digests_verified: true, "
                             "ONLY on a clean match")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    path = spec_path(args.model)
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(spec, dict):
        die(f"{path} must be a mapping")

    tokenizer = spec.get("tokenizer") or {}
    model_id = tokenizer.get("model_id") or spec.get("model_id")
    if not model_id:
        die(f"{args.model!r} declares no tokenizer.model_id or model_id")
    declared_revision = tokenizer.get("revision") or spec.get("revision") or ""
    if declared_revision and not isinstance(declared_revision, str):
        # YAML 1.1 reads an all-digit commit as an integer, which drops its
        # leading zeros and its identity. Refuse rather than resolve something
        # else and call it verified.
        die(f"revision {declared_revision!r} was parsed as {type(declared_revision).__name__}, "
            "not a string; quote it in the spec.")
    pinned_revision = str(declared_revision)

    if args.snapshot:
        snapshot = Path(args.snapshot).expanduser()
        if not snapshot.is_dir():
            die(f"no such snapshot directory: {snapshot}")
        commit = cache_commit(snapshot)
        found_id = cache_repo_id(snapshot)
        if found_id and not model_ids.matches(model_id, found_id):
            # Qwen3.5-9B and Qwen3.5-9B-Base are one character apart and are not
            # the same product (train/BASE_VS_INSTRUCT.md). Exact ids only.
            die(f"--snapshot points at {found_id!r} but {args.model!r} declares "
                f"{model_id!r}. Verifying one checkpoint's digests into another's "
                "spec is exactly the drift this registry exists to prevent.")
    else:
        snapshot, commit = resolve_from_cache(model_id, pinned_revision)

    tok_sha, files = tokenizer_digest(snapshot)
    tpl_sha, tpl_source = chat_template_digest(snapshot)

    declared_tok = str(tokenizer.get("sha256") or "")
    declared_tpl = str((spec.get("chat_template") or {}).get("sha256") or "")

    print(f"model        {args.model}  ({model_id})")
    print(f"snapshot     {snapshot}")
    print(f"commit       {commit or 'UNKNOWN (not a Hugging Face cache path)'}")
    print(f"files        {', '.join(files)}")
    print(f"template     {tpl_source}")
    print(f"tokenizer    {tok_sha}")
    print(f"template sha {tpl_sha}")

    problems: list[str] = []
    for label, declared, measured in (("tokenizer.sha256", declared_tok, tok_sha),
                                      ("chat_template.sha256", declared_tpl, tpl_sha)):
        if is_pinned_digest(declared) and declared.strip() != measured:
            problems.append(f"{label} MISMATCH: registry says {declared.strip()}, "
                            f"snapshot is {measured}")
    if problems:
        die("\n  ".join(["the snapshot does not match the registry:"] + problems)
            + "\nEither the wrong checkpoint is cached or the registry is wrong. "
              "Nothing was written.")

    unfilled = [label for label, declared in (("tokenizer.sha256", declared_tok),
                                              ("chat_template.sha256", declared_tpl))
                if not is_pinned_digest(declared)]

    if not args.write:
        if unfilled:
            die(f"{', '.join(unfilled)} still hold placeholders. Re-run with "
                "--write to record the measured digests.", code=1)
        print("\nverified: both digests match the registry.")
        if spec.get("digests_verified") is not True:
            die("digests match but digests_verified is not true; re-run with "
                "--write to set it.", code=1)
        return

    if commit is None:
        die("--write needs a snapshot whose commit is known. A loose directory "
            "records no revision, so the digest could not be traced back to a "
            "checkpoint. Use the Hugging Face cache copy, or fetch the pinned "
            "commit.")
    if pinned_revision and COMMIT_RE.match(pinned_revision) and pinned_revision != commit:
        die(f"the registry pins {pinned_revision} but the snapshot is {commit}.")

    lines = path.read_text(encoding="utf-8").splitlines()
    lines = set_scalar(lines, ("tokenizer", "sha256"), tok_sha)
    lines = set_scalar(lines, ("chat_template", "sha256"), tpl_sha)
    if not COMMIT_RE.match(pinned_revision):
        # A digest without the commit it came from is not evidence.
        lines = set_scalar(lines, ("revision",), commit)
        lines = set_scalar(lines, ("tokenizer", "revision"), commit)
    lines = set_scalar(lines, ("digests_verified",), "true")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}: digests_verified: true at {commit}")


if __name__ == "__main__":
    main()
