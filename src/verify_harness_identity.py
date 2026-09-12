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
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = ROOT / "configs" / "harnesses"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# The add-in publishes a short djb2 hash, not a sha256; accepted for
# cross-checking only, never as the identity itself.
HEX_RE = re.compile(r"^[0-9a-f]{1,64}$")


def die(msg: str, code: int = 2) -> None:
    print(f"\nHARNESS IDENTITY UNVERIFIED: {msg}", file=sys.stderr)
    sys.exit(code)


def digest_path(path: Path) -> str:
    """sha256 of a file, or of a directory tree's names and bytes.

    Retained for a harness whose prompt really is a file or a directory. It is
    NOT the right answer for the Office add-in: see prompt_registry_digest.
    """
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    entries = sorted(p for p in path.rglob("*") if p.is_file())
    if not entries:
        # The sha256 of an empty traversal is a perfectly stable number that
        # means nothing. Presenting it as a verified prompt identity would be
        # the exact failure this tool exists to prevent.
        die(f"{path} is an empty directory; there is no prompt here to verify.")
    sha = hashlib.sha256()
    for entry in entries:
        sha.update(str(entry.relative_to(path)).encode())
        sha.update(b"\x00")
        sha.update(hashlib.sha256(entry.read_bytes()).digest())
    return sha.hexdigest()


# Read the add-in's own registry rather than reimplementing it. The registry is
# the module that already enumerates every production prompt and owns each
# one's content hash, so this asks it, in its own runtime, and never parses
# JavaScript from Python.
_REGISTRY_PROBE = """
import { allPromptIds, getPrompt } from %s;
const ids = allPromptIds();
if (!Array.isArray(ids) || ids.length === 0) {
  throw new Error("allPromptIds() returned no prompt ids");
}
const rows = [...ids].sort().map((id) => {
  const entry = getPrompt(id);
  if (!entry || typeof entry.contentHash !== "string" || !entry.contentHash) {
    throw new Error(`prompt ${id} has no contentHash`);
  }
  if (typeof entry.content !== "string" || entry.content.length === 0) {
    throw new Error(`prompt ${id} has no content`);
  }
  return { id, contentHash: entry.contentHash, content: entry.content };
});
process.stdout.write(JSON.stringify(rows));
"""


def prompt_registry_digest(path: Path) -> tuple[str, list[dict]]:
    """Digest the EFFECTIVE prompts, via the add-in's own prompt registry.

    Digesting ../tantular_office_addin/src as a tree answers "did any
    JavaScript change?", which is both too sensitive and not sensitive enough:
    an unrelated edit moves the digest while the prompts are identical, and a
    prompt moved between modules leaves it unchanged. The registry knows which
    strings are actually prompts and publishes each one's content hash, so the
    identity is computed over exactly those.

    Fails closed on every step that could silently produce a smaller answer:
    no node, no registry file, a missing export, an empty id list, or an entry
    without a hash.
    """
    if not path.is_file():
        die(f"no prompt registry at {path}. The harness declares "
            "source: prompt_registry, so this file must exist; nothing is "
            "inferred in its absence.")
    if shutil.which("node") is None:
        die("node is required to read the add-in's prompt registry, and is not "
            "installed. Failing closed rather than guessing the prompt "
            "identity from the file's bytes.")

    probe = _REGISTRY_PROBE % json.dumps(str(path))
    proc = subprocess.run(
        ["node", "--input-type=module", "--eval", probe],
        capture_output=True, text=True, cwd=path.parent, timeout=120)
    if proc.returncode != 0:
        die("the add-in prompt registry could not be read:\n"
            f"{(proc.stderr or proc.stdout).strip()[-600:]}")
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        die(f"the prompt registry produced no readable JSON: {exc}")
    if not isinstance(rows, list) or not rows:
        die("the prompt registry reported no prompts")
    # WE hash the content; we do not inherit the add-in's hash. Its hashText is
    # a djb2 32-bit value (~8 hex chars), which is fine for the add-in's own
    # cache-busting but is not a commitment: it is short enough to collide and
    # is not collision-resistant by construction. Digesting it would make the
    # harness prompt identity only as strong as that. The registry's own value
    # is kept alongside, so a mismatch between the two is visible, but the
    # identity is sha256 over the prompt text.
    seen: set[str] = set()
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            die(f"malformed prompt registry entry: {row!r}")
        if row["id"] in seen:
            die(f"the prompt registry reports {row['id']!r} more than once; one "
                "prompt would silently shadow another.")
        seen.add(row["id"])
        content = row.get("content")
        if not isinstance(content, str) or not content:
            die(f"prompt {row['id']!r} has no content to hash")
        registry_hash = row.get("contentHash")
        if not isinstance(registry_hash, str) or not HEX_RE.match(registry_hash):
            die(f"prompt {row['id']!r} has a contentHash that is not lowercase "
                f"hex: {registry_hash!r}")
        # id and content travel in ONE object, so the aggregate binds them:
        # swapping text between two prompt ids changes the identity, which a
        # digest over two independent lists would not catch.
        normalized.append({
            "id": row["id"],
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "registry_content_hash": registry_hash,
        })

    canonical = json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), normalized


# A Git object id, complete and lowercase: 40 hex for SHA-1, 64 for SHA-256.
# An abbreviation is a display convenience, not an identity -- two commits can
# share a short prefix, so an abbreviated pin does not name one snapshot.
GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """One read-only git command against a checkout the caller supplied.

    Every call here inspects; none fetches. Acquisition and verification are
    deliberately separate operations: a verifier that could fetch could be
    talked into auditing a snapshot other than the one it was asked about.
    """
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, timeout=60)


def _normalized_url(url: str) -> str:
    """Only .git / trailing-slash spelling is normalised, never identity."""
    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/")


def resolve_repository_source(name: str, system: dict, checkout: str | None) -> Path:
    """Resolve a repository-backed prompt path inside a supplied checkout.

    The prompts live in another repository. Naming a path on this machine made
    the identity depend on whoever ran the verifier; pinning url + tag + the
    exact commit that tag peels to makes it depend on the snapshot instead.
    This confirms the checkout IS that snapshot before a single prompt is read,
    so bytes that happen to hash correctly cannot stand in for the pinned
    source.
    """
    repository = system.get("repository") or {}
    if not checkout:
        die(f"{name!r} declares a repository-backed prompt source, so it can only "
            "be measured against a checkout you supply:\n"
            "  --source-checkout <path to a clean checkout of the pinned commit>\n"
            "Nothing is resolved from a sibling directory: a path that happens to "
            "exist on this machine is not the pinned snapshot.")

    for key in ("url", "ref", "peeled_commit"):
        value = repository.get(key)
        if not isinstance(value, str) or not value.strip():
            die(f"prompts.system.repository.{key} is required for a "
                "repository-backed prompt source; a partial pin names no snapshot.")
    commit = repository["peeled_commit"].strip()
    ref = repository["ref"].strip()
    if not GIT_OID_RE.match(commit):
        die("prompts.system.repository.peeled_commit must be a complete lowercase "
            f"Git object id (40 hex for SHA-1, 64 for SHA-256), not {commit!r}. "
            "A prompt digest is SHA-256; a Git object id is a different type.")

    root = Path(checkout).expanduser()
    if not root.is_dir():
        die(f"--source-checkout {root} is not a directory.")
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        die(f"--source-checkout {root} is not a git worktree, so its contents "
            "cannot be tied to the pinned commit.")
    root = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip())

    fmt = (_git(root, "rev-parse", "--show-object-format").stdout.strip() or "sha1")
    expected = 64 if fmt == "sha256" else 40
    if len(commit) != expected:
        die(f"prompts.system.repository.peeled_commit has {len(commit)} hex "
            f"characters, but this repository uses {fmt} object ids "
            f"({expected} characters).")

    origin = _git(root, "remote", "get-url", "origin")
    if origin.returncode != 0 or not origin.stdout.strip():
        die("the supplied checkout has no origin remote, so it cannot be shown "
            f"to be a checkout of {repository['url']}.")
    if _normalized_url(origin.stdout) != _normalized_url(repository["url"]):
        die("the supplied checkout's origin is a different repository:\n"
            f"  pinned   {repository['url']}\n"
            f"  checkout {origin.stdout.strip()}")

    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        die("the supplied checkout is at a different commit: its HEAD is\n"
            f"  {head}\n"
            f"but the harness pins peeled_commit\n  {commit}\n"
            "Check out the pinned commit; nothing is measured from another one.")

    peeled = _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if peeled.returncode != 0 or not peeled.stdout.strip():
        die(f"the supplied checkout does not contain {ref!r}, so the pinned tag "
            "cannot be confirmed. Clone the tag itself rather than a branch.")
    if peeled.stdout.strip() != commit:
        die(f"{ref!r} peels to a different commit:\n"
            f"  peels to {peeled.stdout.strip()}\n"
            f"  pinned   {commit}\n"
            "A tag name is not an identity. Re-pointing one must not silently "
            "become a new prompt identity.")

    dirty = _git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        die("the supplied checkout is not clean: tracked files differ from the "
            f"pinned commit.\n{dirty[:400]}\n"
            "The commit says what the source is; edited bytes are not it.")

    declared = system.get("path")
    if not isinstance(declared, str) or not declared.strip():
        die(f"{name!r} declares no prompts.system.path to resolve inside the "
            "pinned repository.")
    relative = PurePosixPath(declared.strip())
    if relative.is_absolute() or ".." in relative.parts:
        die(f"prompts.system.path must stay inside the pinned checkout, but "
            f"{declared!r} is absolute or points outside it. Reading outside "
            "would let the pin name one snapshot and measure another.")
    resolved = (root / relative).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        die(f"prompts.system.path resolves outside the pinned checkout: "
            f"{resolved}")
    return resolved


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
    parser.add_argument("--source-checkout", metavar="PATH",
                        help="a clean local checkout of the commit pinned by "
                             "prompts.system.repository. Required for a "
                             "repository-backed source; this tool reads it and "
                             "never fetches.")
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

    if system.get("repository") is not None:
        prompt_path = resolve_repository_source(
            args.harness, system, args.source_checkout)
    else:
        if args.source_checkout:
            die(f"{args.harness!r} declares no prompts.system.repository, so "
                "--source-checkout has nothing to verify. Remove it, or pin the "
                "harness to a repository snapshot.")
        prompt_path = Path(str(declared)).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = ROOT / prompt_path
        if not prompt_path.exists():
            die(f"no prompt at {prompt_path}.\n"
                f"{args.harness!r} declares prompts.system.path: {declared}\n"
                "That path is usually a sibling checkout of the Office add-in. "
                "Obtain it and re-run; nothing is guessed in its absence.")

    source = str(system.get("source") or "path")
    if source == "prompt_registry":
        measured, rows = prompt_registry_digest(prompt_path)
        shape = f"prompt registry, {len(rows)} prompts"
    elif source == "path":
        measured, rows = digest_path(prompt_path), []
        shape = "directory tree" if prompt_path.is_dir() else "file"
    else:
        die(f"unknown prompts.system.source {source!r} (use prompt_registry or path)")
    pinned = system.get("sha256")

    print(f"harness   {args.harness}")
    print(f"prompt    {prompt_path}  ({shape})")
    for row in rows:
        print(f"            {row['id']:<22} {row['content_sha256'][:16]}")
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
