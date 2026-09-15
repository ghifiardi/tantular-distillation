"""Validate a reviewed licence evidence record, and hash it. Decides nothing.

    # report only — parse, bind, and print the record's digest
    ./.venv/bin/python src/verify_license_evidence.py muse-glimmer-30b

    # write license.evidence_sha256 into configs/models/<name>.yaml
    ./.venv/bin/python src/verify_license_evidence.py muse-glimmer-30b --write

WHAT THIS TOOL IS NOT. It does not decide whether training on a teacher's
outputs is permitted, and it contains no rule that could. Upstream licences,
model documentation, and usage policies are evidence a reviewer interprets;
none is a machine-readable value this tool can map automatically to the
registry's boolean. The answer is a human judgement. This tool checks that a
human recorded one, that the record is complete and bound to the right
checkpoint, and that the digest in the registry is the digest of THAT record.
A record saying `false` verifies exactly as readily as one saying `true`.

WHAT THE DIGEST COVERS. The whole file, byte for byte, as committed — front
matter and prose together. Hashing only the header would let the reasoning be
rewritten under a digest that still matched, and hashing the upstream LICENSE
would bind a document that does not contain the determination. Any edit to the
sources, the determination, the reviewer, the rationale, or the surrounding
explanation moves the digest and invalidates the registry entry until it is
re-reviewed.

WHAT IT CANNOT CHECK. The source hashes are recorded by the reviewer from the
upstream documents at the pinned commit. This tool is offline: it validates their
SHAPE and that they are bound to a model_id and revision, but it cannot confirm
that they are the hashes of real documents. That confirmation happens once, by a
person, at review time — which is the step the whole record exists to capture.

WHAT --write WILL NOT DO. It never replaces a digest that is already valid. Four
states, decided before anything is written:

  MATCHED      the registry records this exact record -> exit 0
  UNRECORDED   a recognised LICENSE_EVIDENCE_DIGEST_*  -> --write fills it;
               placeholder                                report-only exits 1
  SUPERSEDED   a DIFFERENT valid digest is on file    -> REFUSE, both modes
  UNREADABLE   anything else, or no single field      -> REFUSE

UNRECORDED is deliberately narrow. Only the recognised placeholder form is
writable; arbitrary text, a misspelt placeholder, a truncated digest or an empty
value are UNREADABLE, because a field whose meaning is unknown must not be
overwritten on the assumption that it meant nothing.

SUPERSEDED is the state this tool exists for. A valid digest on file that no
longer matches the record means the reviewed document changed after it was
pinned. Re-pinning it automatically would make a tamper-evidence tool erase the
evidence of tampering. Re-review uses two commits: first commit the changed
record with the digest reset to the recognised placeholder, then run --write
and commit the new digest. The verifier enforces the transition through the
placeholder; Git history and human review enforce the two-commit sequence.

NO NETWORK, NO CREDENTIAL, NO MODEL. Reads two local files and writes at most one.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = 1
RECORD_KEYS = ("schema_version", "registry_model", "model_id", "revision",
               "reviewed_at", "reviewed_by", "sources", "determination")
SOURCE_KEYS = ("path", "sha256")
DETERMINATION_KEYS = ("output_training_permitted", "rationale")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_LINE_RE = re.compile(r"^(\s*evidence_sha256:[ \t]*)(\S+)(.*)$",
                               re.MULTILINE)
# The ONLY non-digest value --write may overwrite. Anything else in that field —
# a typo, a truncated digest, prose, an empty value — is a state nobody
# intended, and guessing that it is "just a placeholder" would let --write
# destroy it. A misspelt LICENSE_EVIDNCE_DIGEST_* does not match, deliberately.
_PLACEHOLDER_EVIDENCE_RE = re.compile(r"^LICENSE_EVIDENCE_DIGEST_[A-Z0-9_]+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# A reviewer is a person who can be asked what they meant. This catches the
# obvious machine byline; it cannot prove humanity, and does not pretend to —
# the commit author and the review itself carry that. Word boundaries matter:
# "Abbott" is a surname, not a bot.
_MACHINE_RE = re.compile(
    r"\b(ai|llm|gpt|chatgpt|claude|copilot|gemini|bot|agent|assistant|model|"
    r"automation|automated|script|generated|tool)\b", re.IGNORECASE)

# An unfinished record must not verify. A rationale is the one field nobody can
# fill in later from the other fields.
_PLACEHOLDER_RE = re.compile(
    r"(TODO|TBD|FIXME|XXX|REPLACE_WITH|PLACEHOLDER|LOREM IPSUM|<[^>\n]+>)",
    re.IGNORECASE)
MIN_RATIONALE_CHARS = 40


def die(message: str, code: int = 2) -> None:
    """code=2 refuses; code=1 is "complete, but not recorded yet", the same
    distinction src/verify_model_identity.py draws for unfilled placeholders."""
    label = "LICENCE EVIDENCE REFUSED" if code == 2 else "LICENCE EVIDENCE NOT PINNED"
    print(f"{label}: {message}", file=sys.stderr)
    raise SystemExit(code)


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys.

    PyYAML silently keeps the last of a repeated key, so a record with two
    `determination:` blocks would verify against one and read as the other.
    """


def _no_duplicate_keys(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.YAMLError(f"duplicate key {key!r} in the record")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def split_record(raw: str) -> tuple[str, str]:
    """Front matter and body. Both must exist; the fence must be exact."""
    if not raw.startswith("---\n"):
        die("record must begin with a '---' front-matter fence on line 1")
    end = raw.find("\n---\n", 3)
    if end == -1:
        die("front matter is never closed by a '---' line")
    return raw[4:end + 1], raw[end + 5:]


def _unquoted_number(value: object) -> bool:
    """A digest or commit of only digits parses as a YAML int, not a string.

    Rare for a sha256 and rarer still for a commit, but the failure would be
    baffling — a correct-looking value rejected as "not 64 hex characters" —
    so name the cause instead of the symptom.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        die(f"{field} must be a non-empty string")
    return value.strip()


def parse_record(path: Path) -> dict:
    """Parse and validate the record in isolation, before any registry."""
    if not path.is_file():
        die(f"no evidence record at {path}")
    raw_bytes = path.read_bytes()
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        die(f"{path} is not valid UTF-8")

    front, body = split_record(raw)
    try:
        data = yaml.load(front, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        die(f"front matter is not valid YAML: {exc}")
    if not isinstance(data, dict):
        die("front matter must be a mapping")

    missing = [k for k in RECORD_KEYS if k not in data]
    if missing:
        die(f"front matter is missing required field(s): {', '.join(missing)}")
    unknown = [k for k in data if k not in RECORD_KEYS]
    if unknown:
        die(f"front matter has unknown field(s): {', '.join(sorted(unknown))}. "
            "A typo that is silently ignored is a field nobody reviewed.")

    if data["schema_version"] != SCHEMA_VERSION:
        die(f"schema_version must be {SCHEMA_VERSION}, got "
            f"{data['schema_version']!r}")

    registry_model = _require_str(data["registry_model"], "registry_model")
    model_id = _require_str(data["model_id"], "model_id")

    revision = data["revision"]
    if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
        hint = (" — quote it: a value of only digits parses as a YAML number"
                if _unquoted_number(revision) else "")
        die(f"revision must be exactly 40 lowercase hex characters, got "
            f"{revision!r}{hint}. A licence is reviewed against a specific "
            "checkpoint.")

    reviewed_at = data["reviewed_at"]
    if isinstance(reviewed_at, _dt.date) and not isinstance(reviewed_at, _dt.datetime):
        reviewed_date = reviewed_at
    else:
        try:
            reviewed_date = _dt.date.fromisoformat(str(reviewed_at))
        except (ValueError, TypeError):
            die(f"reviewed_at must be an ISO date, got {reviewed_at!r}")

    reviewed_by = _require_str(data["reviewed_by"], "reviewed_by")
    machine = _MACHINE_RE.search(reviewed_by)
    if machine:
        die(f"reviewed_by {reviewed_by!r} reads as machine-authored "
            f"({machine.group(0)!r}). A licence determination is a human "
            "judgement; no model or agent may sign one.")

    sources = data["sources"]
    if not isinstance(sources, list) or not sources:
        die("sources must be a non-empty list: a determination reached from no "
            "document is not a review")
    seen_paths: set[str] = set()
    parsed_sources = []
    for index, source in enumerate(sources):
        where = f"sources[{index}]"
        if not isinstance(source, dict):
            die(f"{where} must be a mapping with {' and '.join(SOURCE_KEYS)}")
        missing = [k for k in SOURCE_KEYS if k not in source]
        if missing:
            die(f"{where} is missing {', '.join(missing)}")
        extra = [k for k in source if k not in SOURCE_KEYS]
        if extra:
            die(f"{where} has unknown field(s): {', '.join(sorted(extra))}")
        source_path = _require_str(source["path"], f"{where}.path")
        if source_path in seen_paths:
            die(f"{where}.path {source_path!r} is listed twice; two hashes for "
                "one document cannot both be the one that was reviewed")
        seen_paths.add(source_path)
        digest = source["sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            hint = (" — quote it: a value of only digits parses as a YAML number"
                    if _unquoted_number(digest) else "")
            die(f"{where}.sha256 must be exactly 64 lowercase hex characters, "
                f"got {digest!r}{hint}")
        parsed_sources.append({"path": source_path, "sha256": digest})

    determination = data["determination"]
    if not isinstance(determination, dict):
        die("determination must be a mapping")
    missing = [k for k in DETERMINATION_KEYS if k not in determination]
    if missing:
        die(f"determination is missing {', '.join(missing)}")
    extra = [k for k in determination if k not in DETERMINATION_KEYS]
    if extra:
        die(f"determination has unknown field(s): {', '.join(sorted(extra))}")

    permitted = determination["output_training_permitted"]
    if permitted is not True and permitted is not False:
        die("determination.output_training_permitted must be exactly true or "
            f"false, got {permitted!r}. An unanswered question is not an answer, "
            "and a string is not a decision.")

    rationale = _require_str(determination["rationale"], "determination.rationale")
    placeholder = _PLACEHOLDER_RE.search(rationale)
    if placeholder:
        die(f"determination.rationale still contains a placeholder "
            f"({placeholder.group(0)!r})")
    if len(rationale) < MIN_RATIONALE_CHARS:
        die(f"determination.rationale is {len(rationale)} characters; a "
            f"reviewed determination needs at least {MIN_RATIONALE_CHARS}. "
            "Someone will have to act on this reasoning without you.")

    if not body.strip():
        die("the record has no body. The front matter is the summary; the body "
            "is where a reader learns what was actually examined.")

    return {
        "path": path,
        "schema_version": SCHEMA_VERSION,
        "registry_model": registry_model,
        "model_id": model_id,
        "revision": revision,
        "reviewed_at": reviewed_date.isoformat(),
        "reviewed_by": reviewed_by,
        "sources": parsed_sources,
        "output_training_permitted": permitted,
        "rationale": rationale,
        # The WHOLE committed file. Not the front matter, not the LICENSE.
        "record_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def bind_to_spec(record: dict, spec: dict, name: str) -> None:
    """The record must be about the checkpoint the registry actually names.

    The Muse drafter publishes a LICENSE and USAGE_POLICY.md byte-identical to
    its parent's, so source hashes alone cannot tell the two apart. Only the
    model_id and revision can, which is why a mismatch here is fatal rather
    than advisory.
    """
    problems = []
    if record["registry_model"] != name:
        problems.append(f"registry_model {record['registry_model']!r} is not "
                        f"the entry being verified ({name!r})")
    if record["model_id"] != spec.get("model_id"):
        problems.append(f"model_id {record['model_id']!r} != registry "
                        f"{spec.get('model_id')!r}")
    if record["revision"] != spec.get("revision"):
        problems.append(f"revision {record['revision']!r} != registry "
                        f"{spec.get('revision')!r}")
    license_block = spec.get("license") or {}
    registry_reviewed_at = license_block.get("reviewed_at")
    if record["reviewed_at"] != str(registry_reviewed_at):
        problems.append(
            f"reviewed_at {record['reviewed_at']!r} != registry "
            f"{str(registry_reviewed_at)!r}. Freshness is computed from the "
            "registry date, so it must be the date of this reviewed record.")
    declared = license_block.get("output_training_permitted")
    if declared is not record["output_training_permitted"]:
        problems.append(
            f"determination.output_training_permitted "
            f"{record['output_training_permitted']!r} != registry "
            f"{declared!r}. Align the registry with the reviewed record before "
            "writing evidence for it: evidence that contradicts the claim it is "
            "filed against is worse than no evidence.")
    if problems:
        die("record does not bind to the registry entry:\n  - "
            + "\n  - ".join(problems))


MATCHED = "MATCHED"            # the registry already records this record
UNRECORDED = "UNRECORDED"      # no evidence yet: a placeholder, or empty
SUPERSEDED = "SUPERSEDED"      # a DIFFERENT valid digest is already on file
UNREADABLE = "UNREADABLE"      # nowhere safe to read or write the field


def evidence_lines(model_path: Path) -> int:
    return len(_EVIDENCE_LINE_RE.findall(
        model_path.read_text(encoding="utf-8")))


def classify_registry_evidence(spec: dict, model_path: Path, digest: str) -> str:
    """What the registry currently says, relative to this record.

    The dangerous state is SUPERSEDED: a valid digest is already recorded and
    the record now hashes to something else. That means the reviewed document
    changed after it was pinned. Silently re-pinning it would make this tool the
    opposite of tamper-evidence — it would erase the only signal that the
    decision on file is no longer the decision that was reviewed. So it refuses,
    in report-only mode as well as under --write, and a person clears the field
    as a deliberate, visible part of the re-review.
    """
    license_block = spec.get("license")
    if not isinstance(license_block, dict) or "evidence_sha256" not in license_block:
        return UNREADABLE
    current = license_block["evidence_sha256"]
    if not isinstance(current, str):
        return UNREADABLE
    if evidence_lines(model_path) != 1:
        return UNREADABLE
    if _SHA256_RE.fullmatch(current):
        return MATCHED if current == digest else SUPERSEDED
    if _PLACEHOLDER_EVIDENCE_RE.fullmatch(current):
        return UNRECORDED
    return UNREADABLE


def write_evidence(model_path: Path, digest: str) -> None:
    """Replace the evidence digest in place, leaving the file otherwise byte
    identical. A YAML round-trip would drop the comments that explain the
    entry, and those comments are half of why the registry is readable."""
    text = model_path.read_text(encoding="utf-8")
    found = _EVIDENCE_LINE_RE.findall(text)
    if len(found) != 1:
        die(f"expected exactly one evidence_sha256 line in {model_path.name}, "
            f"found {len(found)}")
    text = _EVIDENCE_LINE_RE.sub(
        lambda m: f"{m.group(1)}{digest}{m.group(3)}", text, count=1)
    model_path.write_text(text, encoding="utf-8")


def record_path_for(name: str) -> Path:
    return ROOT / "docs" / "licences" / f"{name}.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a reviewed licence evidence record and hash it. "
                    "Decides no licence question.")
    parser.add_argument("registry_model")
    parser.add_argument("--record", type=Path, default=None,
                        help="defaults to docs/licences/<registry_model>.md")
    parser.add_argument("--write", action="store_true",
                        help="write license.evidence_sha256 into the registry "
                             "entry, only after a clean match")
    args = parser.parse_args(argv)

    name = args.registry_model
    model_path = ROOT / "configs" / "models" / f"{name}.yaml"
    if not model_path.is_file():
        die(f"no registry entry at {model_path}")
    spec = yaml.safe_load(model_path.read_text(encoding="utf-8")) or {}

    record = parse_record(args.record or record_path_for(name))
    bind_to_spec(record, spec, name)

    print(f"record            {record['path']}")
    print(f"registry_model    {record['registry_model']}")
    print(f"model_id          {record['model_id']}")
    print(f"revision          {record['revision']}")
    print(f"reviewed_at       {record['reviewed_at']}")
    print(f"reviewed_by       {record['reviewed_by']}")
    print(f"determination     output_training_permitted="
          f"{str(record['output_training_permitted']).lower()}")
    for source in record["sources"]:
        print(f"source            {source['sha256']}  {source['path']}")
    print(f"record_sha256     {record['record_sha256']}")

    state = classify_registry_evidence(spec, model_path, record["record_sha256"])
    print(f"registry_state    {state}")

    if state == UNREADABLE:
        current = (spec.get("license") or {}).get("evidence_sha256", "<missing>")
        die(f"{model_path.name} has no readable license.evidence_sha256 to act "
            f"on: {current!r}. It is neither a 64-character sha256 nor a "
            "recognised LICENSE_EVIDENCE_DIGEST_* placeholder, so its meaning "
            "is unknown and --write will not overwrite it. Fix the field by "
            "hand.")

    if state == SUPERSEDED:
        current = spec["license"]["evidence_sha256"]
        die(f"{model_path.name} already records a different valid digest.\n"
            f"  recorded  {current}\n"
            f"  record    {record['record_sha256']}\n"
            "The reviewed document changed after it was pinned. Overwriting the "
            "recorded digest would erase the only evidence that the decision on "
            "file is no longer the decision that was reviewed. Re-review in two "
            "commits: first commit the changed record with "
            "license.evidence_sha256 reset to the recognised placeholder; then "
            "run --write and commit the new digest.")

    if state == MATCHED:
        print("registry          already records this record; nothing to do")
        return 0

    if args.write:
        write_evidence(model_path, record["record_sha256"])
        print(f"WROTE             license.evidence_sha256 in {model_path.name}")
        return 0

    # The record is complete and valid, but the registry still records no
    # evidence, so the licence gate still refuses this teacher. A verifier that
    # exits 0 against an unresolved registry reads as "verified" in a script.
    die(f"{model_path.name} records no evidence yet (placeholder "
        f"{spec['license']['evidence_sha256']!r}). The record is valid; re-run "
        "with --write to pin it.", code=1)


if __name__ == "__main__":
    raise SystemExit(main())
