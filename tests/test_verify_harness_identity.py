"""The harness prompt hash must be measured or refused — never guessed.

    ./.venv/bin/python -m pytest tests/test_verify_harness_identity.py -q

`validate_harness` reports "system prompt identity is unverified" for both
shipped harnesses. That warning is the honest state, and this is the only thing
allowed to clear it: it hashes a real file and writes the result, or it exits
non-zero and leaves the config alone.

Why it matters: the system prompt is part of the harness the same way the tool
policy is. Two runs of "the same" harness with different prompts are different
experiments, and the digest that is supposed to tell them apart cannot, unless
the prompt is pinned.

No add-in, no network, no credential: everything here is tmp files.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import harness_distill as hd                                  # noqa: E402
import verify_harness_identity as vhi                         # noqa: E402

PROMPT = "You are Tantular, an Indonesian office assistant.\n"

SPEC = """\
# A comment that must survive a --write.
schema_version: 1
name: fake-harness
status: candidate

model_contract:
  registry_model: qwen35-9b-instruct
  prompt_format: chat

prompts:
  system:
    path: PROMPT_PATH
    sha256: null
    verified: false

tools:
  allow: [read]
  state_change_requires_approval: true

memory:
  active_context: bounded
  ephemeral_execution: request
  durable_task_state: none
  product_memory: disabled

execution:
  isolation: companion_process
  network: denied_by_default
  max_steps: 4
  max_wall_seconds: 300

verification:
  before_action: [request_schema]
  after_action: [edit_contract]
  repair_attempts: 0

mutation:
  production_self_modify: false
  candidate_workspace_only: true
  evaluator_mutation_allowed: false
  human_approval_required: true
"""


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A registry directory holding one harness whose prompt path is real."""
    prompt = tmp_path / "system.txt"
    prompt.write_text(PROMPT, encoding="utf-8")
    directory = tmp_path / "harnesses"
    directory.mkdir()
    path = directory / "fake-harness.yaml"
    path.write_text(SPEC.replace("PROMPT_PATH", str(prompt)), encoding="utf-8")
    monkeypatch.setattr(hd, "HARNESS_DIR", directory)
    monkeypatch.setattr(vhi, "HARNESS_DIR", directory)
    return path


def run(*argv: str) -> int:
    saved = sys.argv
    sys.argv = ["verify_harness_identity.py", *argv]
    try:
        vhi.main()
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved


# --- the success path -------------------------------------------------------

def test_write_pins_the_measured_hash_and_keeps_the_comments(harness, capsys):
    assert run("fake-harness", "--write") == 0

    spec = yaml.safe_load(harness.read_text())
    system = spec["prompts"]["system"]
    assert system["sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert system["verified"] is True
    assert "# A comment that must survive a --write." in harness.read_text()

    # And the warning it exists to clear is gone.
    assert "system prompt identity is unverified" not in hd.validate_harness(spec)


def test_a_second_run_verifies_rather_than_rewriting(harness):
    assert run("fake-harness", "--write") == 0
    before = harness.read_text()
    assert run("fake-harness") == 0
    assert harness.read_text() == before


# --- the refusals -----------------------------------------------------------

def test_a_missing_prompt_file_refuses_and_writes_nothing(harness, capsys):
    """The add-in sibling is often absent. That must fail closed, not produce a
    hash of nothing."""
    spec = yaml.safe_load(harness.read_text())
    Path(spec["prompts"]["system"]["path"]).unlink()

    assert run("fake-harness", "--write") == 2
    assert "no prompt" in capsys.readouterr().err
    assert yaml.safe_load(harness.read_text())["prompts"]["system"]["verified"] is False


def test_a_changed_prompt_is_a_mismatch_not_a_silent_rewrite(harness, capsys):
    """Once pinned, a different prompt must fail loudly. Quietly re-pinning
    would let the harness change identity without anyone deciding to."""
    assert run("fake-harness", "--write") == 0
    pinned = harness.read_text()

    spec = yaml.safe_load(pinned)
    Path(spec["prompts"]["system"]["path"]).write_text("different\n", encoding="utf-8")

    assert run("fake-harness", "--write") == 2
    assert "MISMATCH" in capsys.readouterr().err
    assert harness.read_text() == pinned


def test_reporting_without_write_leaves_an_unverified_harness_alone(harness, capsys):
    assert run("fake-harness") == 1
    out = capsys.readouterr()
    assert "not pinned" in out.err
    assert yaml.safe_load(harness.read_text())["prompts"]["system"]["verified"] is False


def test_a_harness_declaring_no_prompt_path_refuses(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "harnesses"
    directory.mkdir()
    path = directory / "no-prompt.yaml"
    path.write_text(SPEC.replace("PROMPT_PATH", "null").replace(
        "name: fake-harness", "name: no-prompt"), encoding="utf-8")
    monkeypatch.setattr(vhi, "HARNESS_DIR", directory)
    assert run("no-prompt", "--write") == 2
    assert "declares no prompts.system.path" in capsys.readouterr().err


def test_an_unknown_harness_refuses(harness, capsys):
    assert run("no-such-harness") == 2
    assert "no such harness" in capsys.readouterr().err


def test_a_directory_prompt_path_is_digested_as_a_tree(tmp_path, monkeypatch):
    """The shipped harnesses point at ../tantular_office_addin/src — a
    DIRECTORY. Digest it as a tree rather than refusing, so the real add-in can
    be pinned when it is present."""
    src = tmp_path / "addin-src"
    (src / "chat").mkdir(parents=True)
    (src / "chat" / "prompt.js").write_text("export const P = 'x';\n", encoding="utf-8")
    directory = tmp_path / "harnesses"
    directory.mkdir()
    path = directory / "dir-harness.yaml"
    path.write_text(SPEC.replace("PROMPT_PATH", str(src)).replace(
        "name: fake-harness", "name: dir-harness"), encoding="utf-8")
    monkeypatch.setattr(vhi, "HARNESS_DIR", directory)

    assert run("dir-harness", "--write") == 0
    system = yaml.safe_load(path.read_text())["prompts"]["system"]
    assert vhi.SHA256_RE.match(system["sha256"])
    assert system["verified"] is True

    # A changed file under the tree changes the digest.
    before = system["sha256"]
    (src / "chat" / "prompt.js").write_text("export const P = 'y';\n", encoding="utf-8")
    assert vhi.digest_path(src) != before


# --- the EFFECTIVE prompts, not the whole source tree ------------------------
#
# Digesting ../tantular_office_addin/src as a tree answers "did any JavaScript
# change?" — too sensitive (an unrelated edit moves it) and not sensitive
# enough (a prompt moved between modules leaves it unchanged). The add-in's own
# registry knows which strings are prompts and owns each one's content hash.

REGISTRY_JS = """\
const CONTENT = {
  router: "ROUTER PROMPT",
  edit: "EDIT PROMPT",
};
import { createHash } from "node:crypto";
function hashText(text) {
  return createHash("sha256").update(text, "utf8").digest("hex");
}
export function allPromptIds() { return Object.keys(CONTENT); }
export function getPrompt(id) {
  if (!(id in CONTENT)) throw new Error("unknown prompt id: " + id);
  return { id, content: CONTENT[id], contentHash: hashText(CONTENT[id]) };
}
"""

node = pytest.mark.skipif(__import__("shutil").which("node") is None,
                          reason="node is required to read a prompt registry")


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A fake add-in: a prompt registry plus an unrelated source file."""
    src = tmp_path / "addin-src"
    src.mkdir()
    (src / "promptRegistry.js").write_text(REGISTRY_JS, encoding="utf-8")
    (src / "unrelated.js").write_text("export const X = 1;\n", encoding="utf-8")
    (src / "package.json").write_text('{"type":"module"}', encoding="utf-8")

    directory = tmp_path / "harnesses"
    directory.mkdir()
    path = directory / "reg-harness.yaml"
    path.write_text(
        SPEC.replace("PROMPT_PATH", str(src / "promptRegistry.js"))
            .replace("name: fake-harness", "name: reg-harness")
            .replace("    sha256: null", "    source: prompt_registry\n    sha256: null"),
        encoding="utf-8")
    monkeypatch.setattr(vhi, "HARNESS_DIR", directory)
    return path, src


@node
def test_the_registry_digest_covers_the_prompts(registry):
    path, src = registry
    digest, rows = vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert vhi.SHA256_RE.match(digest)
    assert [r["id"] for r in rows] == ["edit", "router"], "sorted by id"


@node
def test_changing_a_prompt_changes_the_digest(registry):
    path, src = registry
    before, _ = vhi.prompt_registry_digest(src / "promptRegistry.js")
    (src / "promptRegistry.js").write_text(
        REGISTRY_JS.replace('router: "ROUTER PROMPT"', 'router: "ROUTER PROMPT v2"'),
        encoding="utf-8")
    after, _ = vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert after != before


@node
def test_changing_an_unrelated_source_file_does_not(registry):
    """The reason for reading the registry instead of the tree."""
    path, src = registry
    before, _ = vhi.prompt_registry_digest(src / "promptRegistry.js")
    (src / "unrelated.js").write_text("export const X = 999;\n", encoding="utf-8")
    after, _ = vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert after == before


@node
def test_a_registry_missing_an_export_fails_closed(registry, capsys):
    path, src = registry
    (src / "promptRegistry.js").write_text(
        "export function allPromptIds() { return ['a']; }\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert "could not be read" in capsys.readouterr().err


@node
def test_a_registry_reporting_no_prompts_fails_closed(registry, capsys):
    path, src = registry
    (src / "promptRegistry.js").write_text(
        "export function allPromptIds() { return []; }\n"
        "export function getPrompt(id) { return { id, contentHash: 'x' }; }\n",
        encoding="utf-8")
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert "could not be read" in capsys.readouterr().err


@node
def test_a_prompt_without_a_content_hash_fails_closed(registry, capsys):
    path, src = registry
    (src / "promptRegistry.js").write_text(
        "export function allPromptIds() { return ['a']; }\n"
        "export function getPrompt(id) { return { id }; }\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert "could not be read" in capsys.readouterr().err


def test_a_missing_registry_file_fails_closed(tmp_path, capsys):
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(tmp_path / "nope.js")
    assert "no prompt registry" in capsys.readouterr().err


def test_a_missing_node_fails_closed(tmp_path, monkeypatch, capsys):
    """Without node the prompt identity is unknowable; it must not fall back to
    hashing the file's bytes, which would be a different claim."""
    stub = tmp_path / "promptRegistry.js"
    stub.write_text(REGISTRY_JS, encoding="utf-8")
    monkeypatch.setattr(vhi.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(stub)
    assert "node is required" in capsys.readouterr().err


def test_an_empty_directory_is_not_a_verified_prompt_identity(tmp_path, capsys):
    """sha256 of an empty traversal is stable and meaningless."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        vhi.digest_path(empty)
    assert "empty directory" in capsys.readouterr().err


# --- regressions: the registry's rows must be well-formed and unique --------

@node
def test_duplicate_prompt_ids_are_refused(registry, capsys):
    """One prompt would silently shadow another, and the digest would stop
    meaning "these are the prompts"."""
    path, src = registry
    (src / "promptRegistry.js").write_text(
        'export function allPromptIds() { return ["a", "a"]; }\n'
        'export function getPrompt(id) {\n'
        '  return { id, content: "x", contentHash: "abcd1234" };\n'
        '}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert "more than once" in capsys.readouterr().err


@node
def test_a_non_hex_content_hash_is_refused(registry, capsys):
    path, src = registry
    (src / "promptRegistry.js").write_text(
        'export function allPromptIds() { return ["a"]; }\n'
        'export function getPrompt(id) {\n'
        '  return { id, content: "x", contentHash: "NOT-HEX!" };\n'
        '}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert "not lowercase" in capsys.readouterr().err


@node
def test_a_prompt_without_content_is_refused(registry, capsys):
    """The identity is sha256 over the prompt TEXT, so no text means no
    identity — it must not fall back to the registry's own short hash."""
    path, src = registry
    (src / "promptRegistry.js").write_text(
        'export function allPromptIds() { return ["a"]; }\n'
        'export function getPrompt(id) {\n'
        '  return { id, contentHash: "abcd1234" };\n'
        '}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert "could not be read" in capsys.readouterr().err


@node
def test_the_identity_does_not_inherit_the_addin_short_hash(registry):
    """The add-in's hashText is a djb2 32-bit value, not a commitment. Two
    prompts differing only in content must produce different identities even if
    a registry reported the same contentHash for both."""
    path, src = registry
    digest, rows = vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert all(vhi.SHA256_RE.match(r["content_sha256"]) for r in rows)
    assert all("registry_content_hash" in r for r in rows)

    (src / "promptRegistry.js").write_text(
        'import { createHash } from "node:crypto";\n'
        'const C = { a: "one" };\n'
        'export function allPromptIds() { return Object.keys(C); }\n'
        'export function getPrompt(id) {\n'
        '  return { id, content: C[id], contentHash: "deadbeef" };\n'
        '}\n', encoding="utf-8")
    first, _ = vhi.prompt_registry_digest(src / "promptRegistry.js")
    (src / "promptRegistry.js").write_text(
        'const C = { a: "two" };\n'
        'export function allPromptIds() { return Object.keys(C); }\n'
        'export function getPrompt(id) {\n'
        '  return { id, content: C[id], contentHash: "deadbeef" };\n'
        '}\n', encoding="utf-8")
    second, _ = vhi.prompt_registry_digest(src / "promptRegistry.js")
    assert first != second, "identity must follow the content, not the short hash"


@node
def test_the_identity_binds_prompt_id_to_prompt_content(tmp_path):
    """Swapping text between two prompt ids must change the aggregate identity.

    A digest over the ids and a separate digest over the contents would both be
    unchanged by such a swap — the PAIRING is what matters, so each id travels
    in one object with its own content hash.
    """
    template = (
        'export function allPromptIds() { return ["alpha", "beta"]; }\n'
        'const C = { alpha: %s, beta: %s };\n'
        'export function getPrompt(id) {\n'
        '  return { id, content: C[id], contentHash: "deadbeef" };\n'
        '}\n')
    registry = tmp_path / "promptRegistry.js"

    registry.write_text(template % ('"ONE"', '"TWO"'), encoding="utf-8")
    original, rows = vhi.prompt_registry_digest(registry)

    registry.write_text(template % ('"TWO"', '"ONE"'), encoding="utf-8")
    swapped, swapped_rows = vhi.prompt_registry_digest(registry)

    assert original != swapped, "the identity must bind id to content"
    # The same two content hashes appear in both — only the pairing differs.
    assert {r["content_sha256"] for r in rows} == \
        {r["content_sha256"] for r in swapped_rows}


# --- repository-backed prompt sources ----------------------------------------
#
# The prompts live in another repository. Naming a path on this machine made the
# identity unreproducible: whoever ran the verifier decided what "the prompt"
# was. A repository-backed source pins the snapshot instead -- url, tag and the
# exact commit that tag peels to -- and the verifier reads a checkout the caller
# supplies. Acquisition (git) and verification (reading) stay separate: the
# verifier never reaches the network, so it cannot be talked into fetching a
# different snapshot than the one it is auditing.

ADDIN_URL = "https://github.com/ghifiardi/LLM-Indonesia.git"
REGISTRY_RELPATH = "tantular_office_addin/src/promptRegistry.js"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip()


@pytest.fixture
def source_repo(tmp_path):
    """A temporary LOCAL repository shaped like the published add-in.

    No network: the tests build and tag their own fixture, so they assert the
    resolver's rules rather than GitHub's availability.
    """
    repo = tmp_path / "addin-repo"
    (repo / REGISTRY_RELPATH).parent.mkdir(parents=True)
    (repo / REGISTRY_RELPATH).write_text(REGISTRY_JS, encoding="utf-8")
    (repo / "tantular_office_addin" / "src" / "package.json").write_text(
        '{"type":"module"}', encoding="utf-8")
    _git(repo.parent, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "remote", "add", "origin", ADDIN_URL)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")
    _git(repo, "tag", "-a", "baseline-tag", "-m", "baseline tag")
    return SimpleNamespace(
        path=repo,
        commit=_git(repo, "rev-parse", "HEAD"),
        tag="baseline-tag",
        url=ADDIN_URL,
    )


def repo_harness(tmp_path, monkeypatch, source, *, url=None, ref=None,
                 commit=None, path=REGISTRY_RELPATH, sha256="null"):
    """A harness whose prompts.system is repository-backed."""
    block = (
        "    source: prompt_registry\n"
        "    repository:\n"
        f"      url: {url if url is not None else source.url}\n"
        f"      ref: {ref if ref is not None else source.tag}\n"
        f"      peeled_commit: {commit if commit is not None else source.commit}\n"
        f"    path: {path}\n"
        f"    sha256: {sha256}\n"
    )
    spec = SPEC.replace("name: fake-harness", "name: repo-harness")
    spec = spec.replace("    path: PROMPT_PATH\n    sha256: null\n", block)
    directory = tmp_path / "repo-harnesses"
    directory.mkdir(exist_ok=True)
    target = directory / "repo-harness.yaml"
    target.write_text(spec, encoding="utf-8")
    monkeypatch.setattr(hd, "HARNESS_DIR", directory)
    monkeypatch.setattr(vhi, "HARNESS_DIR", directory)
    return target


@node
def test_a_pinned_clean_checkout_is_accepted(tmp_path, monkeypatch, source_repo, capsys):
    path = repo_harness(tmp_path, monkeypatch, source_repo)
    assert run("repo-harness", "--source-checkout", str(source_repo.path)) == 1
    out = capsys.readouterr()
    # measured cleanly; the only complaint is that nothing is pinned yet
    assert "not pinned" in (out.err + out.out)
    assert "router" in out.out and "edit" in out.out


@node
def test_write_records_the_measured_aggregate_for_a_repository_source(
        tmp_path, monkeypatch, source_repo):
    path = repo_harness(tmp_path, monkeypatch, source_repo)
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 0
    system = (yaml.safe_load(path.read_text(encoding="utf-8"))["prompts"])["system"]
    expected, _ = vhi.prompt_registry_digest(source_repo.path / REGISTRY_RELPATH)
    assert system["sha256"] == expected
    assert system["verified"] is True
    # the repository pin itself must survive the write untouched
    assert system["repository"]["peeled_commit"] == source_repo.commit
    assert system["repository"]["ref"] == source_repo.tag
    assert "A comment that must survive a --write." in path.read_text(encoding="utf-8")


@node
def test_a_checkout_at_another_commit_is_refused(tmp_path, monkeypatch, source_repo, capsys):
    """Pinning the wrong snapshot must fail on the commit, before any prompt is
    read: a checkout whose bytes happen to hash correctly is still not the
    pinned source."""
    (source_repo.path / "NOTES.md").write_text("later work\n", encoding="utf-8")
    _git(source_repo.path, "add", "-A")
    _git(source_repo.path, "commit", "-m", "second")
    moved = _git(source_repo.path, "rev-parse", "HEAD")
    assert moved != source_repo.commit
    path = repo_harness(tmp_path, monkeypatch, source_repo)  # still pins the first
    before = path.read_text(encoding="utf-8")
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "HEAD" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == before


@node
def test_a_tag_that_peels_elsewhere_is_refused(tmp_path, monkeypatch, source_repo, capsys):
    """A tag name is not an identity. Re-pointing it at other source must not
    quietly become a new prompt identity."""
    (source_repo.path / "NOTES.md").write_text("later work\n", encoding="utf-8")
    _git(source_repo.path, "add", "-A")
    _git(source_repo.path, "commit", "-m", "second")
    _git(source_repo.path, "tag", "-a", "wandering", "-m", "points at the second commit")
    path = repo_harness(tmp_path, monkeypatch, source_repo,
                        ref="wandering", commit=source_repo.commit)
    _git(source_repo.path, "checkout", "--quiet", source_repo.commit)
    before = path.read_text(encoding="utf-8")
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "peel" in capsys.readouterr().err.lower()
    assert path.read_text(encoding="utf-8") == before


@node
def test_a_missing_tag_is_refused(tmp_path, monkeypatch, source_repo, capsys):
    path = repo_harness(tmp_path, monkeypatch, source_repo, ref="no-such-tag")
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "no-such-tag" in capsys.readouterr().err


@node
def test_a_repository_source_without_a_checkout_argument_is_refused(
        tmp_path, monkeypatch, source_repo, capsys):
    """It must refuse rather than resolve the path against this repository or a
    sibling directory that happens to be lying around."""
    path = repo_harness(tmp_path, monkeypatch, source_repo)
    before = path.read_text(encoding="utf-8")
    assert run("repo-harness", "--write") == 2
    assert "--source-checkout" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == before


@node
def test_tracked_changes_in_the_checkout_are_refused(tmp_path, monkeypatch, source_repo, capsys):
    """The commit says what the source is; a dirty tracked file means the bytes
    on disk are not that commit."""
    (source_repo.path / REGISTRY_RELPATH).write_text(
        REGISTRY_JS.replace("ROUTER PROMPT", "EDITED IN PLACE"), encoding="utf-8")
    path = repo_harness(tmp_path, monkeypatch, source_repo)
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "clean" in capsys.readouterr().err.lower()


@node
def test_untracked_files_in_the_checkout_are_tolerated(tmp_path, monkeypatch, source_repo):
    """node_modules and build output are untracked by nature; refusing them
    would make a real checkout unverifiable for no integrity gain."""
    (source_repo.path / "tantular_office_addin" / "node_modules").mkdir()
    (source_repo.path / "tantular_office_addin" / "node_modules" / "x.js").write_text(
        "//\n", encoding="utf-8")
    path = repo_harness(tmp_path, monkeypatch, source_repo)
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 0


def test_a_missing_registry_in_the_checkout_is_refused(tmp_path, monkeypatch, source_repo, capsys):
    path = repo_harness(tmp_path, monkeypatch, source_repo,
                        path="tantular_office_addin/src/nope.js")
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "nope.js" in capsys.readouterr().err


@pytest.mark.parametrize("escape", ["../outside.js", "/etc/hosts",
                                    "tantular_office_addin/../../outside.js"])
def test_a_path_escaping_the_checkout_is_refused(tmp_path, monkeypatch, source_repo,
                                                 capsys, escape):
    """A repository-relative path is relative to THAT repository. Reading
    outside it would let the pin name one snapshot and measure another."""
    path = repo_harness(tmp_path, monkeypatch, source_repo, path=escape)
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    err = capsys.readouterr().err
    assert "outside" in err.lower() or "absolute" in err.lower()


@node
def test_a_mismatched_pin_is_not_rewritten_under_write(tmp_path, monkeypatch,
                                                       source_repo, capsys):
    """--write records a first measurement; it never overwrites a disagreeing
    one. That transition needs a person."""
    stale = "b" * 64
    path = repo_harness(tmp_path, monkeypatch, source_repo, sha256=stale)
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "MISMATCH" in capsys.readouterr().err
    system = (yaml.safe_load(path.read_text(encoding="utf-8"))["prompts"])["system"]
    assert system["sha256"] == stale
    assert system["verified"] is not True


@node
def test_an_unrelated_origin_is_refused(tmp_path, monkeypatch, source_repo, capsys):
    path = repo_harness(tmp_path, monkeypatch, source_repo,
                        url="https://github.com/someone/unrelated.git")
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "origin" in capsys.readouterr().err.lower()


@node
def test_equivalent_url_spellings_are_accepted(tmp_path, monkeypatch, source_repo):
    """Only .git/trailing-slash spelling is normalised -- never a different
    repository."""
    path = repo_harness(tmp_path, monkeypatch, source_repo,
                        url=ADDIN_URL.removesuffix(".git") + "/")
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 0


@pytest.mark.parametrize("bad", ["3e14d25", "3E14D25468AB0CD793BA8DC48CF5F755796C94E2",
                                 "z" * 40, "3e14d25468ab0cd793ba8dc48cf5f755796c94e2a"])
def test_an_abbreviated_or_malformed_commit_is_refused(tmp_path, monkeypatch,
                                                       source_repo, capsys, bad):
    """A display abbreviation is not an object id. Accepting one would let two
    different commits satisfy the same pin."""
    path = repo_harness(tmp_path, monkeypatch, source_repo, commit=bad)
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 2
    assert "peeled_commit" in capsys.readouterr().err


def test_a_non_repository_directory_is_refused(tmp_path, monkeypatch, source_repo, capsys):
    plain = tmp_path / "not-a-repo"
    (plain / REGISTRY_RELPATH).parent.mkdir(parents=True)
    (plain / REGISTRY_RELPATH).write_text(REGISTRY_JS, encoding="utf-8")
    path = repo_harness(tmp_path, monkeypatch, source_repo)
    assert run("repo-harness", "--source-checkout", str(plain), "--write") == 2
    assert "git" in capsys.readouterr().err.lower()


@node
def test_the_verifier_reads_the_checkout_without_touching_it(tmp_path, monkeypatch, source_repo):
    """Verification is a read. It must not mutate the evidence it is auditing."""
    before = _git(source_repo.path, "status", "--porcelain")
    before_head = _git(source_repo.path, "rev-parse", "HEAD")
    path = repo_harness(tmp_path, monkeypatch, source_repo)
    assert run("repo-harness", "--source-checkout", str(source_repo.path), "--write") == 0
    assert _git(source_repo.path, "status", "--porcelain") == before
    assert _git(source_repo.path, "rev-parse", "HEAD") == before_head


# --- the canonical payload, spelled out --------------------------------------


@node
def test_the_canonical_prompt_payload_is_one_sorted_json_array(registry):
    """The framing is part of the identity.

    The same rows serialised as newline-delimited objects in declaration-key
    order produce a different digest -- that is exactly what happened when the
    add-in baseline was published with a one-off probe, and why the published
    tag annotation records a value this repository does not measure. Assert the
    bytes, not just the hash, so the framing can never again be implicit.
    """
    _, src = registry
    digest, rows = vhi.prompt_registry_digest(src / "promptRegistry.js")

    edit = hashlib.sha256("EDIT PROMPT".encode()).hexdigest()
    router = hashlib.sha256("ROUTER PROMPT".encode()).hexdigest()
    # built by hand, NOT by json.dumps: a bug in the serializer must not be able
    # to agree with itself here.
    expected_bytes = (
        '[{"content_sha256":"%s","id":"edit","registry_content_hash":"%s"},'
        '{"content_sha256":"%s","id":"router","registry_content_hash":"%s"}]'
        % (edit, edit, router, router)
    ).encode("utf-8")

    assert [r["id"] for r in rows] == ["edit", "router"], "rows sort by prompt id"
    assert json.dumps(rows, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8") == expected_bytes
    assert digest == hashlib.sha256(expected_bytes).hexdigest()

    # the framing the publication probe used, for contrast: same rows, different
    # identity. This must NOT be what we compute.
    newline_framed = "\n".join(
        '{"id":%s,"content_sha256":%s,"registry_content_hash":%s}'
        % (json.dumps(r["id"]), json.dumps(r["content_sha256"]),
           json.dumps(r["registry_content_hash"]))
        for r in rows).encode("utf-8")
    assert hashlib.sha256(newline_framed).hexdigest() != digest


# --- the shipped harnesses, pinned to the published baseline -----------------


ADDIN_TAG = "tantular-office-addin-harness-baseline-2026-09-11"
ADDIN_COMMIT = "3e14d25468ab0cd793ba8dc48cf5f755796c94e2"
ADDIN_PROMPT_SHA256 = "1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e"


@pytest.mark.parametrize("name", ["tantular-office-current", "tantular-office-candidate"])
def test_the_shipped_harnesses_pin_the_published_baseline(name):
    """Replaces the three "these stay unverified" assertions.

    They were correct while the add-in was unpublished: a digest of an
    unpublished tree is not an identity anyone else could reproduce. The tree is
    published now, so the honest assertion is the exact pin -- offline, with no
    add-in checkout required.
    """
    system = (hd.load_harness(name).get("prompts") or {}).get("system") or {}
    assert system["source"] == "prompt_registry"
    assert system["path"] == REGISTRY_RELPATH
    assert system["sha256"] == ADDIN_PROMPT_SHA256
    assert system["verified"] is True
    repository = system["repository"]
    assert repository["url"] == ADDIN_URL
    assert repository["ref"] == ADDIN_TAG
    assert repository["peeled_commit"] == ADDIN_COMMIT
    assert len(repository["peeled_commit"]) == 40


def test_the_shipped_harnesses_do_not_pin_the_publication_probe_framing():
    """The tag annotation records 5086061..., measured by the publication
    probe's newline framing. It is evidence of that probe, not this
    repository's prompt identity, and must never reach a harness."""
    probe_framing = "5086061098575be10267c1b4a0676f75ad3e65723b497ca3d06eee6b4aa26794"
    for name in ("tantular-office-current", "tantular-office-candidate"):
        system = (hd.load_harness(name).get("prompts") or {}).get("system") or {}
        assert system["sha256"] != probe_framing


def test_both_shipped_harnesses_share_a_prompt_identity_but_not_a_digest():
    """Same production prompts, different policy. Equal complete digests would
    mean the policy differences were lost."""
    current = hd.load_harness("tantular-office-current")
    candidate = hd.load_harness("tantular-office-candidate")
    assert (current["prompts"]["system"]["sha256"]
            == candidate["prompts"]["system"]["sha256"] == ADDIN_PROMPT_SHA256)
    assert hd.canonical_digest(current) != hd.canonical_digest(candidate)


@pytest.mark.parametrize("name", ["tantular-office-current", "tantular-office-candidate"])
def test_a_verified_prompt_does_not_make_the_harness_executable(name):
    """Prompt identity is not execution attribution: generate.py still has no
    harness executor."""
    spec = hd.load_harness(name)
    assert (spec.get("trace_generation") or {}).get("supported_by_generate_py") is False
    assert hd.validate_harness(spec) == []
