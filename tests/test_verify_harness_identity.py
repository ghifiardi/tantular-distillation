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
import sys
from pathlib import Path

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


def test_the_shipped_harnesses_remain_unverified_here():
    """No add-in snapshot is pinned in this repository, so the real harnesses
    stay unverified. Delete this when they are pinned against the real add-in."""
    for name in ("tantular-office-current", "tantular-office-candidate"):
        system = (hd.load_harness(name).get("prompts") or {}).get("system") or {}
        assert system.get("verified") is not True, name
        assert not system.get("sha256"), name


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
function hashText(text) {
  let h = 0;
  for (const ch of text) { h = (h * 31 + ch.codePointAt(0)) >>> 0; }
  return String(h);
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


# The ONLY test here that needs the real add-in. The synthetic prompt-registry
# tests above need node but not the add-in, and must keep running in CI: they
# are what prove a prompt change moves the digest and an unrelated source change
# does not.
@pytest.mark.requires_addin
@node
def test_the_real_harnesses_measure_but_stay_unpinned():
    """The add-in is present in this checkout and the registry reads cleanly,
    but nothing is pinned: that tree is unpublished (docs/CI.md), so a digest of
    it could not be reproduced by anyone else."""
    registry_path = ROOT.parent / "tantular_office_addin" / "src" / "promptRegistry.js"
    if not registry_path.is_file():
        pytest.skip("the Office add-in sibling is not checked out")
    digest, rows = vhi.prompt_registry_digest(registry_path)
    assert vhi.SHA256_RE.match(digest)
    assert len(rows) >= 5
    for name in ("tantular-office-current", "tantular-office-candidate"):
        system = (hd.load_harness(name).get("prompts") or {}).get("system") or {}
        assert system.get("source") == "prompt_registry", name
        assert system.get("verified") is not True, name
        assert not system.get("sha256"), name
