"""The digest filler must measure, or refuse — it must never guess.

    ./.venv/bin/python -m pytest tests/test_verify_model_identity.py -q

The tokenizer sha256 in configs/models/*.yaml is the COMPATIBILITY KEY that
decides whether Mode C (token-level KL) is possible at all. A digest written
from anywhere but a real local snapshot would make that decision on fiction, so
every path that cannot measure has to exit non-zero and leave the file alone.

No network, no transformers, no credential: the whole Hugging Face cache is a
handful of files under tmp_path.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import verify_model_identity as vmi                          # noqa: E402

COMMIT = "3f2a1b" + "0" * 33 + "d"
OTHER_COMMIT = "9e7c4d" + "1" * 33 + "a"
NUMERIC_COMMIT = "0" * 39 + "1"
TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"

SPEC = """\
# A comment that carries the reasoning and MUST survive a --write.
schema_version: 1

model_id: fake-org/Fake-9B
revision: REPLACE_WITH_PINNED_HUB_COMMIT
role: student
family: fake

tokenizer:
  model_id: fake-org/Fake-9B
  revision: REPLACE_WITH_PINNED_HUB_COMMIT
  sha256: TOKENIZER_DIGEST_FAKE_9B    # compatibility key (placeholder)

chat_template:
  source: model
  path: null
  sha256: CHAT_TEMPLATE_DIGEST_FAKE_9B

digests_verified: false
"""


def make_snapshot(cache: Path, repo: str = "fake-org/Fake-9B", *,
                  commit: str = COMMIT, template: str | None = TEMPLATE,
                  vocab_extra: str = "") -> Path:
    snapshot = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}) + vocab_extra, encoding="utf-8")
    config = {"tokenizer_class": "FakeTokenizer", "model_max_length": 32768}
    if template is not None:
        config["chat_template"] = template
    (snapshot / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


@pytest.fixture
def registry(tmp_path, monkeypatch) -> Path:
    """A fake repo root with one registry model, and an empty HF cache."""
    models = tmp_path / "repo" / "configs" / "models"
    models.mkdir(parents=True)
    (models / "fake-9b.yaml").write_text(SPEC, encoding="utf-8")
    monkeypatch.setattr(vmi, "ROOT", tmp_path / "repo")
    cache = tmp_path / "hf"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return models / "fake-9b.yaml"


def run(*argv: str) -> int:
    """main() with argv; returns the exit code."""
    import sys as _sys
    saved = _sys.argv
    _sys.argv = ["verify_model_identity.py", *argv]
    try:
        vmi.main()
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        _sys.argv = saved


def cache_dir() -> Path:
    return vmi.hf_cache_root()


# --- refusals ---------------------------------------------------------------

def test_absent_snapshot_fails_closed(registry, capsys):
    assert run("fake-9b", "--write") == 2
    assert "no local snapshot" in capsys.readouterr().err
    assert yaml.safe_load(registry.read_text())["digests_verified"] is False


def test_unknown_registry_model_fails_closed(registry, capsys):
    assert run("no-such-model") == 2
    assert "no such registry model" in capsys.readouterr().err


def test_report_without_write_leaves_placeholders(registry, capsys):
    make_snapshot(cache_dir())
    assert run("fake-9b", "--offline") == 1
    out = capsys.readouterr()
    assert "still hold placeholders" in out.err
    assert registry.read_text() == SPEC          # byte-for-byte untouched


def test_snapshot_of_a_different_repo_is_refused(registry, capsys):
    """Qwen3.5-9B and Qwen3.5-9B-Base are one suffix apart and not the same
    product (train/BASE_VS_INSTRUCT.md)."""
    other = make_snapshot(cache_dir(), repo="fake-org/Fake-9B-Base")
    assert run("fake-9b", "--write", "--snapshot", str(other)) == 2
    assert "declares" in capsys.readouterr().err
    assert yaml.safe_load(registry.read_text())["digests_verified"] is False


@pytest.mark.parametrize("derivative", [
    "unsloth/Qwen3.5-122B-A10B",
    "Intel/Qwen3.5-122B-A10B-int4-AutoRound",
    "nvidia/Qwen3.5-122B-A10B-NVFP4",
])
def test_the_qwen122b_canonical_id_does_not_match_derivatives(derivative):
    canonical = "Qwen/Qwen3.5-122B-A10B"
    assert vmi.model_ids.matches(canonical, canonical)
    assert not vmi.model_ids.matches(canonical, derivative)


def test_the_verifier_refuses_a_qwen122b_derivative_snapshot(
        registry, capsys):
    canonical = "Qwen/Qwen3.5-122B-A10B"
    derivative = "unsloth/Qwen3.5-122B-A10B"
    registry.write_text(
        registry.read_text().replace("fake-org/Fake-9B", canonical),
        encoding="utf-8")
    snapshot = make_snapshot(cache_dir(), repo=derivative)

    assert run("fake-9b", "--write", "--snapshot", str(snapshot)) == 2
    err = capsys.readouterr().err
    assert derivative in err
    assert canonical in err
    assert yaml.safe_load(registry.read_text())["digests_verified"] is False


def test_loose_directory_cannot_mark_the_spec_verified(registry, tmp_path, capsys):
    """A directory of files records no commit, so its digest is not evidence."""
    loose = tmp_path / "loose"
    loose.mkdir()
    snapshot = make_snapshot(cache_dir())
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (loose / name).write_bytes((snapshot / name).read_bytes())
    assert run("fake-9b", "--write", "--snapshot", str(loose)) == 2
    assert "commit is known" in capsys.readouterr().err
    assert yaml.safe_load(registry.read_text())["digests_verified"] is False


def test_directory_without_a_tokenizer_is_refused(registry, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run("fake-9b", "--snapshot", str(empty)) == 2
    assert "not a tokenizer snapshot" in capsys.readouterr().err


def test_missing_chat_template_is_refused(registry, capsys):
    make_snapshot(cache_dir(), template=None)
    assert run("fake-9b", "--write") == 2
    assert "no chat template" in capsys.readouterr().err


def test_pinned_revision_absent_from_the_cache_is_refused(registry, capsys):
    make_snapshot(cache_dir(), commit=OTHER_COMMIT)
    spec = registry.read_text().replace("revision: REPLACE_WITH_PINNED_HUB_COMMIT",
                                        f"revision: {COMMIT}")
    registry.write_text(spec, encoding="utf-8")
    assert run("fake-9b", "--write") == 2
    assert "not in the local cache" in capsys.readouterr().err


def test_an_unquoted_numeric_revision_is_refused(registry, capsys):
    """YAML 1.1 reads 000...1 as the integer 1. Resolving "whatever else is
    cached" instead would then be recorded as a verified identity."""
    make_snapshot(cache_dir())
    registry.write_text(
        registry.read_text().replace("revision: REPLACE_WITH_PINNED_HUB_COMMIT",
                                     f"revision: {NUMERIC_COMMIT}"),
        encoding="utf-8")
    assert run("fake-9b", "--write") == 2
    assert "not a string" in capsys.readouterr().err
    assert yaml.safe_load(registry.read_text())["digests_verified"] is False


def test_a_changed_tokenizer_is_a_mismatch_not_a_rewrite(registry, capsys):
    """Once a digest is pinned, a different snapshot must fail loudly rather
    than quietly overwrite the compatibility key."""
    make_snapshot(cache_dir())
    assert run("fake-9b", "--write") == 0
    filled = registry.read_text()

    # Same commit id, different tokenizer bytes: the digest must no longer match.
    import shutil
    shutil.rmtree(cache_dir())
    cache_dir().mkdir()
    make_snapshot(cache_dir(), vocab_extra=" ")
    assert run("fake-9b", "--write") == 2
    assert "MISMATCH" in capsys.readouterr().err
    assert registry.read_text() == filled        # unchanged


# --- the one success path ---------------------------------------------------

def test_write_fills_the_digests_and_keeps_the_comments(registry):
    make_snapshot(cache_dir())
    assert run("fake-9b", "--write", "--offline") == 0

    spec = yaml.safe_load(registry.read_text())
    assert spec["digests_verified"] is True
    assert vmi.SHA256_RE.match(spec["tokenizer"]["sha256"])
    assert vmi.SHA256_RE.match(spec["chat_template"]["sha256"])
    # A digest is only evidence alongside the commit it came from.
    assert spec["revision"] == COMMIT
    assert spec["tokenizer"]["revision"] == COMMIT

    text = registry.read_text()
    assert "# A comment that carries the reasoning and MUST survive a --write." in text
    assert "# compatibility key (placeholder)" in text   # trailing comment kept

    # Re-running is a clean verification, not a second write.
    assert run("fake-9b", "--offline") == 0


def test_the_chat_template_digest_is_the_template_and_nothing_else(registry):
    snapshot = make_snapshot(cache_dir())
    digest, source = vmi.chat_template_digest(snapshot)
    assert digest == hashlib.sha256(TEMPLATE.encode()).hexdigest()
    assert source == "tokenizer_config.json:chat_template"


def test_a_chat_template_jinja_file_wins_over_tokenizer_config(registry):
    snapshot = make_snapshot(cache_dir())
    (snapshot / "chat_template.jinja").write_text("OTHER", encoding="utf-8")
    digest, source = vmi.chat_template_digest(snapshot)
    assert digest == hashlib.sha256(b"OTHER").hexdigest()
    assert source == "chat_template.jinja"


def test_editing_only_the_chat_template_does_not_move_the_compatibility_key(registry):
    """The two digests are deliberately independent: a prompt-template edit must
    not flip a Mode C decision that is about the vocabulary."""
    snapshot = make_snapshot(cache_dir())
    before_tok, _ = vmi.tokenizer_digest(snapshot)
    before_tpl, _ = vmi.chat_template_digest(snapshot)

    config = json.loads((snapshot / "tokenizer_config.json").read_text())
    config["chat_template"] = TEMPLATE + "{# changed #}"
    (snapshot / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")

    after_tok, _ = vmi.tokenizer_digest(snapshot)
    after_tpl, _ = vmi.chat_template_digest(snapshot)
    assert after_tok == before_tok
    assert after_tpl != before_tpl


def test_key_order_in_tokenizer_config_is_not_a_difference(registry):
    snapshot = make_snapshot(cache_dir())
    before, _ = vmi.tokenizer_digest(snapshot)
    config = json.loads((snapshot / "tokenizer_config.json").read_text())
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps(dict(reversed(list(config.items()))), indent=2), encoding="utf-8")
    assert vmi.tokenizer_digest(snapshot)[0] == before


# --- the shipped spec, pinned exactly --------------------------------------
#
# Everything above proves the MEASUREMENT is sound: the digest covers the right
# files, ignores key order, moves when a prompt changes, refuses a missing
# snapshot. None of it pins what is actually checked in. A revision or digest
# could be edited to another syntactically valid value and the whole suite
# would stay green, because no test reads configs/models/qwen35-9b-instruct.yaml
# and says what it must contain.
#
# The architecture profile already has that assertion. This is its counterpart
# for model identity. The expectations below are LITERALS on purpose: reading
# them from the qualification document would make the test agree with whatever
# the document said, which checks nothing.
#
# Measured from Qwen/Qwen3.5-9B at commit
# c202236235762e1c871ad0ccb60c8ee5ba337b9a, metadata-only, offline. See
# docs/QWEN35_9B_IDENTITY_QUALIFICATION.md for the acquisition and the proof no
# weights were downloaded.

def test_the_shipped_qwen35_instruct_spec_pins_the_qualified_snapshot():
    spec = yaml.safe_load(
        (ROOT / "configs" / "models" / "qwen35-9b-instruct.yaml").read_text())

    assert spec["model_id"] == "Qwen/Qwen3.5-9B"
    assert spec["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

    tokenizer = spec["tokenizer"]
    assert tokenizer["model_id"] == "Qwen/Qwen3.5-9B"
    assert tokenizer["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert tokenizer["sha256"] == (
        "6f3a76fa0ff84cba487813d4024623233c4664ecedfc3f3857536f95d25504af")

    template = spec["chat_template"]
    assert template["source"] == "model"
    assert template["sha256"] == (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715")

    assert spec["architecture_profile"] == "qwen35-hybrid-dense-9b"
    assert spec["digests_verified"] is True

    # The tokenizer sha256 is the compatibility key that selects the
    # distillation mode, so it must be a real digest, never a placeholder that
    # merely looks filled in.
    assert vmi.SHA256_RE.match(tokenizer["sha256"])
    assert vmi.SHA256_RE.match(template["sha256"])
    # A Git object id is a different type from a SHA-256 prompt digest.
    assert len(spec["revision"]) == 40


# Measured from Qwen/Qwen3.5-122B-A10B at commit
# dc4d348443bc740c68e2d77492492c11606384d5, metadata-only and offline. As
# above, these are literals so the test cannot agree with an accidental edit.
QWEN122_COMMIT = "dc4d348443bc740c68e2d77492492c11606384d5"
QWEN122_TOKENIZER_SHA = \
    "6f3a76fa0ff84cba487813d4024623233c4664ecedfc3f3857536f95d25504af"
QWEN122_TEMPLATE_SHA = \
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
QWEN122_CANONICAL = "Qwen/Qwen3.5-122B-A10B"


def test_the_shipped_qwen122b_spec_pins_the_qualified_canonical_snapshot():
    spec = yaml.safe_load(
        (ROOT / "configs" / "models" / "qwen35-122b-a10b.yaml").read_text())

    assert spec["model_id"] == QWEN122_CANONICAL
    assert spec["revision"] == QWEN122_COMMIT
    assert spec["tokenizer"]["model_id"] == QWEN122_CANONICAL
    assert spec["tokenizer"]["revision"] == QWEN122_COMMIT
    assert spec["tokenizer"]["sha256"] == QWEN122_TOKENIZER_SHA
    assert spec["chat_template"]["source"] == "model"
    assert spec["chat_template"]["sha256"] == QWEN122_TEMPLATE_SHA
    assert spec["digests_verified"] is True

    assert len(spec["revision"]) == 40
    assert vmi.SHA256_RE.match(spec["tokenizer"]["sha256"])
    assert vmi.SHA256_RE.match(spec["chat_template"]["sha256"])

    for derivative in (
        "unsloth/Qwen3.5-122B-A10B",
        "Intel/Qwen3.5-122B-A10B-int4-AutoRound",
        "nvidia/Qwen3.5-122B-A10B-NVFP4",
    ):
        assert spec["model_id"] != derivative
        assert not vmi.model_ids.matches(spec["model_id"], derivative)


# --- the teacher's SOURCE, pinned exactly (not its digests) -----------------
#
# This entry named meta-models/Muse-Glimmer-30B-assistant for months. That repo
# is the DFlash speculative-decoding drafter, not this teacher: five layers,
# MuseGlimmerAssistantModel, block_size 16, target_layer_ids into a deeper
# parent, and no tokenizer or chat template at all -- because a drafter shares
# its parent's. The teacher is the parent, meta-models/Muse-Glimmer-30B.
#
# The mistake was survivable precisely because nothing asserted the source, so
# assert it: the positive value AND the negative, so the drafter cannot return
# through an equality check written loosely against a similar-looking name.
#
# This says nothing about verification. The revisions are still placeholders and
# digests_verified is still false; measuring the parent is a later, separate
# change. Test 5 below is what keeps this file honest about that.

MUSE_GLIMMER_PARENT = "meta-models/Muse-Glimmer-30B"
MUSE_GLIMMER_DRAFTER = "meta-models/Muse-Glimmer-30B-assistant"


def test_the_muse_glimmer_registry_names_the_parent_not_the_drafter():
    spec = yaml.safe_load(
        (ROOT / "configs" / "models" / "muse-glimmer-30b.yaml").read_text())

    # 1. the registry names the parent, and explicitly not the drafter
    assert spec["model_id"] == MUSE_GLIMMER_PARENT
    assert spec["model_id"] != MUSE_GLIMMER_DRAFTER

    # 2. the tokenizer source is the parent
    assert spec["tokenizer"]["model_id"] == MUSE_GLIMMER_PARENT
    assert spec["tokenizer"]["model_id"] != MUSE_GLIMMER_DRAFTER

    # 5. the source correction has since been qualified: these were the
    #    placeholder assertions that kept PR A from drifting into a
    #    qualification, and they are replaced by the exact pin below.
    assert spec["digests_verified"] is True
    assert vmi.SHA256_RE.match(str(spec["tokenizer"]["sha256"]))
    assert vmi.SHA256_RE.match(str(spec["chat_template"]["sha256"]))


def test_the_muse_glimmer_serving_repos_agree_with_the_corrected_source():
    """repos.bf16 lives in the serving config, and distill_plan reconciles it
    against the registry model_id -- so a half-applied correction breaks the
    planner rather than silently disagreeing. The quantized entries are
    derivatives OF the parent and are unaffected."""
    serving = yaml.safe_load(
        (ROOT / "configs" / "teachers" / "muse-glimmer.yaml").read_text())
    repos = serving["repos"]

    # 3. the BF16 weights source is the parent
    assert repos["bf16"] == MUSE_GLIMMER_PARENT
    assert repos["bf16"] != MUSE_GLIMMER_DRAFTER

    # 4. the derivatives are unchanged, and are derivatives of the parent
    assert repos["fp8"] == "RedHatAI/Muse-Glimmer-30B-FP8-block"
    assert repos["int4_mlx"] == "mlx-community/Muse-Glimmer-30B-4bit"

    # Serving aliases may remain, but they are not identity evidence: an Ollama
    # tag is mutable, and the legacy corpus records only this tag.
    assert repos["int4_ollama"] == "muse-glimmer:30b"
    assert repos["remote"] == "ollama/muse-glimmer-30b"

    # the two files must keep pointing at each other
    assert serving["registry_model"] == "muse-glimmer-30b"
    spec = yaml.safe_load(
        (ROOT / "configs" / "models" / "muse-glimmer-30b.yaml").read_text())
    assert spec["serving_config"] == "muse-glimmer"
    assert repos["bf16"] == spec["model_id"]


# Measured from meta-models/Muse-Glimmer-30B at commit
# a4e59da52a7bc87ae7251dd5545c0dd437c44b68, metadata only, offline. The
# expectations are literals on purpose: reading them from the registry would
# make the test agree with whatever the registry said.
MUSE_COMMIT = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
MUSE_TOKENIZER_SHA = \
    "f945b361c8e542b5d2cb6737d03537d656aaeb94f5fe80b9ec7c00234260cf76"
MUSE_TEMPLATE_SHA = \
    "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678"


def test_the_shipped_muse_glimmer_spec_pins_the_qualified_parent():
    """The teacher's counterpart to the student's exact-pin test.

    Qualifying the teacher does NOT make the legacy corpus identity-ready:
    that corpus records only a mutable Ollama tag, and execution-artifact
    readiness is a separate claim. See docs/QWEN35_9B_IDENTITY_QUALIFICATION.md
    for the student and the audit's execution_artifact block for the corpus.
    """
    spec = yaml.safe_load(
        (ROOT / "configs" / "models" / "muse-glimmer-30b.yaml").read_text())

    assert spec["model_id"] == MUSE_GLIMMER_PARENT
    assert spec["model_id"] != MUSE_GLIMMER_DRAFTER
    assert spec["revision"] == MUSE_COMMIT
    assert spec["tokenizer"]["model_id"] == MUSE_GLIMMER_PARENT
    assert spec["tokenizer"]["revision"] == MUSE_COMMIT
    assert spec["tokenizer"]["sha256"] == MUSE_TOKENIZER_SHA
    assert spec["chat_template"]["source"] == "model"
    assert spec["chat_template"]["sha256"] == MUSE_TEMPLATE_SHA
    assert spec["digests_verified"] is True

    assert len(spec["revision"]) == 40           # a Git object id
    assert vmi.SHA256_RE.match(spec["tokenizer"]["sha256"])
    assert vmi.SHA256_RE.match(spec["chat_template"]["sha256"])
