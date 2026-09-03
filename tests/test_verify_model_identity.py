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
