"""The verifier validates and hashes a human decision. It never makes one.

    ./.venv/bin/python -m pytest tests/test_verify_license_evidence.py -q

Every test builds its own registry and record under tmp_path. Nothing here
reads or writes configs/models/, except the two tests that deliberately assert
the shipped entries are still unresolved.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import verify_license_evidence as vle                          # noqa: E402

MODEL_ID = "example-org/Example-Teacher-7B"
REVISION = "0123456789abcdef0123456789abcdef01234567"
SRC_A = "a" * 64
SRC_B = "b1" * 32
REVIEWER = "A. Reviewer <reviewer@example.com>"
RATIONALE = ("The licence is silent on model outputs and the usage policy does "
             "not address training, so nothing reviewed grants this right.")

BODY = """
# Licence review — Example Teacher 7B

## What was examined
Both documents were read at the pinned revision.

## Reasoning
Recorded in full so a later reader can disagree with it.
"""


def front_matter(**over) -> dict:
    block = {
        "schema_version": 1,
        "registry_model": "example-teacher-7b",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "reviewed_at": "2026-01-15",
        "reviewed_by": REVIEWER,
        "sources": [{"path": "LICENSE", "sha256": SRC_A},
                    {"path": "USAGE_POLICY.md", "sha256": SRC_B}],
        "determination": {"output_training_permitted": False,
                          "rationale": RATIONALE},
    }
    block.update(over)
    return block


def write_record(tmp_path: Path, body: str = BODY, raw: str | None = None,
                 **over) -> Path:
    path = tmp_path / "docs" / "licences" / "example-teacher-7b.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        front = yaml.safe_dump(front_matter(**over), sort_keys=False,
                               default_flow_style=False, allow_unicode=True)
        raw = f"---\n{front}---\n{body}"
    path.write_text(raw, encoding="utf-8")
    return path


def write_registry(tmp_path: Path, **over) -> Path:
    """A registry entry with comments, so the writer can be shown preserving them."""
    spec = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "role": "teacher",
        "license": {
            "identifier": "apache-2.0",
            "output_training_permitted": False,
            "reviewed_at": "2026-01-15",
            "recheck_max_age_days": 180,
            "evidence_sha256": "LICENSE_EVIDENCE_DIGEST_EXAMPLE",
        },
    }
    for key, value in over.items():
        if key in ("output_training_permitted", "evidence_sha256"):
            spec["license"][key] = value
        else:
            spec[key] = value
    path = tmp_path / "configs" / "models" / "example-teacher-7b.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("# A comment that a YAML round-trip would destroy.\n"
            + yaml.safe_dump(spec, sort_keys=False, default_flow_style=False)
            + "# Trailing comment, also load-bearing for readers.\n")
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(vle, "ROOT", tmp_path)
    write_registry(tmp_path)
    return tmp_path


def run(*argv) -> int:
    return vle.main(list(argv))


def refuses(*argv) -> pytest.ExceptionInfo:
    with pytest.raises(SystemExit) as exc:
        run(*argv)
    assert exc.value.code == 2
    return exc


# --- the happy path ---------------------------------------------------------

def test_a_complete_record_verifies_and_reports_its_digest(repo, capsys):
    path = write_record(repo)
    assert run("example-teacher-7b") == 0
    out = capsys.readouterr().out
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected in out
    assert "output_training_permitted=false" in out


def test_the_digest_is_of_the_whole_file_not_the_front_matter(repo):
    """Hashing only the header would let the reasoning be rewritten under a
    digest that still matched. The prose is part of the decision."""
    path = write_record(repo)
    first = vle.parse_record(path)["record_sha256"]
    assert first == hashlib.sha256(path.read_bytes()).hexdigest()

    write_record(repo, body=BODY + "\nA sentence added to the reasoning.\n")
    assert vle.parse_record(path)["record_sha256"] != first


@pytest.mark.parametrize("field,value", [
    ("reviewed_by", "B. Different <b@example.com>"),
    ("reviewed_at", "2026-02-01"),
])
def test_changing_any_bound_field_moves_the_digest(repo, field, value):
    path = write_record(repo)
    before = vle.parse_record(path)["record_sha256"]
    write_record(repo, **{field: value})
    assert vle.parse_record(path)["record_sha256"] != before


def test_changing_a_source_hash_moves_the_digest(repo):
    path = write_record(repo)
    before = vle.parse_record(path)["record_sha256"]
    write_record(repo, sources=[{"path": "LICENSE", "sha256": "c" * 64},
                                {"path": "USAGE_POLICY.md", "sha256": SRC_B}])
    assert vle.parse_record(path)["record_sha256"] != before


def test_the_verifier_decides_nothing_a_permitted_record_verifies_identically(repo):
    """A determination of `true` gets no extra scrutiny and no less. The tool
    checks that a human recorded a decision, not which decision they reached."""
    write_record(repo, determination={"output_training_permitted": True,
                                      "rationale": RATIONALE})
    write_registry(repo, output_training_permitted=True)
    assert run("example-teacher-7b") == 0


# --- binding to the registry entry ------------------------------------------

@pytest.mark.parametrize("over,marker", [
    ({"model_id": "other-org/Other-7B"}, "model_id"),
    ({"revision": "f" * 40}, "revision"),
    ({"registry_model": "something-else"}, "registry_model"),
])
def test_a_record_about_another_checkpoint_does_not_bind(repo, over, marker, capsys):
    write_record(repo, **over)
    refuses("example-teacher-7b")
    assert marker in capsys.readouterr().err


def test_a_determination_contradicting_the_registry_refuses(repo, capsys):
    """Evidence filed against the opposite claim is worse than no evidence."""
    write_record(repo, determination={"output_training_permitted": True,
                                      "rationale": RATIONALE})
    refuses("example-teacher-7b")                # registry says false
    assert "output_training_permitted" in capsys.readouterr().err


def test_the_drafter_substitution_is_caught_by_binding_not_by_hashes(repo, capsys):
    """The Muse DFlash drafter publishes a LICENSE and USAGE_POLICY.md that are
    byte-identical to the parent's, so source hashes cannot tell them apart.
    Only model_id and revision can, which is why binding is fatal."""
    write_record(repo, model_id="example-org/Example-Teacher-7B-assistant")
    refuses("example-teacher-7b")
    err = capsys.readouterr().err
    assert "model_id" in err and "assistant" in err


# --- shape: every field, every way it can be wrong --------------------------

@pytest.mark.parametrize("missing", list(vle.RECORD_KEYS))
def test_a_partial_record_refuses(repo, missing, capsys):
    block = front_matter()
    block.pop(missing)
    write_record(repo, raw=f"---\n{yaml.safe_dump(block, sort_keys=False)}---\n{BODY}")
    refuses("example-teacher-7b")
    assert missing in capsys.readouterr().err


def test_an_unknown_field_refuses_rather_than_being_ignored(repo, capsys):
    """A typo silently dropped is a field nobody reviewed."""
    write_record(repo, reviewd_by="A. Typo")
    refuses("example-teacher-7b")
    assert "reviewd_by" in capsys.readouterr().err


def test_a_duplicate_key_refuses(repo, capsys):
    """PyYAML keeps the last of a repeated key, so a record could verify
    against one determination and read as another."""
    front = yaml.safe_dump(front_matter(), sort_keys=False)
    front += "determination:\n  output_training_permitted: true\n  rationale: " \
             f"{RATIONALE}\n"
    write_record(repo, raw=f"---\n{front}---\n{BODY}")
    refuses("example-teacher-7b")
    assert "duplicate key" in capsys.readouterr().err


@pytest.mark.parametrize("revision", [
    "", "REPLACE_WITH_PINNED_HUB_COMMIT", "0123456789abcdef", "A" * 40,
    "0123456789abcdef0123456789abcdef0123456789", 123456,
])
def test_a_revision_that_is_not_a_commit_refuses(repo, revision, capsys):
    write_record(repo, revision=revision)
    refuses("example-teacher-7b")
    assert "revision" in capsys.readouterr().err


@pytest.mark.parametrize("digest", [
    "", "a" * 63, "a" * 65, ("ab" * 32).upper(), "g" * 64,
    "sha256:" + "a" * 64, 1234, None,
])
def test_a_source_hash_that_is_not_a_sha256_refuses(repo, digest, capsys):
    write_record(repo, sources=[{"path": "LICENSE", "sha256": digest}])
    refuses("example-teacher-7b")
    assert "sha256" in capsys.readouterr().err


def test_an_all_digit_digest_names_the_yaml_number_trap(repo, capsys):
    """64 digits is valid hex, but unquoted YAML parses it as a number. The
    message must name the cause, not report a correct value as malformed."""
    write_record(repo, raw="---\n" + yaml.safe_dump(front_matter(), sort_keys=False)
                 .replace(SRC_A, "1" * 64) + "---\n" + BODY)
    refuses("example-teacher-7b")
    assert "quote it" in capsys.readouterr().err


def test_an_empty_source_list_refuses(repo, capsys):
    """A determination reached from no document is not a review."""
    write_record(repo, sources=[])
    refuses("example-teacher-7b")
    assert "sources" in capsys.readouterr().err


def test_a_duplicated_source_path_refuses(repo, capsys):
    write_record(repo, sources=[{"path": "LICENSE", "sha256": SRC_A},
                                {"path": "LICENSE", "sha256": SRC_B}])
    refuses("example-teacher-7b")
    assert "twice" in capsys.readouterr().err


# --- the reviewer must be a person ------------------------------------------

@pytest.mark.parametrize("reviewer", [
    "Claude", "claude-opus-5", "GPT-5", "an AI assistant", "review-bot",
    "automated licence agent", "generated by a script", "Copilot",
])
def test_a_machine_byline_refuses(repo, reviewer, capsys):
    write_record(repo, reviewed_by=reviewer)
    refuses("example-teacher-7b")
    assert "human judgement" in capsys.readouterr().err


@pytest.mark.parametrize("reviewer", [
    "Elizabeth Abbott", "Naomi Aime", "R. Bottomley <r@example.com>",
])
def test_a_human_name_that_merely_contains_those_letters_is_accepted(repo, reviewer):
    """Word boundaries matter: Abbott is a surname, not a bot."""
    write_record(repo, reviewed_by=reviewer)
    assert run("example-teacher-7b") == 0


@pytest.mark.parametrize("reviewer", ["", "   ", None])
def test_an_unsigned_record_refuses(repo, reviewer, capsys):
    write_record(repo, reviewed_by=reviewer)
    refuses("example-teacher-7b")
    assert "reviewed_by" in capsys.readouterr().err


# --- the determination must be explicit -------------------------------------

@pytest.mark.parametrize("permitted", ["true", "yes", 1, 0, None, "unknown", ""])
def test_a_determination_that_is_not_a_boolean_refuses(repo, permitted, capsys):
    write_record(repo, determination={"output_training_permitted": permitted,
                                      "rationale": RATIONALE})
    refuses("example-teacher-7b")
    # The SHAPE check must be what refuses. Asserting only the field name let
    # the later binding check ("!= registry") stand in for it, and a mutation
    # that accepted "yes" as a decision still passed.
    assert "must be exactly true or false" in capsys.readouterr().err


@pytest.mark.parametrize("rationale,marker", [
    ("", "rationale"),
    ("TODO: ask legal", "placeholder"),
    ("<fill this in later>", "placeholder"),
    ("Seems fine.", "at least"),
])
def test_an_unreasoned_determination_refuses(repo, rationale, marker, capsys):
    write_record(repo, determination={"output_training_permitted": False,
                                      "rationale": rationale})
    refuses("example-teacher-7b")
    assert marker in capsys.readouterr().err


def test_a_record_with_no_body_refuses(repo, capsys):
    write_record(repo, body="\n   \n")
    refuses("example-teacher-7b")
    assert "no body" in capsys.readouterr().err


# --- malformed files --------------------------------------------------------

@pytest.mark.parametrize("raw,marker", [
    ("no front matter at all\n", "must begin with"),
    ("---\nschema_version: 1\nstill going\n", "never closed"),
    ("---\n---\n" + BODY, "must be a mapping"),      # empty front matter -> None
    ("---\n: :\n---\n" + BODY, "not valid YAML"),
    ("---\n- a\n- b\n---\n" + BODY, "must be a mapping"),
])
def test_a_malformed_record_refuses(repo, raw, marker, capsys):
    write_record(repo, raw=raw)
    refuses("example-teacher-7b")
    assert marker in capsys.readouterr().err


def test_a_non_utf8_record_refuses(repo, capsys):
    path = repo / "docs" / "licences" / "example-teacher-7b.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"---\n\xff\xfe not utf-8\n---\nbody\n")
    refuses("example-teacher-7b")
    assert "UTF-8" in capsys.readouterr().err


def test_a_missing_record_refuses(repo, capsys):
    refuses("example-teacher-7b")
    assert "no evidence record" in capsys.readouterr().err


def test_a_missing_registry_entry_refuses(repo, capsys):
    write_record(repo)
    refuses("no-such-model")
    assert "no registry entry" in capsys.readouterr().err


def test_a_wrong_schema_version_refuses(repo, capsys):
    write_record(repo, schema_version=2)
    refuses("example-teacher-7b")
    assert "schema_version" in capsys.readouterr().err


# --- --write ----------------------------------------------------------------

def test_write_fills_the_digest_and_preserves_the_comments(repo):
    path = write_record(repo)
    model = repo / "configs" / "models" / "example-teacher-7b.yaml"
    before = model.read_text()
    assert run("example-teacher-7b", "--write") == 0

    after = model.read_text()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert yaml.safe_load(after)["license"]["evidence_sha256"] == digest
    assert "# A comment that a YAML round-trip would destroy." in after
    assert "# Trailing comment, also load-bearing for readers." in after
    assert before.replace("LICENSE_EVIDENCE_DIGEST_EXAMPLE", digest) == after


def test_write_is_refused_when_the_record_does_not_bind(repo):
    write_record(repo, revision="f" * 40)
    model = repo / "configs" / "models" / "example-teacher-7b.yaml"
    before = model.read_text()
    refuses("example-teacher-7b", "--write")
    assert model.read_text() == before, "a refused run must write nothing"


def test_a_report_only_run_writes_nothing(repo):
    write_record(repo)
    model = repo / "configs" / "models" / "example-teacher-7b.yaml"
    before = model.read_text()
    assert run("example-teacher-7b") == 0
    assert model.read_text() == before


def test_write_refuses_a_registry_with_no_single_evidence_line(repo, capsys):
    write_record(repo)
    model = repo / "configs" / "models" / "example-teacher-7b.yaml"
    model.write_text("".join(line for line in model.read_text().splitlines(True)
                             if "evidence_sha256" not in line))
    refuses("example-teacher-7b", "--write")
    assert "exactly one evidence_sha256" in capsys.readouterr().err


def test_rewriting_after_an_edit_moves_the_registry_digest(repo):
    """The whole point: editing the record invalidates the recorded digest."""
    write_record(repo)
    run("example-teacher-7b", "--write")
    model = repo / "configs" / "models" / "example-teacher-7b.yaml"
    first = yaml.safe_load(model.read_text())["license"]["evidence_sha256"]

    write_record(repo, body=BODY + "\nReconsidered after a second reading.\n")
    run("example-teacher-7b", "--write")
    second = yaml.safe_load(model.read_text())["license"]["evidence_sha256"]
    assert first != second


# --- the shipped repository stays unresolved --------------------------------

def test_the_template_binds_to_no_shipped_registry_entry():
    """It is fixture data. It must never be mistaken for a determination."""
    template = ROOT / "docs" / "licences" / "TEMPLATE.md"
    record = vle.parse_record(template)
    assert record["registry_model"] == "example-teacher-7b"
    assert not (ROOT / "configs" / "models" / "example-teacher-7b.yaml").exists()
    for name in ("muse-glimmer-30b", "qwen35-9b-instruct", "qwen35-122b-a10b"):
        spec = yaml.safe_load(
            (ROOT / "configs" / "models" / f"{name}.yaml").read_text())
        with pytest.raises(SystemExit):
            vle.bind_to_spec(record, spec, name)


def test_no_shipped_registry_entry_has_licence_evidence_yet():
    """PR B adds the format and the verifier, and answers no licence question.
    This test is expected to change when a reviewed record lands."""
    for name in ("muse-glimmer-30b", "qwen35-9b-instruct", "qwen35-122b-a10b"):
        spec = yaml.safe_load(
            (ROOT / "configs" / "models" / f"{name}.yaml").read_text())
        digest = spec["license"]["evidence_sha256"]
        assert not vle._SHA256_RE.fullmatch(str(digest)), name
        assert not (ROOT / "docs" / "licences" / f"{name}.md").exists(), name
