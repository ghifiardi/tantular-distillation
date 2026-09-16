"""Corpus manifests, identities, leakage checking and registration.

No real corpus is read. Every fixture is synthesised in-test, the HMAC key is
an obviously non-production fixture key, and one test breaks `socket` across
the whole path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_corpus_manifest as bcm
import case_set_registry as registry
import check_case_leakage as ccl
import corpus_identity as ci
import harness_eval as he
import verify_case_set_approval as approval
import verify_case_set_leakage_evidence as evidence

# Obviously not production, and the generator refuses to treat it as such
# without --fixture.
FIXTURE_KEY = b"fixture-key-not-for-production-0123456789abcdef"
FIXTURE_KEY_ID = "corpus-hmac-fixture-v1"
OTHER_KEY = b"another-fixture-key-not-production-fedcba98765432"

SECRET = "Angka rahasia tujuh belas ribu"       # must never appear in output


def write_key(tmp_path, material=FIXTURE_KEY, mode=0o600, name="key"):
    path = tmp_path / name
    path.write_bytes(material)
    os.chmod(path, mode)
    return path


def corpus_rows(n=4):
    return [{"family": f"row::{i}", "system": "Anda editor.",
             "user": f"{SECRET} nomor {i}", "completion": f"Hasil {i}",
             "source_class": "synthetic", "checks": {},
             "provenance": {"teacher": "muse-glimmer"}} for i in range(n)]


def write_corpus(tmp_path, rows=None, name="corpus.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in (rows or corpus_rows())) + "\n",
                    encoding="utf-8")
    return path


# --- normalization ----------------------------------------------------------

def test_case_changes_identity():
    assert ci.normalize_text("Angka") != ci.normalize_text("angka")
    a = ci.text_identity(FIXTURE_KEY, "user_payload", "Angka")
    b = ci.text_identity(FIXTURE_KEY, "user_payload", "angka")
    assert a != b, "case is content in an Office edit"


def test_paragraph_structure_changes_identity():
    one = ci.text_identity(FIXTURE_KEY, "user_payload", "satu\ndua")
    two = ci.text_identity(FIXTURE_KEY, "user_payload", "satu\n\ndua")
    assert one != two, "a paragraph break is structure, not formatting noise"


def test_crlf_normalization_does_not_change_identity():
    for variant in ("a\r\nb", "a\rb", "a\nb"):
        assert ci.normalize_text(variant) == "a\nb"
    assert ci.text_identity(FIXTURE_KEY, "user_payload", "a\r\nb") == \
        ci.text_identity(FIXTURE_KEY, "user_payload", "a\nb")


def test_whitespace_runs_collapse_but_lines_survive():
    assert ci.normalize_text("a  \t b") == "a b"
    assert ci.normalize_text("a\n\n\n\n\nb") == "a\n\nb"
    assert ci.normalize_text("  a  ") == "a"


def test_framing_prevents_concatenation_collisions():
    """("ab","c") and ("a","bc") must not produce one identity."""
    first = ci.framed("d/v1", b"ab") + ci.framed("d/v1", b"c")
    second = ci.framed("d/v1", b"a") + ci.framed("d/v1", b"bc")
    assert first != second
    # And a domain change alone separates the spaces.
    assert ci.identity(FIXTURE_KEY, ci.DOMAIN_USER, b"x") != \
        ci.identity(FIXTURE_KEY, ci.DOMAIN_SYSTEM, b"x")


def test_a_short_key_is_refused():
    with pytest.raises(ci.IdentityError, match="at least"):
        ci.identity(b"too-short", ci.DOMAIN_USER, b"x")


def test_a_normalization_version_mismatch_refuses_rather_than_rehashing():
    with pytest.raises(ci.IdentityError, match="cannot be compared"):
        ci.require_same_normalization(1, 2, what="test")


def test_a_key_id_mismatch_refuses():
    with pytest.raises(ci.IdentityError, match="cannot be compared"):
        ci.require_same_key_id("a", "b", what="test")


def test_merkle_root_is_order_independent_and_content_sensitive():
    a = ci.merkle_root(["x", "y", "z"])
    assert a == ci.merkle_root(["z", "y", "x"])
    assert a != ci.merkle_root(["x", "y"])
    assert a != ci.merkle_root(["x", "y", "w"])


# --- generator --------------------------------------------------------------

def build(tmp_path, rows=None, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = write_corpus(tmp_path, rows)
    return bcm.build(source, key=key, key_id=key_id, role="training",
                     now="2026-09-16T00:00:00Z"), source


def test_output_is_deterministic_for_one_input_and_key(tmp_path):
    (pub1, priv1), _ = build(tmp_path / "a")
    (pub2, priv2), _ = build(tmp_path / "b")
    assert priv1 == priv2
    assert pub1["corpus"]["private_manifest_sha256"] == \
        pub2["corpus"]["private_manifest_sha256"]
    assert pub1["corpus"]["private_merkle_root"] == \
        pub2["corpus"]["private_merkle_root"]


def test_the_source_is_never_modified(tmp_path):
    source = write_corpus(tmp_path)
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    bcm.build(source, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID, role="training")
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_malformed_json_fails_closed(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"family":"a","system":"s","user":"u","completion":"c"}\n'
                      "{not json\n", encoding="utf-8")
    with pytest.raises(bcm.ManifestError, match="invalid JSON"):
        bcm.build(source, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID, role="training")


def test_a_non_object_row_fails_closed(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text("[1,2,3]\n", encoding="utf-8")
    with pytest.raises(bcm.ManifestError, match="must be a JSON object"):
        bcm.build(source, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID, role="training")


def test_a_missing_required_field_fails_closed(tmp_path):
    rows = corpus_rows(2)
    del rows[1]["user"]
    source = write_corpus(tmp_path, rows)
    with pytest.raises(bcm.ManifestError, match="missing required field"):
        bcm.build(source, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID, role="training")


def test_a_duplicate_stable_id_fails_closed(tmp_path):
    rows = corpus_rows(2)
    rows[1]["family"] = rows[0]["family"]
    source = write_corpus(tmp_path, rows)
    with pytest.raises(bcm.ManifestError, match="duplicate stable id"):
        bcm.build(source, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID, role="training")


def test_an_empty_corpus_fails_closed(tmp_path):
    source = tmp_path / "empty.jsonl"
    source.write_text("\n\n", encoding="utf-8")
    with pytest.raises(bcm.ManifestError, match="no rows"):
        bcm.build(source, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID, role="training")


def test_no_corpus_text_appears_in_either_artifact(tmp_path):
    (public, private), _ = build(tmp_path)
    blob = yaml.safe_dump(public) + yaml.safe_dump(private)
    assert SECRET not in blob
    for fragment in ("Anda editor", "Hasil 0", "nomor 1"):
        assert fragment not in blob, f"corpus text leaked: {fragment!r}"


def test_no_key_material_appears_in_either_artifact(tmp_path):
    (public, private), _ = build(tmp_path)
    blob = yaml.safe_dump(public) + yaml.safe_dump(private)
    assert FIXTURE_KEY.decode() not in blob
    assert public["hmac_key_id"] == FIXTURE_KEY_ID      # the id is not the key


def test_different_keys_produce_incompatible_identities(tmp_path):
    (_, a), _ = build(tmp_path / "a")
    (_, b), _ = build(tmp_path / "b", key=OTHER_KEY, key_id="corpus-hmac-fixture-v2")
    assert a["merkle_root"] != b["merkle_root"]
    a_ids = {r["components"]["user_payload"] for r in a["rows"]}
    b_ids = {r["components"]["user_payload"] for r in b["rows"]}
    assert not (a_ids & b_ids)


def test_the_public_manifest_says_the_checkout_is_not_a_binding(tmp_path):
    (public, _), _ = build(tmp_path)
    assert public["source"]["checkout_head_is_not_a_binding"] is True
    assert public["source"]["kind"] == "machine_local_gitignored"
    assert public["source"]["raw_sha256"]


def test_components_without_an_extractor_are_unverifiable(tmp_path):
    (public, _), _ = build(tmp_path)
    for name, reason in ci.UNVERIFIABLE_COMPONENTS.items():
        block = public["components"][name]
        assert block["available"] is False
        assert block["status"] == "unverifiable"
        assert block["reason"] == reason
    for name in ("user_payload", "system_payload", "completion_payload",
                 "canonical_row"):
        assert public["components"][name]["status"] == "available"


def test_unattributed_harness_is_recorded_as_such(tmp_path):
    (public, _), _ = build(tmp_path)
    assert public["corpus"]["harness_attribution_status"] == "unattributed"
    assert public["corpus"]["harness_identities"] == []


# --- key file validation ----------------------------------------------------

@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666])
def test_unsafe_key_permissions_refuse(tmp_path, mode):
    path = write_key(tmp_path, mode=mode)
    with pytest.raises(bcm.ManifestError, match="group- or world-accessible"):
        bcm.read_key(path)


def test_a_short_key_file_refuses(tmp_path):
    path = write_key(tmp_path, material=b"short")
    with pytest.raises(bcm.ManifestError, match="real key material"):
        bcm.read_key(path)


def test_a_missing_key_file_refuses(tmp_path):
    with pytest.raises(bcm.ManifestError, match="no HMAC key file"):
        bcm.read_key(tmp_path / "absent")


def test_key_material_never_appears_in_a_refusal(tmp_path):
    path = write_key(tmp_path, mode=0o644)
    with pytest.raises(bcm.ManifestError) as caught:
        bcm.read_key(path)
    assert FIXTURE_KEY.decode() not in str(caught.value)


def test_an_atomic_write_failure_leaves_no_artifact(tmp_path):
    target = tmp_path / "out" / "public.yaml"

    class Boom(Exception):
        pass

    real = os.replace

    def exploding(src, dst):
        raise Boom("disk full")

    os.replace = exploding
    try:
        with pytest.raises(Boom):
            bcm.atomic_write(target, "content")
    finally:
        os.replace = real
    assert not target.exists(), "a half-written manifest must not survive"
    assert not list(target.parent.glob("*.tmp")), "no temp file left behind"


# --- CLI fixture/production distinction -------------------------------------

def run_cli(module, *args):
    return subprocess.run([sys.executable, str(ROOT / "src" / module), *args],
                          capture_output=True, text=True)


def test_the_cli_requires_fixture_to_be_declared(tmp_path):
    source = write_corpus(tmp_path)
    key = write_key(tmp_path)
    proc = run_cli("build_corpus_manifest.py", str(source),
                   "--hmac-key-file", str(key), "--hmac-key-id", FIXTURE_KEY_ID,
                   "--public-out", str(tmp_path / "p.yaml"),
                   "--private-out", str(tmp_path / "s.yaml"))
    assert proc.returncode == 2
    assert "looks like a fixture key" in proc.stderr
    assert FIXTURE_KEY.decode() not in proc.stderr + proc.stdout


def test_the_cli_refuses_fixture_flag_with_a_production_looking_id(tmp_path):
    source = write_corpus(tmp_path)
    key = write_key(tmp_path)
    proc = run_cli("build_corpus_manifest.py", str(source),
                   "--hmac-key-file", str(key), "--hmac-key-id", "corpus-hmac-v1",
                   "--public-out", str(tmp_path / "p.yaml"),
                   "--private-out", str(tmp_path / "s.yaml"), "--fixture")
    assert proc.returncode == 2
    assert "does not say so" in proc.stderr


def test_the_cli_refuses_to_overwrite_the_source(tmp_path):
    source = write_corpus(tmp_path)
    key = write_key(tmp_path)
    proc = run_cli("build_corpus_manifest.py", str(source),
                   "--hmac-key-file", str(key), "--hmac-key-id", FIXTURE_KEY_ID,
                   "--public-out", str(source),
                   "--private-out", str(tmp_path / "s.yaml"), "--fixture")
    assert proc.returncode == 2
    assert "over the source" in proc.stderr


def test_the_cli_writes_the_private_manifest_unreadable_to_others(tmp_path):
    source = write_corpus(tmp_path)
    key = write_key(tmp_path)
    private = tmp_path / "private.yaml"
    proc = run_cli("build_corpus_manifest.py", str(source),
                   "--hmac-key-file", str(key), "--hmac-key-id", FIXTURE_KEY_ID,
                   "--public-out", str(tmp_path / "public.yaml"),
                   "--private-out", str(private), "--fixture")
    assert proc.returncode == 0, proc.stderr
    assert not stat.S_IMODE(private.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
    assert SECRET not in proc.stdout


# --- leakage checker --------------------------------------------------------

def case_set(users, name="draft-set"):
    return {
        "schema_version": he.CASE_SET_SCHEMA, "name": name, "approved": False,
        "split": "fixture", "source_class": "synthetic",
        "provenance": {"origin": "test", "created_at": "2026-09-16", "note": "n"},
        "cases": [{"case_id": f"c{i}",
                   "request": {"system": "Anda editor.", "user": u},
                   "expected": {"scorers": ["capability_pass_rate"]},
                   "expects_state_change": True, "requires_approval": True}
                  for i, u in enumerate(users)],
    }


def manifests(tmp_path, rows=None, key=FIXTURE_KEY, key_id=FIXTURE_KEY_ID):
    source = write_corpus(tmp_path, rows)
    return bcm.build(source, key=key, key_id=key_id, role="training")


def test_exact_user_payload_overlap_is_detected(tmp_path):
    public, private = manifests(tmp_path)
    reused = corpus_rows(1)[0]["user"]
    report = ccl.check(case_set([reused, "benar-benar baru"]), public, private,
                       FIXTURE_KEY, FIXTURE_KEY_ID)
    assert report["components"]["user_payload"]["status"] == ccl.OVERLAP
    assert report["exit_code"] == ccl.EXIT_OVERLAP
    assert len(report["components"]["user_payload"]["matches"]) == 1
    # The identity, never the text.
    blob = json.dumps(report)
    assert SECRET not in blob
    assert reused not in blob


def test_a_clean_comparison_returns_zero(tmp_path):
    public, private = manifests(tmp_path)
    report = ccl.check(case_set(["sesuatu yang lain", "dan lainnya"]),
                       public, private, FIXTURE_KEY, FIXTURE_KEY_ID)
    assert report["components"]["user_payload"]["status"] == ccl.CLEAN
    assert report["exit_code"] == ccl.EXIT_CLEAN


def test_a_shared_system_prompt_is_expected_and_does_not_block(tmp_path):
    """Every production case reuses one of the product's own system prompts.
    Counting that as leakage would fail every legitimate set."""
    public, private = manifests(tmp_path)
    report = ccl.check(case_set(["sesuatu yang lain"]), public, private,
                       FIXTURE_KEY, FIXTURE_KEY_ID)
    assert report["components"]["system_payload"]["status"] == ccl.OVERLAP
    assert report["expected_overlap_components"] == ["system_payload"]
    assert report["overlap_components"] == []
    assert report["exit_code"] == ccl.EXIT_CLEAN
    assert any("system_payload overlap is expected" in l
               for l in report["limitations"])


def test_normalization_only_differences_still_match(tmp_path):
    public, private = manifests(tmp_path)
    reused = corpus_rows(1)[0]["user"].replace(" ", "  ") + "  \r\n"
    report = ccl.check(case_set([reused]), public, private, FIXTURE_KEY,
                       FIXTURE_KEY_ID)
    assert report["components"]["user_payload"]["status"] == ccl.OVERLAP


def test_a_key_mismatch_refuses(tmp_path):
    public, private = manifests(tmp_path)
    with pytest.raises(ci.IdentityError, match="cannot be compared"):
        ccl.check(case_set(["x"]), public, private, OTHER_KEY, "different-id")


def test_a_changed_corpus_digest_voids_prior_conclusions(tmp_path):
    public, private = manifests(tmp_path)
    with pytest.raises(ccl.LeakageError, match="now void"):
        ccl.check(case_set(["x"]), public, private, FIXTURE_KEY, FIXTURE_KEY_ID,
                  expected_source_sha256="f" * 64)


def test_a_tampered_private_manifest_refuses(tmp_path):
    public, private = manifests(tmp_path)
    pub_path, priv_path = tmp_path / "p.yaml", tmp_path / "s.yaml"
    pub_path.write_text(yaml.safe_dump(public))
    tampered = copy.deepcopy(private)
    tampered["rows"][0]["components"]["user_payload"] = "0" * 64
    priv_path.write_text(yaml.safe_dump(tampered))
    with pytest.raises(ccl.LeakageError, match="does not match the digest"):
        ccl.load_manifests(pub_path, priv_path)


def test_a_missing_private_manifest_refuses(tmp_path):
    public, _ = manifests(tmp_path)
    pub_path = tmp_path / "p.yaml"
    pub_path.write_text(yaml.safe_dump(public))
    with pytest.raises(ccl.LeakageError, match="no private manifest"):
        ccl.load_manifests(pub_path, tmp_path / "absent.yaml")


def test_unverifiable_components_are_never_clean(tmp_path):
    public, private = manifests(tmp_path)
    report = ccl.check(case_set(["x"]), public, private, FIXTURE_KEY,
                       FIXTURE_KEY_ID)
    for name in ci.UNVERIFIABLE_COMPONENTS:
        assert report["components"][name]["status"] == ccl.UNVERIFIABLE
        assert report["components"][name]["status"] != ccl.CLEAN


def test_no_request_or_document_claim_is_inferred_from_user_payload(tmp_path):
    public, private = manifests(tmp_path)
    report = ccl.check(case_set(["benar-benar baru"]), public, private,
                       FIXTURE_KEY, FIXTURE_KEY_ID)
    assert report["components"]["user_payload"]["status"] == ccl.CLEAN
    assert report["components"]["request"]["status"] == ccl.UNVERIFIABLE
    assert report["components"]["document"]["status"] == ccl.UNVERIFIABLE
    assert any("rephrased prompt" in lim for lim in report["limitations"])


def test_a_required_component_with_nothing_to_check_refuses(tmp_path):
    """Zero checked cases is not a clean result."""
    public, private = manifests(tmp_path)
    empty = case_set(["x"])
    empty["cases"][0]["request"] = {"system": "only system"}
    report = ccl.check(empty, public, private, FIXTURE_KEY, FIXTURE_KEY_ID)
    assert report["components"]["user_payload"]["status"] == ccl.UNVERIFIABLE
    assert report["exit_code"] == ccl.EXIT_REFUSED


# --- development registration ----------------------------------------------

def test_development_registration_is_one_way(tmp_path):
    path = tmp_path / "REGISTRY.jsonl"
    cs = case_set(["a", "b"], name="dev-set")
    registry.register(cs, role=registry.ROLE_DEVELOPMENT,
                      purpose="scorer validation", first_used_at="2026-09-16",
                      path=path)
    with pytest.raises(registry.RegistryError, match="already been used"):
        registry.register(cs, role=registry.ROLE_HELD_OUT, purpose="decide",
                          first_used_at="2026-09-20", reviewer="A Person",
                          path=path)


def test_renaming_cannot_promote_development_data(tmp_path):
    path = tmp_path / "REGISTRY.jsonl"
    cs = case_set(["a", "b"], name="dev-set")
    registry.register(cs, role=registry.ROLE_CALIBRATION, purpose="tuning",
                      first_used_at="2026-09-16", path=path)
    renamed = copy.deepcopy(cs)
    renamed["name"] = "totally-new-name"
    # The NAME changed but the cases did not... which changes the digest,
    # because the name is part of the set. So the guard that matters is the
    # one below: identical BYTES cannot be laundered.
    same_bytes = copy.deepcopy(cs)
    allowed, why = registry.may_be_held_out(he.case_set_digest(same_bytes),
                                            registry.load(path))
    assert allowed is False
    assert "digest is the key" in why


def test_modified_bytes_produce_a_new_digest_that_is_not_yet_registered(tmp_path):
    path = tmp_path / "REGISTRY.jsonl"
    cs = case_set(["a", "b"], name="dev-set")
    registry.register(cs, role=registry.ROLE_DEVELOPMENT, purpose="p",
                      first_used_at="2026-09-16", path=path)
    edited = copy.deepcopy(cs)
    edited["cases"][0]["request"]["user"] = "a "     # one character
    allowed, _ = registry.may_be_held_out(he.case_set_digest(edited),
                                          registry.load(path))
    assert allowed is True, (
        "a new digest is genuinely unregistered -- which is exactly why a "
        "component-overlap check against registered material is still required")


def test_a_held_out_registration_names_a_reviewer(tmp_path):
    path = tmp_path / "REGISTRY.jsonl"
    cs = case_set(["a"], name="held-set")
    with pytest.raises(registry.RegistryError, match="names its reviewer"):
        registry.register(cs, role=registry.ROLE_HELD_OUT, purpose="decide",
                          first_used_at="2026-09-16", path=path)


def test_the_registry_is_append_only(tmp_path):
    path = tmp_path / "REGISTRY.jsonl"
    cs = case_set(["a"], name="s1")
    registry.register(cs, role=registry.ROLE_DEVELOPMENT, purpose="p",
                      first_used_at="2026-09-16", path=path)
    first = path.read_text()
    registry.register(case_set(["b"], name="s2"), role=registry.ROLE_DEVELOPMENT,
                      purpose="p", first_used_at="2026-09-17", path=path)
    assert path.read_text().startswith(first), "earlier entries are never rewritten"


# --- evidence records and approval integration ------------------------------

def test_the_shipped_leakage_template_binds_nothing():
    record = evidence.parse_record(
        ROOT / "docs" / "case_sets" / "LEAKAGE_EVIDENCE_TEMPLATE.md")
    with pytest.raises(evidence.LeakageEvidenceError):
        evidence.validate_record(record, source="TEMPLATE")


def complete_evidence(cs, **overrides):
    record = {
        "case_set": cs["name"], "case_set_sha256": he.case_set_digest(cs),
        "public_manifest_sha256": "a" * 64, "private_manifest_sha256": "b" * 64,
        "source_raw_sha256": "c" * 64,
        "checker_implementation_digest": "d" * 64,
        "normalization_version": 1, "hmac_key_id": FIXTURE_KEY_ID,
        "checked_at": "2026-09-16T00:00:00Z",
        "components": {"user_payload": {"status": "clean", "checked_cases": 2,
                                        "matches": 0}},
        "reviewed_by": "Raditio Ghifiardi", "reviewer_role": "Product owner",
        "reviewed_at": "2026-09-16",
        "limitations_understood": True,
        "unverifiable_components_accepted": True,
    }
    record.update(overrides)
    return record


def test_evidence_binds_to_the_exact_case_set_bytes():
    cs = case_set(["a", "b"], name="s")
    bound = evidence.bind_to_case_set(complete_evidence(cs), cs)
    assert bound["status"] == "MATCHED"
    modified = copy.deepcopy(cs)
    modified["cases"][0]["request"]["user"] = "edited"
    with pytest.raises(evidence.LeakageEvidenceError, match="different case set"):
        evidence.bind_to_case_set(complete_evidence(cs), modified)


def test_an_agent_cannot_sign_leakage_evidence():
    cs = case_set(["a"], name="s")
    with pytest.raises(evidence.LeakageEvidenceError, match="automated actor"):
        evidence.bind_to_case_set(
            complete_evidence(cs, reviewed_by="Claude Code"), cs)


def test_unverifiable_cannot_be_recorded_as_clean():
    cs = case_set(["a"], name="s")
    record = complete_evidence(cs, components={
        "user_payload": {"status": "clean", "checked_cases": 0, "matches": 0}})
    with pytest.raises(evidence.LeakageEvidenceError, match="zero checked cases"):
        evidence.bind_to_case_set(record, cs)


def test_recorded_overlap_blocks_binding():
    cs = case_set(["a"], name="s")
    record = complete_evidence(cs, components={
        "user_payload": {"status": "overlap", "checked_cases": 2, "matches": 1}})
    with pytest.raises(evidence.LeakageEvidenceError, match="overlap recorded"):
        evidence.bind_to_case_set(record, cs)


def test_office_document_coverage_is_reported_as_blocked():
    cs = case_set(["a"], name="s")
    bound = evidence.bind_to_case_set(complete_evidence(cs), cs)
    assert set(bound["office_components_blocked"]) == {"document",
                                                       "full_office_case"}


def test_production_approval_requires_every_term(tmp_path):
    cs = case_set([f"u{i}" for i in range(5)], name="small-set")
    verdict = approval.production_approval(cs, registry_path=tmp_path / "r.jsonl")
    assert verdict["approved_for_production"] is False
    joined = " ".join(verdict["problems"])
    assert "no human approval record" in joined
    assert "no leakage-evidence record" in joined
    assert "below the 320" in joined


def test_no_shipped_case_set_is_production_approved(tmp_path):
    cs = he.load_case_set(ROOT / "tests" / "fixtures" / "harness_cases" /
                          "fixture-office-v1.yaml")
    verdict = approval.production_approval(cs, registry_path=tmp_path / "r.jsonl")
    assert verdict["approved_for_production"] is False
    assert verdict["problems"]


def test_only_the_two_templates_exist_in_docs_case_sets():
    names = sorted(p.name for p in (ROOT / "docs" / "case_sets").glob("*.md"))
    assert names == ["LEAKAGE_EVIDENCE_TEMPLATE.md", "TEMPLATE.md"], (
        "this milestone approves nothing and registers no real set")


def test_no_socket_is_opened_on_this_path(tmp_path, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the manifest layer must not use the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    public, private = manifests(tmp_path)
    ccl.check(case_set(["x"]), public, private, FIXTURE_KEY, FIXTURE_KEY_ID)
    registry.may_be_held_out("0" * 64, [])
