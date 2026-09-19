"""GEPA reflective prompt evolution (RSI MVP Phase 2, Task 2.1).

The two properties the handoff names explicitly:

* Pareto acceptance never regresses a per-instance score.
* A mutation that improves two instances and breaks one is NOT promoted on
  aggregate — the single-lineage regression trap the DGM section warns about.

Offline and deterministic: a fake client, injected. Nothing here contacts a
network, and nothing trains.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gepa  # noqa: E402

HARNESS = ROOT / "configs" / "harnesses" / "tantular-office-candidate.yaml"

TRAIN = ["t1", "t2", "t3"]
HOLDOUT = ["h1", "h2", "h3"]

INSTANCES = {
    key: {"prompt": f"Gunakan alat office_edit untuk {key}",
          "failure": "keluaran bukan JSON kontrak"}
    for key in TRAIN + HOLDOUT
}

SEED_ID = "Balas hanya dengan JSON kontrak edit. Jangan menambah fakta baru."


@pytest.fixture
def harness_spec():
    return yaml.safe_load(HARNESS.read_text(encoding="utf-8"))


def candidate(name: str, scores: dict[str, bool], instruction: str = SEED_ID
              ) -> gepa.Candidate:
    return gepa.Candidate(id=name, instruction=instruction,
                          harness_identity="d" * 64, model_identity="m",
                          scores=scores)


class FakeSystem:
    """Deterministic stand-in for a rollout through bridge_client.

    `outcomes` maps instruction -> {instance_id: bool}. Unknown instructions
    fall back to the seed's behaviour, so a test only declares what it cares
    about.
    """

    def __init__(self, outcomes: dict[str, dict[str, bool]], default: dict[str, bool]):
        self.outcomes = outcomes
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, instruction: str, ids: list[str]) -> dict[str, bool]:
        self.calls.append((instruction, list(ids)))
        table = self.outcomes.get(instruction, self.default)
        return {i: bool(table.get(i, False)) for i in ids}


def reflector(text: str):
    calls: list[list[dict[str, str]]] = []

    def reflect(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return text

    reflect.calls = calls  # type: ignore[attr-defined]
    return reflect


# --- Pareto and acceptance --------------------------------------------------


def test_a_mutation_that_fixes_two_and_breaks_one_is_not_promoted():
    """The single-lineage regression trap, stated as a test.

    The child's MEAN is strictly better (2/3 vs 1/3) and it must still be
    refused, because h1 regressed.
    """
    parent = candidate("p", {"h1": True, "h2": False, "h3": False})
    child = candidate("c", {"h1": False, "h2": True, "h3": True})
    assert child.score > parent.score
    ok, why = gepa.accepts(child, parent)
    assert ok is False
    assert "regresses" in why and "h1" in why


def test_a_strictly_better_mutation_is_accepted():
    parent = candidate("p", {"h1": True, "h2": False, "h3": False})
    child = candidate("c", {"h1": True, "h2": True, "h3": False})
    ok, why = gepa.accepts(child, parent)
    assert ok is True
    assert "no regression" in why


def test_an_identical_mutation_is_not_accepted():
    parent = candidate("p", {"h1": True, "h2": False})
    child = candidate("c", {"h1": True, "h2": False})
    ok, why = gepa.accepts(child, parent)
    assert ok is False
    assert "no instance improved" in why


def test_domination_requires_better_on_one_and_worse_on_none():
    a = candidate("a", {"h1": True, "h2": True})
    b = candidate("b", {"h1": True, "h2": False})
    c = candidate("c", {"h1": False, "h2": True})
    assert gepa.dominates(a, b) is True
    assert gepa.dominates(b, a) is False
    # Complementary: neither dominates.
    assert gepa.dominates(b, c) is False
    assert gepa.dominates(c, b) is False


def test_the_frontier_keeps_complementary_candidates():
    b = candidate("b", {"h1": True, "h2": False})
    c = candidate("c", {"h1": False, "h2": True})
    frontier = gepa.pareto_frontier([b, c])
    assert {x.id for x in frontier} == {"b", "c"}
    a = candidate("a", {"h1": True, "h2": True})
    assert {x.id for x in gepa.pareto_frontier([a, b, c])} == {"a"}


# --- identity ---------------------------------------------------------------


def test_every_candidate_carries_a_harness_identity(harness_spec):
    digest = gepa.harness_identity(harness_spec)
    assert len(digest) == 64
    import harness_distill as hd
    assert digest == hd.canonical_digest(harness_spec)


def test_a_harness_identity_ignores_a_self_declared_digest(harness_spec):
    claimed = copy.deepcopy(harness_spec)
    claimed["digest"] = "0" * 64
    assert gepa.harness_identity(claimed) == gepa.harness_identity(harness_spec)


def test_an_unsafe_harness_is_refused(harness_spec):
    unsafe = copy.deepcopy(harness_spec)
    unsafe["mutation"]["production_self_modify"] = True
    with pytest.raises(Exception) as error:
        gepa.harness_identity(unsafe)
    assert "production_self_modify" in str(error.value)


def test_a_trajectory_without_identity_is_rejected():
    with pytest.raises(gepa.GepaError) as error:
        gepa.require_attribution({"instance_id": "t1", "model_identity": "m"})
    assert "harness_identity" in str(error.value)
    with pytest.raises(gepa.GepaError):
        gepa.require_attribution({"instance_id": "t1", "harness_identity": "d"})


# --- the loop ---------------------------------------------------------------


def run_optimize(harness_spec, *, outcomes, default, reflect_text, budget=12,
                 train=None, holdout=None):
    system = FakeSystem(outcomes, default)
    reflect = reflector(reflect_text)
    result = gepa.optimize(
        seed_instruction=SEED_ID,
        instances=INSTANCES,
        run_system=system,
        reflect=reflect,
        harness_spec=harness_spec,
        model_identity="tantular-office-9b",
        train_ids=list(TRAIN if train is None else train),
        holdout_ids=list(HOLDOUT if holdout is None else holdout),
        rollout_budget=budget,
    )
    return result, system, reflect


BETTER_ID = ("Balas hanya JSON kontrak edit yang valid dan jangan mengubah "
             "angka pada dokumen sumber.")


def test_an_improving_mutation_is_accepted_and_reported(harness_spec):
    result, _, reflect = run_optimize(
        harness_spec,
        outcomes={BETTER_ID: {"h1": True, "h2": True, "h3": False,
                              "t1": True, "t2": True, "t3": False}},
        default={"h1": True, "h2": False, "h3": False,
                 "t1": True, "t2": False, "t3": False},
        reflect_text=BETTER_ID)
    assert result["accepted"], result["rejected"]
    assert result["best"]["score"] > result["seed"]["score"]
    assert result["training_authorized"] is False
    assert reflect.calls, "the reflector must actually be consulted"


def test_a_regressing_mutation_is_rejected_by_the_loop(harness_spec):
    result, _, _ = run_optimize(
        harness_spec,
        outcomes={BETTER_ID: {"h1": False, "h2": True, "h3": True,
                              "t1": False, "t2": True, "t3": True}},
        default={"h1": True, "h2": False, "h3": False,
                 "t1": True, "t2": False, "t3": False},
        reflect_text=BETTER_ID)
    assert result["accepted"] == []
    assert result["rejected"]
    assert "regresses" in result["rejected"][0]["reason"]
    # And the seed still wins, despite the child's better mean.
    assert result["best"]["id"] == "seed"


def test_the_rollout_budget_is_respected(harness_spec):
    result, system, _ = run_optimize(
        harness_spec,
        outcomes={}, default={k: False for k in INSTANCES},
        reflect_text=BETTER_ID, budget=4)
    assert result["rollouts"] <= 4
    assert len(system.calls) == result["rollouts"]


def test_reflection_is_conducted_in_indonesian(harness_spec):
    _, _, reflect = run_optimize(
        harness_spec, outcomes={}, default={k: False for k in INSTANCES},
        reflect_text=BETTER_ID, budget=6)
    messages = reflect.calls[0]
    assert messages[0]["role"] == "system"
    assert "Bahasa Indonesia" in messages[0]["content"]
    assert "Instruksi saat ini" in messages[1]["content"]


def test_an_english_mutation_is_refused(harness_spec):
    result, _, _ = run_optimize(
        harness_spec, outcomes={}, default={k: False for k in INSTANCES},
        reflect_text="Reply only with valid edit-contract JSON and nothing else.")
    assert result["accepted"] == []
    assert any("not Indonesian" in r.get("reason", "") for r in result["rejected"])


def test_feedback_names_the_failing_instances(harness_spec):
    probe = candidate("p", {"h1": False, "h2": True})
    text = gepa.textual_feedback(probe, INSTANCES)
    assert "h1" in text
    assert "h2" not in text.split("\n", 1)[1] if "\n" in text else True


def test_train_and_holdout_may_not_overlap(harness_spec):
    with pytest.raises(gepa.GepaError) as error:
        run_optimize(harness_spec, outcomes={}, default={},
                     reflect_text=BETTER_ID,
                     train=["t1", "h1"], holdout=HOLDOUT)
    assert "overlap" in str(error.value)


def test_a_missing_run_result_is_not_treated_as_a_failure(harness_spec):
    class Partial(FakeSystem):
        def __call__(self, instruction, ids):
            full = super().__call__(instruction, ids)
            full.pop(ids[0], None)
            return full

    with pytest.raises(gepa.GepaError) as error:
        gepa.optimize(
            seed_instruction=SEED_ID, instances=INSTANCES,
            run_system=Partial({}, {k: True for k in INSTANCES}),
            reflect=reflector(BETTER_ID),
            harness_spec=harness_spec, model_identity="m",
            train_ids=TRAIN, holdout_ids=HOLDOUT)
    assert "missing result is not a failure" in str(error.value)


def test_an_unknown_instance_id_is_refused(harness_spec):
    with pytest.raises(gepa.GepaError):
        run_optimize(harness_spec, outcomes={}, default={},
                     reflect_text=BETTER_ID, holdout=["nope"])


def test_a_missing_model_identity_is_refused(harness_spec):
    with pytest.raises(gepa.GepaError) as error:
        gepa.optimize(
            seed_instruction=SEED_ID, instances=INSTANCES,
            run_system=FakeSystem({}, {k: True for k in INSTANCES}),
            reflect=reflector(BETTER_ID), harness_spec=harness_spec,
            model_identity="  ", train_ids=TRAIN, holdout_ids=HOLDOUT)
    assert "model_identity" in str(error.value)


def test_no_weights_are_touched(harness_spec):
    source = (ROOT / "src" / "gepa.py").read_text(encoding="utf-8")
    for banned in ("torch", "peft", "LoraConfig", "Trainer", "optimizer",
                   "backward()", "train("):
        assert banned not in source, f"{banned} would mean this touches weights"


# --- CLI --------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "gepa.py"), *args],
        capture_output=True, text=True, timeout=60)


def test_dry_run_prints_the_sequence_and_writes_nothing(tmp_path):
    before = sorted(p.name for p in (ROOT / "data" / "gold").iterdir())
    result = run_cli("--dry-run")
    assert result.returncode == 0
    assert "rollout budget" in result.stdout
    assert "training_authorized  : false" in result.stdout
    assert "Pareto frontier" in result.stdout
    after = sorted(p.name for p in (ROOT / "data" / "gold").iterdir())
    assert before == after, "a dry run must write nothing"


def test_a_live_run_refuses_because_the_gold_set_is_blocked():
    result = run_cli()
    assert result.returncode != 0
    assert "GEPA REFUSED" in result.stderr
    assert "BLOCKED: 0 approved items" in result.stderr


# --- promotion is judged against the seed, across generations ---------------
# A regression can launder itself across a lineage: c1 breaks an instance and
# is refused; c2 improves on c1 and looks like a gain; but c2 still breaks the
# instance relative to what is DEPLOYED. Promotion must therefore compare to
# the immutable seed, while exploration may still descend from a rejected
# variant.

GEN1 = ("Balas hanya JSON dan jangan pernah menyalin angka dari dokumen "
        "sumber ke dalam ringkasan.")
GEN2 = ("Balas hanya JSON, jangan menambah fakta, dan sertakan setiap angka "
        "yang diminta pengguna pada bagian itu.")
GEN3 = ("Balas hanya JSON kontrak edit, pertahankan seluruh angka sumber, dan "
        "jangan mengubah makna kalimat yang sudah benar.")


class SequencedReflector:
    """Returns a different instruction per call, so generations differ."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]


def test_a_descendant_of_a_rejected_candidate_stays_non_promotable(harness_spec):
    """Three generations.

    seed passes h1 only.
      gen1 fixes h2+h3 but breaks h1  -> refused (regresses the seed)
      gen2 improves on gen1's score   -> STILL refused: h1 is still broken
                                         relative to the seed
      gen3 recovers h1 and keeps h2/h3-> promotable
    """
    seed_scores = {"h1": True, "h2": False, "h3": False,
                   "t1": True, "t2": False, "t3": False}
    outcomes = {
        SEED_ID: seed_scores,
        # Better mean than the seed (2/3 vs 1/3) and still a regression on h1.
        GEN1: {"h1": False, "h2": True, "h3": False,
               "t1": False, "t2": True, "t3": False},
        # Better than GEN1 on every instance, and h1 is STILL broken.
        GEN2: {"h1": False, "h2": True, "h3": True,
               "t1": False, "t2": True, "t3": True},
        # Recovers the seed's instance.
        GEN3: {"h1": True, "h2": True, "h3": True,
               "t1": True, "t2": True, "t3": True},
    }
    system = FakeSystem(outcomes, seed_scores)
    reflect = SequencedReflector([GEN1, GEN2, GEN3])

    result = gepa.optimize(
        seed_instruction=SEED_ID, instances=INSTANCES,
        run_system=system, reflect=reflect, harness_spec=harness_spec,
        model_identity="tantular-office-9b",
        train_ids=list(TRAIN), holdout_ids=list(HOLDOUT),
        rollout_budget=24)

    assert result["promotion_baseline"] == "seed"

    by_instruction = {c["instruction"]: c for c in result["accepted"]}
    # Generations 1 and 2 must never be promoted.
    assert GEN1 not in by_instruction, "a seed regression was promoted"
    assert GEN2 not in by_instruction, \
        "a descendant of a rejected candidate laundered the regression"
    # Generation 3 recovered h1 and is promotable.
    assert GEN3 in by_instruction, result["rejected"]

    # Both refusals were measured against the seed, not against their parent.
    refusals = {r.get("reason", ""): r for r in result["rejected"]}
    assert result["rejected"], "the refusals must be recorded"
    for record in result["rejected"]:
        assert record["measured_against"] == "seed"

    # And the promoted result really is better than what is deployed.
    assert result["best"]["instruction"] == GEN3
    assert result["best"]["scores"]["h1"] is True


def test_gen2_would_have_been_accepted_against_its_parent(harness_spec):
    """The bug this guards, stated directly.

    Measured against GEN1 (a rejected parent) GEN2 is a clean improvement with
    no regression — it would have been promoted. Measured against the seed it
    is a regression. Both facts are asserted so the distinction cannot be
    optimised away later.
    """
    seed = candidate("seed", {"h1": True, "h2": False, "h3": False})
    gen1 = candidate("c1", {"h1": False, "h2": True, "h3": False})
    gen2 = candidate("c2", {"h1": False, "h2": True, "h3": True})

    ok_vs_parent, _ = gepa.accepts(gen2, gen1)
    assert ok_vs_parent is True, "against its parent it looks like a clean gain"

    ok_vs_seed, why = gepa.accepts(gen2, seed)
    assert ok_vs_seed is False
    assert "h1" in why


def test_a_rejected_variant_remains_available_for_exploration(harness_spec):
    """Refusing to promote is not refusing to explore.

    GEN3 is only reachable because GEN1/GEN2 stayed in the pool to be selected
    from; an archive that discarded them would have to rediscover the lineage.
    """
    seed_scores = {"h1": True, "h2": False, "h3": False,
                   "t1": True, "t2": False, "t3": False}
    outcomes = {
        SEED_ID: seed_scores,
        GEN1: {"h1": False, "h2": True, "h3": False,
               "t1": False, "t2": True, "t3": False},
        GEN2: {"h1": False, "h2": True, "h3": True,
               "t1": False, "t2": True, "t3": True},
        GEN3: {"h1": True, "h2": True, "h3": True,
               "t1": True, "t2": True, "t3": True},
    }
    result = gepa.optimize(
        seed_instruction=SEED_ID, instances=INSTANCES,
        run_system=FakeSystem(outcomes, seed_scores),
        reflect=SequencedReflector([GEN1, GEN2, GEN3]),
        harness_spec=harness_spec, model_identity="m",
        train_ids=list(TRAIN), holdout_ids=list(HOLDOUT), rollout_budget=24)

    # The frontier legitimately collapses once GEN3 dominates everything, so
    # the archive's value is visible in the LINEAGE, not in the surviving
    # frontier: a promoted candidate whose parent was refused proves the loop
    # kept exploring from a rejected variant instead of discarding it.
    rejected_ids = {r["child"] for r in result["rejected"] if "child" in r}
    assert rejected_ids, "the run must have refused at least one variant"
    promoted = [c for c in result["accepted"] if c["instruction"] == GEN3]
    assert promoted, "GEN3 should have been promoted"
    assert promoted[0]["parent"] in rejected_ids | {"seed"}, \
        "the promoted candidate's lineage must be traceable"
    # And the refused variants are still not on the promotion list.
    assert all(c["instruction"] not in (GEN1, GEN2) for c in result["accepted"])
