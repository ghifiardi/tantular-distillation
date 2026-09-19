"""GEPA — reflective prompt evolution (RSI MVP Phase 2, Task 2.1).

From the Recursive Self-Improvement Report v2 §B.1: GEPA "treats your whole
pipeline — system prompts, tool descriptions, module instructions — as the
thing being optimised. It samples trajectories including reasoning and tool
calls, reflects on them in natural language to diagnose what went wrong,
proposes prompt updates, and keeps a Pareto frontier over individual data
instances rather than a single best prompt, so complementary lessons can be
recombined instead of averaged away."

Reported: beats GRPO by up to 20% with ~35x fewer rollouts on Qwen3 8B — very
nearly Tantular's base family. It is inference-only and touches no weights,
which is why it is the report's "do first" and why it can run here at all.

Three properties this implementation refuses to compromise:

1. **The Pareto frontier is not an average.** A mutation that improves two
   instances and breaks one has a better mean score and is a regression. The
   report's DGM section is explicit that "greedy hill-climbing on a single
   lineage stagnates in local optima", so acceptance requires no per-instance
   regression and the archive keeps complementary candidates.

2. **Indonesian, natively.** Reflection prompts and mutated instructions are
   written in Indonesian. The report: "a prompt optimised in English and
   translated loses exactly the register control you are trying to keep."

3. **Identity or nothing.** Every candidate carries a `harness_identity`
   digest from `harness_distill.canonical_digest`, and a trajectory without
   model AND harness identity is rejected — the architecture doc's attribution
   rule. A rollout you cannot attribute cannot support a claim.

Rollouts go through `src/bridge_client.py`'s URL abstraction. Tests inject a
deterministic fake client; nothing here contacts a network by itself.

NO TRAINING. NO WEIGHTS. `train/TRAINING_BLOCKED.md` is controlling; every
result carries `training_authorized: false`.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gold_set as gs  # noqa: E402
import harness_distill as hd  # noqa: E402

# Small by default and always explicit. "The rollout budget is an explicit
# argument; default small. No hidden loops."
DEFAULT_ROLLOUT_BUDGET = 12

# The slice the report says to optimise first: "run this on the agent/tool-use
# workload first — that is where prompt structure carries the most weight and
# where your eval is most automatable."
DEFAULT_SLICE = "tool_use"

REFLECTION_SYSTEM_ID = (
    "Anda adalah pengoptimal instruksi berbahasa Indonesia. "
    "Anda membaca jejak kegagalan sebuah sistem dan mengusulkan PERBAIKAN "
    "INSTRUKSI, bukan jawaban. Tulis instruksi baru dalam Bahasa Indonesia "
    "yang baku, ringkas, dan dapat diperiksa. Jangan menambah fakta. "
    "Balas HANYA instruksi barunya."
)


class GepaError(Exception):
    """GEPA could not run. Never the same thing as a candidate scoring badly."""


def die(msg: str, code: int = 2) -> None:
    print(f"\nGEPA REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


@dataclass(frozen=True)
class Candidate:
    """One prompt variant and its per-instance scores.

    `scores` maps instance id -> bool. Per-instance, never aggregated at rest:
    the aggregate is derived on demand so no code path can accidentally treat
    the mean as the thing being optimised.
    """

    id: str
    instruction: str
    harness_identity: str
    model_identity: str
    scores: dict[str, bool] = field(default_factory=dict)
    parent: str | None = None
    note: str = ""

    @property
    def score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for v in self.scores.values() if v) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "parent": self.parent, "note": self.note,
            "instruction": self.instruction,
            "harness_identity": self.harness_identity,
            "model_identity": self.model_identity,
            "scores": dict(self.scores), "score": self.score,
        }


# --- identity ---------------------------------------------------------------


def harness_identity(spec: dict[str, Any]) -> str:
    """Canonical digest of a harness spec, via the repo's own function."""
    if not isinstance(spec, dict) or not spec:
        raise GepaError("harness spec is required to compute an identity")
    hd.validate_harness(spec)          # raises on unsafe or incomplete policy
    return hd.canonical_digest(spec)


def require_attribution(trajectory: dict[str, Any]) -> None:
    """A trajectory without model AND harness identity is rejected."""
    if not isinstance(trajectory, dict):
        raise GepaError("trajectory must be an object")
    for field_name in ("model_identity", "harness_identity"):
        value = trajectory.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise GepaError(
                f"trajectory {trajectory.get('instance_id', '<unknown>')!r} "
                f"carries no {field_name}; an unattributable rollout cannot "
                "support a claim")


# --- Pareto -----------------------------------------------------------------


def dominates(a: Candidate, b: Candidate) -> bool:
    """Strictly better on at least one instance and worse on none."""
    ids = set(a.scores) | set(b.scores)
    better = False
    for key in ids:
        x, y = bool(a.scores.get(key)), bool(b.scores.get(key))
        if not x and y:
            return False
        if x and not y:
            better = True
    return better


def pareto_frontier(pool: Iterable[Candidate]) -> list[Candidate]:
    """Every candidate not dominated by another.

    Keeps complementary candidates rather than collapsing to the best mean —
    the mechanism the report credits for GEPA recombining "complementary
    lessons ... instead of averaged away".
    """
    items = list(pool)
    return [c for c in items if not any(o is not c and dominates(o, c) for o in items)]


def regressions(candidate: Candidate, baseline: Candidate) -> list[str]:
    """Instances the baseline passed and the candidate does not."""
    return sorted(k for k, v in baseline.scores.items()
                  if v and not candidate.scores.get(k))


def accepts(candidate: Candidate, baseline: Candidate) -> tuple[bool, str]:
    """Promotion check against the DEPLOYED baseline, on a held-out slice.

    `baseline` must be the immutable seed — what is actually deployed — never
    the archive parent the candidate happened to be explored from. Comparing a
    child to a REJECTED parent is how a regression launders itself across
    generations: c1 breaks h1 and is refused; c2 improves on c1 and looks like
    a gain; but c2 still breaks h1 relative to what is deployed, and promoting
    it ships the regression c1 was refused for.

    Explicitly NOT `candidate.score > baseline.score`. A mutation that fixes
    two instances and breaks one scores higher and is a regression; that is the
    single-lineage trap the DGM section warns about.
    """
    lost = regressions(candidate, baseline)
    if lost:
        return False, (f"regresses {len(lost)} instance(s): "
                       f"{', '.join(lost[:5])}")
    gained = sorted(k for k, v in candidate.scores.items()
                    if v and not baseline.scores.get(k))
    if not gained:
        return False, "no instance improved"
    return True, f"improves {len(gained)} instance(s) with no regression"


# --- feedback ---------------------------------------------------------------


def textual_feedback(candidate: Candidate, instances: dict[str, dict[str, Any]],
                     limit: int = 5) -> str:
    """The diagnostic, in words and in Indonesian.

    GEPA's signal is `metric + textual feedback`; a bare score tells the
    reflector nothing about WHAT to change.
    """
    failed = [k for k, ok in sorted(candidate.scores.items()) if not ok]
    if not failed:
        return "Semua instance lulus. Tidak ada kegagalan untuk dianalisis."
    lines = [f"Instruksi saat ini gagal pada {len(failed)} instance:"]
    for key in failed[:limit]:
        item = instances.get(key, {})
        lines.append(
            f"- {key}: diminta {item.get('prompt', '(tidak ada prompt)')!r}; "
            f"kegagalan: {item.get('failure', 'keluaran tidak memenuhi verifier')}"
        )
    if len(failed) > limit:
        lines.append(f"- ... dan {len(failed) - limit} lainnya")
    return "\n".join(lines)


def reflection_messages(candidate: Candidate, feedback: str) -> list[dict[str, str]]:
    """Chat messages for the reflector. Indonesian, per the report."""
    return [
        {"role": "system", "content": REFLECTION_SYSTEM_ID},
        {"role": "user", "content": (
            "Instruksi saat ini:\n"
            f"\"\"\"{candidate.instruction}\"\"\"\n\n"
            f"Umpan balik dari eksekusi:\n{feedback}\n\n"
            "Tulis satu instruksi pengganti dalam Bahasa Indonesia."
        )},
    ]


def is_indonesian_enough(text: str, minimum: int = 2) -> bool:
    """A cheap register guard on a mutated instruction.

    Not a language detector. It exists so an English instruction cannot enter
    the pool unnoticed, which would quietly undo the point of optimising in
    Indonesian at all.
    """
    markers = ("yang", "dan", "tidak", "harus", "jangan", "pada", "dengan",
               "untuk", "dari", "adalah", "setiap", "hanya", "boleh")
    lowered = f" {str(text).lower()} "
    return sum(1 for m in markers if f" {m} " in lowered) >= minimum


# --- the loop ---------------------------------------------------------------


def optimize(*, seed_instruction: str,
             instances: dict[str, dict[str, Any]],
             run_system: Callable[[str, list[str]], dict[str, bool]],
             reflect: Callable[[list[dict[str, str]]], str],
             harness_spec: dict[str, Any],
             model_identity: str,
             train_ids: list[str],
             holdout_ids: list[str],
             rollout_budget: int = DEFAULT_ROLLOUT_BUDGET,
             on_event: Callable[[dict[str, Any]], None] = lambda e: None,
             ) -> dict[str, Any]:
    """Run GEPA to exhaustion of the rollout budget.

    `run_system` executes the system under one instruction on a set of instance
    ids and returns per-instance booleans. In production it goes through
    `bridge_client`; in tests it is a deterministic fake. `reflect` proposes a
    mutated instruction from chat messages.
    """
    if not str(seed_instruction).strip():
        raise GepaError("seed_instruction is required")
    if not instances:
        raise GepaError("no instances supplied")
    if not train_ids or not holdout_ids:
        raise GepaError("both a train and a held-out slice are required")
    overlap = sorted(set(train_ids) & set(holdout_ids))
    if overlap:
        # Accepting on instances the mutation was proposed from is how a
        # prompt optimiser reports a gain it does not have.
        raise GepaError(
            f"train and held-out slices overlap on {len(overlap)} instance(s) "
            f"(first: {overlap[0]!r}); acceptance must be on unseen instances")
    unknown = sorted(set(train_ids + holdout_ids) - set(instances))
    if unknown:
        raise GepaError(f"unknown instance id(s): {unknown[:3]!r}")
    if rollout_budget < 2:
        raise GepaError("rollout_budget must allow at least a baseline and one "
                        "mutation")
    if not str(model_identity).strip():
        raise GepaError("model_identity is required for attribution")

    identity = harness_identity(harness_spec)

    rollouts = 0

    def evaluate(instruction: str, ids: list[str]) -> dict[str, bool]:
        nonlocal rollouts
        rollouts += 1
        scores = run_system(instruction, list(ids))
        if not isinstance(scores, dict):
            raise GepaError("run_system must return {instance_id: bool}")
        missing = sorted(set(ids) - set(scores))
        if missing:
            raise GepaError(
                f"run_system returned no result for {len(missing)} instance(s) "
                f"(first: {missing[0]!r}); a missing result is not a failure")
        return {k: bool(v) for k, v in scores.items()}

    baseline = Candidate(
        id="seed", instruction=seed_instruction, harness_identity=identity,
        model_identity=model_identity,
        scores=evaluate(seed_instruction, holdout_ids), note="seed")
    on_event({"event": "seed", "score": baseline.score, "rollouts": rollouts})

    pool: list[Candidate] = [baseline]
    accepted: list[Candidate] = []
    rejected: list[dict[str, Any]] = []

    while rollouts + 2 <= rollout_budget:
        # "select from Pareto frontier of P — not just the best-scoring"
        frontier = pareto_frontier(pool)
        parent = min(frontier, key=lambda c: (c.score, c.id))

        train_scores = evaluate(parent.instruction, train_ids)
        probe = Candidate(id=f"{parent.id}::probe", instruction=parent.instruction,
                          harness_identity=identity, model_identity=model_identity,
                          scores=train_scores)
        feedback = textual_feedback(probe, instances)

        if rollouts + 1 > rollout_budget:
            break

        proposed = reflect(reflection_messages(parent, feedback))
        if not isinstance(proposed, str) or not proposed.strip():
            rejected.append({"parent": parent.id, "reason": "reflector returned nothing"})
            break
        if not is_indonesian_enough(proposed):
            rejected.append({"parent": parent.id,
                             "reason": "mutated instruction is not Indonesian"})
            on_event({"event": "rejected", "reason": "not_indonesian"})
            break

        child = Candidate(
            id=f"c{len(pool)}", instruction=proposed.strip(),
            harness_identity=identity, model_identity=model_identity,
            scores=evaluate(proposed, holdout_ids), parent=parent.id,
            note="reflected")

        # Promotion is judged against the SEED, not against `parent`. `parent`
        # selects where to explore next and may itself be a rejected variant;
        # the deployed baseline is what a promotion would replace.
        ok, why = accepts(child, baseline)
        on_event({"event": "candidate", "id": child.id, "accepted": ok,
                  "why": why, "score": child.score, "rollouts": rollouts})
        if ok:
            pool.append(child)
            accepted.append(child)
        else:
            # A rejected child still records which instances it moved, and the
            # archive exists so a complementary variant is not thrown away for
            # being worse on average. It joins the pool but never `accepted`,
            # so it can seed a later iteration without being reported as a win.
            rejected.append({"parent": parent.id, "child": child.id,
                             "reason": why, "score": child.score,
                             "measured_against": baseline.id})
            pool.append(child)

    # The archive (pool) keeps rejected variants so a complementary one can
    # seed a later iteration. `best` is a PROMOTION decision and must therefore
    # come from what was actually accepted: a candidate refused for regressing
    # an instance still has a higher mean, and reporting it as best would
    # reintroduce by the back door the aggregate rule acceptance just refused.
    frontier = pareto_frontier(pool)
    promotable = [baseline, *accepted]
    best = max(promotable, key=lambda c: (c.score, c.id))

    return {
        "slice": DEFAULT_SLICE,
        "seed": baseline.to_dict(),
        "best": best.to_dict(),
        # The frontier is the ARCHIVE view and may contain refused variants;
        # `best` above is the promotion decision. Reported separately so the
        # two are never confused.
        "frontier": [c.to_dict() for c in frontier],
        "promotable": [c.id for c in promotable],
        "accepted": [c.to_dict() for c in accepted],
        "rejected": rejected,
        "rollouts": rollouts,
        "rollout_budget": rollout_budget,
        "harness_identity": identity,
        "model_identity": model_identity,
        # Named so a reader can see what a promotion was measured against.
        "promotion_baseline": baseline.id,
        "measured": True,
        "training_authorized": False,
    }


# --- CLI --------------------------------------------------------------------


def dry_run_plan(rollout_budget: int = DEFAULT_ROLLOUT_BUDGET,
                 slice_name: str = DEFAULT_SLICE) -> list[str]:
    """What a run WOULD do. Writes nothing, contacts nothing."""
    return [
        f"slice                : {slice_name}",
        f"rollout budget       : {rollout_budget} (explicit; no hidden loops)",
        "gold set             : data/gold (BLOCKED: 0 approved items)",
        "rollout transport    : src/bridge_client.py (TeacherClient)",
        "reflection language  : Indonesian",
        "acceptance           : held-out slice, no per-instance regression",
        "selection            : Pareto frontier, not best mean score",
        "identity             : harness_distill.canonical_digest per candidate",
        "weights              : untouched (inference only)",
        "training_authorized  : false",
        "",
        "sequence per iteration:",
        "  1. select a candidate from the Pareto frontier",
        "  2. run the system on the TRAIN slice          (1 rollout)",
        "  3. build metric + textual feedback in Indonesian",
        "  4. reflect to propose a mutated instruction   (1 reflection call)",
        "  5. score the mutation on the HELD-OUT slice   (1 rollout)",
        "  6. accept only if no held-out instance regressed",
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GEPA reflective prompt evolution. Inference only; "
                    "optimises prompts and tool schemas, never weights.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command/rollout sequence and write nothing")
    parser.add_argument("--rollout-budget", type=int, default=DEFAULT_ROLLOUT_BUDGET)
    parser.add_argument("--slice", default=DEFAULT_SLICE, choices=list(gs.SPLITS))
    args = parser.parse_args(argv)

    if args.dry_run:
        print("GEPA dry run — nothing was executed and nothing was written.\n")
        for line in dry_run_plan(args.rollout_budget, args.slice):
            print(f"  {line}")
        return

    # A live run needs an approved gold set, which is BLOCKED by construction.
    gold = gs.load(gs.GOLD_DIR)
    reasons = gs.gate(gold)
    if reasons:
        die("cannot run against the gold set: " + "; ".join(reasons))
    die("live GEPA runs are not wired in this phase; use --dry-run")


if __name__ == "__main__":
    main()
