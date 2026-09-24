"""Bounded GEPA runner: the tested `gepa.optimize()` controller behind a CLI.

    # what would run; writes nothing, contacts nothing
    python src/run_gepa.py plan \
        --train-set data/gepa/train.jsonl --holdout-gold data/gold \
        --model-registry qwen35-9b-instruct --endpoint http://127.0.0.1:8000/v1 \
        --seed-instruction-file configs/prompts/seed_id.txt

    # a live run needs --real and an APPROVED held-out gold set
    python src/run_gepa.py run ... --output output/rsi/gepa/candidate-001 --real

`src/gepa.py --dry-run` is untouched and remains the zero-cost preview.

TWO DATASETS, KEPT APART. `--train-set` is material GEPA may learn from
(approved synthetic is fine); `--holdout-gold` is the human-authored gold that
decides acceptance. The proposer (the reflector) sees train failures and
bounded train feedback. It never sees a held-out prompt, id, expected answer,
verifier configuration or per-item held-out feedback: the held-out slice
reaches `gepa.optimize` only as ids whose scores are booleans, and the
reflection messages are built from train probes alone. The only thing the
loop learns from the held-out slice is accept or reject.

ISOLATION. The candidate directory must lie outside `configs/`; the production
harness files are read to compute the base identity and never written. A
directory that already holds candidate evidence is refused, so evidence is
append-only by construction.

REUSED: `gepa.optimize` (the controller, Pareto acceptance, no-per-instance-
regression rule), `gold_set` (loading and gating), `verifiers` (scoring),
`run_gold_evaluation` (registry identity, served-model check, teacher check,
fake/real client shape, atomic writes), `harness_distill.canonical_digest`
through `gepa.harness_identity`, `bridge_client.TeacherClient` for the wire.

NO TRAINING. `train/TRAINING_BLOCKED.md` is controlling. Every artifact carries
`training_authorized: false`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gepa                                                # noqa: E402
import gold_set as gs                                      # noqa: E402
import harness_distill as hd                               # noqa: E402
import harness_eval as he                                  # noqa: E402
import run_gold_evaluation as rge                          # noqa: E402
import verifiers                                           # noqa: E402

DEFAULT_HARNESS = "tantular-office-candidate"
DEFAULT_SLICE = gepa.DEFAULT_SLICE
DEFAULT_ROLLOUT_BUDGET = gepa.DEFAULT_ROLLOUT_BUDGET
DECODING = rge.DECODING
CONFIG_ROOT = (ROOT / "configs").resolve()

RESULT_FILE = "result.json"
CANDIDATES_FILE = "candidates.jsonl"
RESTRICTED_DIR = "restricted"
PER_INSTANCE_FILE = "per_instance.jsonl"


class GepaRunError(Exception):
    """The runner could not proceed. Never the same thing as a candidate losing."""


def die(msg: str, code: int = 2) -> None:
    print(f"\nGEPA RUN REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


# --- clients and reflectors -------------------------------------------------


class FakeGepaClient(rge.FakeGoldClient):
    """Replays outputs per (instruction, item). Never a measurement source.

    The bundle is {"served": [...], "default": {item: output},
    "outputs": {instruction: {item: output}}, "reflections": [...]}.
    An instruction the bundle does not name falls back to `default`, so a
    fixture declares only what it cares about.
    """

    def __init__(self, bundle: dict[str, Any], *, endpoint: str = "fake://bundle"):
        if not isinstance(bundle, dict) or not isinstance(bundle.get("default"), dict):
            raise GepaRunError("fake bundle must carry a 'default' outputs object")
        outputs = bundle.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise GepaRunError("fake bundle 'outputs' must map instruction -> outputs")
        super().__init__({"served": bundle.get("served") or [], "outputs": bundle["default"]},
                         endpoint=endpoint)
        self.by_instruction = outputs
        self.default = bundle["default"]

    def answer(self, record: dict[str, Any], decoding: dict[str, Any],
               instruction: str = "") -> Any:
        table = self.by_instruction.get(instruction, self.default)
        if record["id"] not in table:
            raise KeyError(f"fake bundle declares no output for {record['id']!r}")
        return table[record["id"]]


class BridgeGepaClient(rge.BridgeGoldClient):
    """The real client: the candidate instruction is the system message, the
    item prompt is the user message. Constructed only behind --real."""

    def answer(self, record: dict[str, Any], decoding: dict[str, Any],
               instruction: str = "") -> Any:
        async def call() -> dict[str, Any]:
            async with self._httpx.AsyncClient() as http:
                return await self.client.complete(http, [
                    {"role": "system", "content": str(instruction)},
                    {"role": "user", "content": str(record["prompt"])},
                ])
        return asyncio.run(call())["content"]


class SequencedReflector:
    """Replays declared mutations in order; the last one repeats."""

    def __init__(self, texts: list[str]):
        if not isinstance(texts, list) or not texts:
            raise GepaRunError("a fake reflector needs at least one reflection text")
        self.texts = [str(t) for t in texts]
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]


class BridgeReflector:
    """The real reflector: the same endpoint answers the Indonesian reflection
    prompt built by gepa.reflection_messages. Constructed only behind --real."""

    def __init__(self, client: BridgeGepaClient):
        self.client = client
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)

        async def call() -> dict[str, Any]:
            async with self.client._httpx.AsyncClient() as http:
                return await self.client.client.complete(http, messages)
        return asyncio.run(call())["content"]


def load_fake_bundle(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GepaRunError(f"cannot read fake bundle {path}: {error}") from error
    if not isinstance(value, dict):
        raise GepaRunError("fake bundle must be a JSON object")
    return value


# --- datasets ---------------------------------------------------------------


def load_train_set(path: Path, slice_name: str) -> list[dict[str, Any]]:
    """Structurally valid gold-format records in the slice. Synthetic is
    allowed here: this is what GEPA learns from, not what accepts it."""
    path = Path(path)
    if not path.is_file():
        raise GepaRunError(f"train set does not exist: {path}")
    try:
        records = gs.parse_jsonl(path.read_text(encoding="utf-8"))
    except gs.GoldSetError as error:
        raise GepaRunError(f"train set: {error}") from error
    problems: list[str] = []
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        issues = gs.validate_record(record, index=index)
        if issues:
            problems.extend(issues)
            continue
        if record["id"] in seen:
            problems.append(f"line {index}: duplicate id {record['id']!r}")
            continue
        seen.add(record["id"])
        valid.append(record)
    if problems:
        raise GepaRunError(f"train set has {len(problems)} problem(s); first: {problems[0]}")
    in_slice = [r for r in valid if r["split"] == slice_name]
    if not in_slice:
        raise GepaRunError(f"train set has no items in slice {slice_name!r}")
    blockers = rge.executor_blockers(in_slice)
    if blockers:
        raise GepaRunError(f"train set: {blockers[0]}")
    return sorted(in_slice, key=lambda r: str(r["id"]))


def holdout_items(holdout_gold: Path, slice_name: str, *,
                  allow_fixture: bool) -> tuple[gs.GoldSet, list[dict[str, Any]]]:
    gold = rge.load_gold(holdout_gold)
    try:
        items = rge.select_items(gold, allow_fixture=allow_fixture)
    except rge.GoldEvaluationError as error:
        raise GepaRunError(str(error)) from error
    in_slice = [r for r in items if r["split"] == slice_name]
    if not in_slice:
        raise GepaRunError(f"held-out gold has no items in slice {slice_name!r}")
    return gold, in_slice


def _overlap(train: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> list[str]:
    train_ids = {str(r["id"]) for r in train}
    holdout_ids = {str(r["id"]) for r in holdout}
    prompts = {str(r["prompt"]).strip() for r in train}
    by_prompt = [str(r["id"]) for r in holdout if str(r["prompt"]).strip() in prompts]
    return sorted(train_ids & holdout_ids) + by_prompt


def _instruction_digest(text: str) -> str:
    return he.digest_payload(str(text))


def _candidate_harness_digest(base_digest: str, instruction: str) -> str:
    """Identity of 'this harness with this instruction'. The base harness
    file is never rewritten; the candidate is the pair."""
    return hd.canonical_digest({"base_harness_digest": base_digest,
                                "instruction_sha256": _instruction_digest(instruction)})


def _isolated(output: Path) -> Path:
    output = Path(output).resolve()
    if output == CONFIG_ROOT or CONFIG_ROOT in output.parents:
        raise GepaRunError(
            f"{output} lies inside {CONFIG_ROOT}; a GEPA run must write to an "
            "isolated candidate directory, never into the production configs")
    for name in (RESULT_FILE, CANDIDATES_FILE):
        if (output / name).exists():
            raise GepaRunError(
                f"{output / name} already exists; candidate evidence is "
                "append-only and a run never overwrites another. Use a new "
                "--output.")
    return output


# --- plan -------------------------------------------------------------------


def plan_gepa(*, train_set: Path, holdout_gold: Path, model_registry: str,
              model_dir: Path = rge.MODEL_DIR, endpoint: str,
              harness: str = DEFAULT_HARNESS, slice_name: str = DEFAULT_SLICE,
              rollout_budget: int = DEFAULT_ROLLOUT_BUDGET, allow_fixture: bool = False,
              seed_instruction: str, experiment_path: Path = rge.EXPERIMENT_PATH,
              ) -> dict[str, Any]:
    """What would run. Writes nothing, constructs no client."""
    blockers: list[str] = []
    model = None
    try:
        model = rge.resolve_model(model_dir, model_registry)
    except rge.GoldEvaluationError as error:
        blockers.append(str(error))
    if rollout_budget < 2:
        blockers.append("rollout_budget must allow at least a baseline and one mutation")
    if not str(seed_instruction).strip():
        blockers.append("a seed instruction is required")
    elif not gepa.is_indonesian_enough(seed_instruction):
        blockers.append("the seed instruction is not Indonesian")

    harness_digest = None
    try:
        harness_digest = gepa.harness_identity(hd.load_harness(harness))
    except (gepa.GepaError, hd.HarnessPlanError) as error:
        blockers.append(f"harness {harness!r}: {error}")

    train: list[dict[str, Any]] = []
    try:
        train = load_train_set(train_set, slice_name)
    except GepaRunError as error:
        blockers.append(str(error))

    gold = rge.load_gold(holdout_gold)
    blockers.extend(rge.gold_blockers(gold, allow_fixture=allow_fixture))
    pool = list(gold.valid if allow_fixture else gold.eligible)
    holdout = [r for r in pool if r["split"] == slice_name]
    blockers.extend(rge.executor_blockers(holdout))
    if not blockers and not holdout:
        blockers.append(f"held-out gold has no items in slice {slice_name!r}")
    overlap = _overlap(train, holdout)
    if overlap:
        blockers.append(f"train and held-out overlap on {len(overlap)} item(s)")

    return {
        "phase": "plan",
        "slice": slice_name,
        "rollout_budget": rollout_budget,
        "harness": harness,
        "harness_digest": harness_digest,
        "model_registry": model_registry,
        "model_identity": {"expected": model["model_id"] if model else None,
                           "revision": model["revision"] if model else None},
        "endpoint": endpoint,
        "train_set": str(train_set),
        "train_count": len(train),
        "holdout_gold": str(holdout_gold),
        "holdout_gold_digest": rge.gold_set_digest(gold),
        "holdout_count": len(holdout),
        "holdout_posture": "fixture" if allow_fixture else "production",
        "seed_instruction_sha256": _instruction_digest(seed_instruction),
        "decoding": dict(DECODING),
        "proposer_sees": ["train failures", "bounded train feedback",
                          "aggregate held-out accept/reject"],
        "proposer_never_sees": ["held-out prompts", "held-out ids",
                                "expected answers", "verifier internals"],
        "blockers": blockers,
        "executable": not blockers,
        "training_authorized": False,
    }


# --- run --------------------------------------------------------------------


def run_gepa(*, train_set: Path, holdout_gold: Path, model_registry: str,
             model_dir: Path = rge.MODEL_DIR, client: Any,
             reflect: Callable[[list[dict[str, str]]], str],
             harness: str = DEFAULT_HARNESS, slice_name: str = DEFAULT_SLICE,
             rollout_budget: int = DEFAULT_ROLLOUT_BUDGET, output: Path,
             allow_fixture: bool = False, seed_instruction: str, seed: int = 0,
             run_id: str = "gepa-1", experiment_path: Path = rge.EXPERIMENT_PATH,
             ) -> dict[str, Any]:
    """One bounded GEPA run. Every refusal happens before the client is asked."""
    if not str(seed_instruction).strip():
        raise GepaRunError("a seed instruction is required")
    model = rge.resolve_model(model_dir, model_registry)
    gold, holdout = holdout_items(holdout_gold, slice_name, allow_fixture=allow_fixture)
    train = load_train_set(train_set, slice_name)
    overlap = _overlap(train, holdout)
    if overlap:
        raise GepaRunError(
            f"train and held-out slices overlap on {len(overlap)} item(s) "
            f"(first: {overlap[0]!r}); acceptance must be on unseen items")
    spec = hd.load_harness(harness)
    base_digest = gepa.harness_identity(spec)
    output = _isolated(output)
    teachers = rge.teacher_model_ids(experiment_path)

    identity = client.identity()
    rge.check_served(model, identity)
    recorded_identity = {
        "expected": model["model_id"],
        "served": [str(s) for s in identity["served"]],
        "endpoint": str(identity.get("endpoint") or ""),
        "revision": model["revision"],
    }
    rge.check_not_teacher(recorded_identity, teachers, "the served model")
    is_fake = identity.get("produces_real_measurements") is not True
    if not is_fake and allow_fixture:
        raise GepaRunError("a real client cannot run in fixture posture; drop --allow-fixture")
    model_identity = f"{model['model_id']}@{model['revision']}"

    by_id = {str(r["id"]): r for r in train + holdout}
    train_ids = [str(r["id"]) for r in train]
    holdout_ids = [str(r["id"]) for r in holdout]
    # The proposer's view: train prompts and a bounded failure note. Held-out
    # entries are present only so optimize() knows the ids; they carry nothing.
    instances: dict[str, dict[str, Any]] = {}
    for record in train:
        instances[str(record["id"])] = {
            "prompt": str(record["prompt"]),
            "failure": f"keluaran tidak memenuhi verifier {record['verifier']['type']}",
        }
    for record in holdout:
        instances[str(record["id"])] = {}

    rollouts: list[dict[str, Any]] = []

    def run_system(instruction: str, ids: list[str]) -> dict[str, bool]:
        scores: dict[str, bool] = {}
        errors: dict[str, str] = {}
        for item_id in ids:
            record = by_id[item_id]
            try:
                raw = client.answer(record, dict(DECODING), instruction)
                result = verifiers.verify(record, rge._shape_output(record, raw))
                scores[item_id] = bool(result.passed and result.counts_toward_correctness)
            except Exception as exc:                      # noqa: BLE001
                # A failed rollout is a failed instance, and it is recorded.
                scores[item_id] = False
                errors[item_id] = f"{type(exc).__name__}: {exc}"
        kind = "holdout" if set(ids) == set(holdout_ids) else "train"
        rollouts.append({"rollout": len(rollouts) + 1, "kind": kind,
                         "instruction": str(instruction),
                         "instruction_sha256": _instruction_digest(instruction),
                         "scores": dict(scores), "errors": errors})
        return scores

    events: list[dict[str, Any]] = []
    try:
        result = gepa.optimize(
            seed_instruction=seed_instruction, instances=instances,
            run_system=run_system, reflect=reflect, harness_spec=spec,
            model_identity=model_identity, train_ids=train_ids,
            holdout_ids=holdout_ids, rollout_budget=rollout_budget,
            on_event=events.append)
    except gepa.GepaError as error:
        raise GepaRunError(str(error)) from error

    # --- candidate evidence. Instruction TEXT is kept in result.json (it is
    # the artifact under optimisation); candidates.jsonl carries digests and
    # decisions; per-instance booleans go to the restricted file only.
    train_by_sha: dict[str, dict[str, Any]] = {}
    for row in rollouts:
        if row["kind"] == "train":
            train_by_sha[row["instruction_sha256"]] = {
                "instances": len(row["scores"]),
                "passed": sum(1 for v in row["scores"].values() if v),
            }
    holdout_rollouts = [r for r in rollouts if r["kind"] == "holdout"]
    decisions = [e for e in events if e.get("event") == "candidate"]
    accepted_ids = {c["id"] for c in result["accepted"]}
    rejected_by_child = {r.get("child"): r for r in result["rejected"] if r.get("child")}

    candidates: list[dict[str, Any]] = []

    def record_candidate(cid: str, parent: str | None, instruction: str,
                         decision: str, reason: str, rollouts_at: int) -> None:
        promotable = cid == "seed" or cid in accepted_ids
        candidates.append({
            "candidate_id": cid,
            "parent_id": parent,
            "instruction_sha256": _instruction_digest(instruction),
            "base_harness_digest": base_digest,
            "candidate_harness_digest": _candidate_harness_digest(base_digest, instruction),
            "model_identity": recorded_identity,
            "train_summary": train_by_sha.get(_instruction_digest(instruction)),
            "holdout_decision": decision,
            "reason": reason,
            "rollouts": rollouts_at,
            "rollout_budget": rollout_budget,
            "archive_only": not promotable,
            "promotable": promotable,
            "training_authorized": False,
        })

    record_candidate("seed", None, seed_instruction, "baseline", "seed",
                     next((e["rollouts"] for e in events if e.get("event") == "seed"), 1))
    for index, event in enumerate(decisions):
        cid = event["id"]
        rollout = holdout_rollouts[index + 1] if index + 1 < len(holdout_rollouts) else None
        parent = rejected_by_child.get(cid, {}).get("parent") or next(
            (c["parent"] for c in result["accepted"] if c["id"] == cid), None)
        record_candidate(cid, parent, rollout["instruction"] if rollout else "",
                         "accepted" if event["accepted"] else "rejected",
                         str(event.get("why", "")), int(event.get("rollouts", 0)))
    for entry in result["rejected"]:
        if "child" not in entry:
            record_candidate(f"proposal-{len(candidates)}", entry.get("parent"), "",
                             "rejected_before_evaluation", str(entry.get("reason", "")),
                             result["rollouts"])

    output.mkdir(parents=True, exist_ok=True)
    (output / RESTRICTED_DIR).mkdir(exist_ok=True)
    with (output / CANDIDATES_FILE).open("a", encoding="utf-8") as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / RESTRICTED_DIR / PER_INSTANCE_FILE).open("a", encoding="utf-8") as stream:
        for row in rollouts:
            restricted = {k: v for k, v in row.items() if k != "instruction"}
            restricted["training_authorized"] = False
            stream.write(json.dumps(restricted, ensure_ascii=False, sort_keys=True) + "\n")

    full = {
        **result,
        "slice": slice_name,
        "run_id": run_id,
        "seed_parameter": seed,
        "harness": harness,
        "base_harness_digest": base_digest,
        "model_registry": model_registry,
        "model_identity_record": recorded_identity,
        "client": {"kind": str(identity.get("kind")),
                   "produces_real_measurements": not is_fake},
        "fixture": is_fake,
        "holdout_gold_digest": rge.gold_set_digest(gold),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "decoding": dict(DECODING),
        "candidates_written": len(candidates),
        "training_authorized": False,
    }
    try:
        rge._write_atomic(output / RESULT_FILE, full)
    except rge.GoldEvaluationError as error:
        raise GepaRunError(str(error)) from error
    return full


# --- CLI --------------------------------------------------------------------


def _seed_text(args: argparse.Namespace) -> str:
    if args.seed_instruction_file:
        path = Path(args.seed_instruction_file)
        if not path.is_file():
            raise GepaRunError(f"seed instruction file does not exist: {path}")
        return path.read_text(encoding="utf-8").strip()
    if args.seed_instruction:
        return str(args.seed_instruction)
    raise GepaRunError("pass --seed-instruction or --seed-instruction-file")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-set", type=Path, required=True,
                        help="JSONL of gold-format items GEPA may learn from")
    parser.add_argument("--holdout-gold", type=Path, default=gs.GOLD_DIR,
                        help="directory of human-authored gold used only for acceptance")
    parser.add_argument("--model-registry", required=True)
    parser.add_argument("--model-dir", type=Path, default=rge.MODEL_DIR)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--harness", default=DEFAULT_HARNESS,
                        help=f"base harness under configs/harnesses (default {DEFAULT_HARNESS})")
    parser.add_argument("--slice", default=DEFAULT_SLICE, choices=list(gs.SPLITS),
                        help=f"split to optimise (default {DEFAULT_SLICE})")
    parser.add_argument("--rollout-budget", type=int, default=DEFAULT_ROLLOUT_BUDGET,
                        help=f"explicit rollout budget (default {DEFAULT_ROLLOUT_BUDGET})")
    parser.add_argument("--seed-instruction", default="")
    parser.add_argument("--seed-instruction-file", type=Path)
    parser.add_argument("--experiment", type=Path, default=rge.EXPERIMENT_PATH)
    parser.add_argument("--allow-fixture", action="store_true",
                        help="accept a synthetic fixture held-out set (fake client only)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("plan", help="what would run (writes nothing)")
    _common(p)

    r = commands.add_parser("run", help="one bounded GEPA run into an isolated directory")
    _common(r)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--run-id", default="gepa-1")
    r.add_argument("--seed", type=int, default=0, help="recorded for reproducibility")
    r.add_argument("--real", action="store_true",
                   help="contact --endpoint through bridge_client; requires an "
                        "approved held-out gold set")
    r.add_argument("--fake-bundle", type=Path,
                   help="deterministic fake client and reflector from this bundle")
    r.add_argument("--api-key-env", default="")

    args = parser.parse_args(argv)
    try:
        seed_instruction = _seed_text(args)
        if args.command == "plan":
            result = plan_gepa(
                train_set=args.train_set, holdout_gold=args.holdout_gold,
                model_registry=args.model_registry, model_dir=args.model_dir,
                endpoint=args.endpoint, harness=args.harness, slice_name=args.slice,
                rollout_budget=args.rollout_budget, allow_fixture=args.allow_fixture,
                seed_instruction=seed_instruction, experiment_path=args.experiment)
        else:
            if args.real and args.fake_bundle:
                raise GepaRunError("--real and --fake-bundle are exclusive")
            if args.real and args.allow_fixture:
                raise GepaRunError("--real cannot be combined with --allow-fixture")
            if not args.real and not args.fake_bundle:
                raise GepaRunError(
                    "no client selected. Pass --fake-bundle <file> for a "
                    "deterministic fake run, or --real to contact --endpoint. "
                    "The default posture of this tool is plan.")
            # The held-out gate is decided before any client exists.
            holdout_items(args.holdout_gold, args.slice, allow_fixture=args.allow_fixture)
            model = rge.resolve_model(args.model_dir, args.model_registry)
            if args.real:
                key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
                client: Any = BridgeGepaClient(args.endpoint, model["model_id"], api_key=key)
                reflect: Any = BridgeReflector(client)
            else:
                bundle = load_fake_bundle(args.fake_bundle)
                client = FakeGepaClient(bundle, endpoint=args.endpoint)
                reflect = SequencedReflector(bundle.get("reflections") or [])
            full = run_gepa(
                train_set=args.train_set, holdout_gold=args.holdout_gold,
                model_registry=args.model_registry, model_dir=args.model_dir,
                client=client, reflect=reflect, harness=args.harness,
                slice_name=args.slice, rollout_budget=args.rollout_budget,
                output=args.output, allow_fixture=args.allow_fixture,
                seed_instruction=seed_instruction, seed=args.seed,
                run_id=args.run_id, experiment_path=args.experiment)
            result = {
                "phase": "run",
                "output": str(Path(args.output).resolve()),
                "rollouts": full["rollouts"],
                "rollout_budget": full["rollout_budget"],
                "accepted": [c["id"] for c in full["accepted"]],
                "best": full["best"]["id"],
                "fixture": full["fixture"],
                "training_authorized": False,
            }
    except (GepaRunError, rge.GoldEvaluationError, gepa.GepaError,
            hd.HarnessPlanError) as error:
        die(str(error))
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
