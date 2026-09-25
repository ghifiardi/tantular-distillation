"""Gold evaluation runner: receipts and measurements for the separation gate.

    # what would run, against which identity; writes nothing, contacts nothing
    python src/run_gold_evaluation.py plan \
        --gold-dir data/gold --model-registry qwen35-9b-instruct \
        --endpoint http://127.0.0.1:8000/v1 --repetitions 2

    # one receipt per (item, repetition); a REAL endpoint needs --real
    python src/run_gold_evaluation.py run \
        --gold-dir data/gold --model-registry qwen35-9b-instruct \
        --endpoint http://127.0.0.1:8000/v1 --repetitions 2 \
        --output output/rsi/9b-receipts --real

    # derive the measurement the separation gate reads, from receipts only
    python src/run_gold_evaluation.py aggregate \
        --gold-dir data/gold --receipts output/rsi/9b-receipts \
        --arm student_9b --output output/rsi/measurement-9b.json

    python src/eval_harness.py separation-gate \
        --stronger output/rsi/measurement-9b.json \
        --weaker  output/rsi/measurement-4b.json

Three phases, kept apart on purpose. `plan` is the default posture and touches
nothing. `run` talks to exactly one model client -- behind `--real`, either
`bridge_client.TeacherClient` (`--client openai`) or the Ollama /api/chat
adapter in `ollama_chat_client.py` (`--client ollama`, think:false, chosen
explicitly and checked against the registry entry's `serving:` block, never
inferred from the endpoint); or a deterministic fake behind `--fake-outcomes`
-- and leaves
a receipt for every item and repetition, failures included. `aggregate` derives
a measurement from validated receipts and from nothing else: there is no code
path that accepts a pass rate from outside.

WHAT A RECEIPT DOES NOT CONTAIN. The expected answer, the verifier's schema or
claims, the raw model output, and the verifier's free-text diagnosis all stay
out. A receipt carries the gold-set digest, the item id and split, the model
identity (expected, served, endpoint, revision), the decoding configuration,
bounded verifier booleans, digests of the output and the diagnosis, and
`training_authorized: false`. It is provenance, not a copy of the test.

REUSED, NOT REIMPLEMENTED: `gold_set` loads and gates the items; `verifiers`
judges outputs; `model_ids` matches served identities; `eval_harness` checks
for a teacher in a student slot and validates the measurement shape;
`harness_eval.digest_payload` is the one digest construction; the atomic
receipt write mirrors `harness_eval.write_receipt`.

NO TRAINING. `train/TRAINING_BLOCKED.md` is controlling. Every artifact this
module writes carries `training_authorized: false`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import eval_harness as eh                                  # noqa: E402
import gold_set as gs                                      # noqa: E402
import harness_distill as hd                               # noqa: E402
import harness_eval as he                                  # noqa: E402
import model_ids                                           # noqa: E402
import verifiers                                           # noqa: E402
from run_harness_evaluation import COMMIT_RE                # noqa: E402

MODEL_DIR = ROOT / "configs" / "models"
EXPERIMENT_PATH = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"

RECEIPT_SCHEMA = 1
MEASUREMENT_SCHEMA = 1
RECEIPT_KIND = "gold_evaluation_receipt"
MEASUREMENT_KIND = "gold_measurement"

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUSES = (STATUS_OK, STATUS_ERROR)

# Fixed decoding for every gold run. Recorded on every receipt so two
# measurements can be shown to have asked the model the same way. Stage 6 of
# the handoff requires the owner to accept these before a live run.
DECODING: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "seed": 0,
}

MINIMUM_REPETITIONS = 2

# Which wire a registry entry is qualified for, and which CLI client speaks it.
PROTOCOL_OPENAI = "openai_chat"
PROTOCOL_OLLAMA = "ollama_chat"
CLIENT_OPENAI = "openai"
CLIENT_OLLAMA = "ollama"
CLIENTS = (CLIENT_OPENAI, CLIENT_OLLAMA)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

_RECEIPT_REQUIRED = (
    "schema_version", "kind", "gold_set_digest", "item_id", "split",
    "score_role", "verifier_type", "model_registry", "model_identity",
    "client", "run_id", "repetition", "repetitions_planned", "decoding",
    "started_at", "ended_at", "status", "error", "output_sha256",
    "output_chars", "verifier_result", "training_authorized",
)


class GoldEvaluationError(Exception):
    """The runner could not proceed. Never the same thing as an item failing."""


def die(msg: str, code: int = 2) -> None:
    print(f"\nGOLD EVALUATION REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


# --- identity ---------------------------------------------------------------


def resolve_model(model_dir: Path, name: str) -> dict[str, Any]:
    """The registry identity a receipt binds to. Same rules as
    run_harness_evaluation._model_spec: an exact model_id and a full commit."""
    path = Path(model_dir) / f"{name}.yaml"
    try:
        spec = hd.load_yaml(path)
    except hd.HarnessPlanError as error:
        raise GoldEvaluationError(str(error)) from error
    model_id = spec.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise GoldEvaluationError(f"registry model {name!r} declares no model_id")
    revision = spec.get("revision")
    if not isinstance(revision, str) or not COMMIT_RE.fullmatch(revision):
        raise GoldEvaluationError(
            f"registry model {name!r} has no pinned revision ({revision!r}); a "
            "receipt must name the exact checkpoint it ran against")
    role = spec.get("role")
    if role != "student":
        # The separation gate compares STUDENTS. A teacher here is the failure
        # run_gates.verify_served_model exists to prevent, caught at the
        # registry before any endpoint is involved.
        raise GoldEvaluationError(
            f"registry model {name!r} has role {role!r}; a gold evaluation arm "
            "must be a student. A teacher identity in a student slot measures "
            "nothing.")
    return {"model_registry": name, "model_id": model_id, "revision": revision,
            "serving": serving_declaration(spec, name)}


def serving_declaration(spec: dict[str, Any], name: str) -> dict[str, Any]:
    """How the registry says this checkpoint is served, if it says.

    A `serving:` block names the protocol and the served tags the runtime
    identity may match. Absent means the OpenAI-compatible path with the
    model_id itself as the served name, which is what every vLLM-served entry
    already assumes.
    """
    block = spec.get("serving")
    if block is None:
        return {"protocol": None, "tags": []}
    if not isinstance(block, dict):
        raise GoldEvaluationError(f"registry model {name!r}: serving must be a mapping")
    protocol = block.get("protocol")
    if protocol not in (PROTOCOL_OPENAI, PROTOCOL_OLLAMA):
        raise GoldEvaluationError(
            f"registry model {name!r}: serving.protocol {protocol!r} is not one of "
            f"{[PROTOCOL_OPENAI, PROTOCOL_OLLAMA]}")
    tags = block.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags):
        raise GoldEvaluationError(
            f"registry model {name!r}: serving.tags must be a list of tag strings")
    if protocol == PROTOCOL_OLLAMA and not tags:
        raise GoldEvaluationError(
            f"registry model {name!r}: an Ollama-served entry must declare the "
            "tag(s) the served identity may match")
    return {"protocol": protocol, "tags": list(tags)}


def check_client_protocol(model: dict[str, Any], client_name: str) -> None:
    """The client must be the one the registry entry was qualified for.

    An Ollama-served Qwen3.5 driven through /v1 ignores think:false and can
    answer nothing; an OpenAI-served entry driven through /api/chat names a
    route that does not exist. Either way the measurement would be of the
    wrong thing, so the pairing is checked, not assumed.
    """
    declared = (model.get("serving") or {}).get("protocol")
    if declared == PROTOCOL_OLLAMA and client_name != CLIENT_OLLAMA:
        raise GoldEvaluationError(
            f"registry model {model['model_registry']!r} is served by Ollama "
            f"({PROTOCOL_OLLAMA}) and must be driven through /api/chat with "
            f"think:false; the {client_name!r} client is refused. Pass --client "
            f"{CLIENT_OLLAMA}.")
    if declared != PROTOCOL_OLLAMA and client_name == CLIENT_OLLAMA:
        raise GoldEvaluationError(
            f"registry model {model['model_registry']!r} declares no Ollama "
            "serving tags; the ollama client cannot attribute a served tag to it")


def teacher_model_ids(experiment_path: Path) -> list[str]:
    """Teacher identities the experiment declares, resolved through the
    production registry so the check is against a model_id, not a nickname."""
    try:
        experiment = hd.load_yaml(Path(experiment_path))
    except hd.HarnessPlanError as error:
        raise GoldEvaluationError(str(error)) from error
    names = [experiment.get("teacher_model")]
    ids: list[str] = []
    for name in names:
        if not name:
            continue
        path = MODEL_DIR / f"{name}.yaml"
        if not path.is_file():
            continue
        spec = hd.load_yaml(path)
        model_id = spec.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id)
    return ids


def check_not_teacher(identity: dict[str, Any], teachers: list[str],
                      where: str) -> None:
    found = eh.identity_is_teacher(identity, teachers)
    if found:
        raise GoldEvaluationError(
            f"{where} is the teacher {found!r}; a teacher identity in a "
            "student slot measures nothing")


def check_served(model: dict[str, Any], identity: dict[str, Any]) -> None:
    served = identity.get("served")
    if not isinstance(served, list) or not served:
        raise GoldEvaluationError(
            "the client reports no served model; an endpoint that cannot name "
            "what it serves cannot be attributed")
    accepted = [model["model_id"], *((model.get("serving") or {}).get("tags") or [])]
    if not any(model_ids.any_match(name, served) for name in accepted):
        raise GoldEvaluationError(
            f"served model(s) {served!r} do not match the expected registry "
            f"identity {model['model_id']!r} or its declared serving tags "
            f"{accepted[1:]!r} (endpoint {identity.get('endpoint')!r}); refusing "
            "to evaluate a model other than the one planned")


# --- gold selection ---------------------------------------------------------


def load_gold(gold_dir: Path) -> gs.GoldSet:
    try:
        return gs.load(Path(gold_dir))
    except gs.GoldSetError as error:
        raise GoldEvaluationError(str(error)) from error


def gold_set_digest(gold: gs.GoldSet) -> str:
    """Digest over the structurally valid records, in id order.

    Over the records, so a changed prompt or expected answer changes the
    digest; only the digest ever leaves this function.
    """
    return he.digest_payload(sorted(gold.valid, key=lambda r: str(r["id"])))


def executor_blockers(items: list[dict[str, Any]]) -> list[str]:
    """Executor items the verifier layer cannot safely run (production
    records on the timeout-bounded backend)."""
    blockers: list[str] = []
    for record in items:
        if record["verifier"]["type"] != "executor":
            continue
        allowed, why = verifiers.executor_backend_allows(record)
        if not allowed:
            blockers.append(why)
    return blockers


def gold_blockers(gold: gs.GoldSet, *, allow_fixture: bool) -> list[str]:
    """Why this gold set cannot be evaluated in the requested posture."""
    if allow_fixture:
        reasons: list[str] = []
        if gold.problems:
            reasons.append(
                f"{len(gold.problems)} structural problem(s); first: "
                f"{gold.problems[0]}")
        if not gold.valid:
            reasons.append("no structurally valid items")
        return reasons
    return gs.gate(gold)


def select_items(gold: gs.GoldSet, *, allow_fixture: bool) -> list[dict[str, Any]]:
    """The items an evaluation covers, or a refusal.

    Without allow_fixture only production-eligible items count and the loader's
    own gate decides; with it, every structurally valid record is evaluated
    and the result is labelled a fixture artifact.
    """
    reasons = gold_blockers(gold, allow_fixture=allow_fixture)
    if reasons:
        raise GoldEvaluationError("; ".join(reasons))
    items = list(gold.valid if allow_fixture else gold.eligible)
    blockers = executor_blockers(items)
    if blockers:
        raise GoldEvaluationError(
            f"{len(blockers)} executor item(s) cannot be run by the current "
            f"verifier layer: {blockers[0]}")
    return sorted(items, key=lambda r: str(r["id"]))


# --- clients ----------------------------------------------------------------


class FakeGoldClient:
    """Replays declared outputs. Never a measurement source.

    `outcomes` is {"served": [...], "outputs": {item_id: output}} where output
    is a string or an {answer, reasoning} object. Its identity says it is a
    fake, and aggregate refuses to label the result real.
    """

    kind = "fake"

    def __init__(self, outcomes: dict[str, Any], *, endpoint: str = "fake://outcomes"):
        if not isinstance(outcomes, dict) or not isinstance(outcomes.get("outputs"), dict):
            raise GoldEvaluationError("fake outcomes must carry an 'outputs' object")
        self.outcomes = outcomes
        self.endpoint = endpoint

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "produces_real_measurements": False,
            "served": list(self.outcomes.get("served") or []),
            "endpoint": self.endpoint,
            "note": "replays declared outputs; no model, no network",
        }

    def answer(self, record: dict[str, Any], decoding: dict[str, Any]) -> Any:
        outputs = self.outcomes["outputs"]
        if record["id"] not in outputs:
            raise KeyError(f"fake outcomes declare no output for {record['id']!r}")
        return outputs[record["id"]]

    @staticmethod
    def stamp(run_id: str, item_id: str, repetition: int, edge: str) -> str:
        return f"fake:{run_id}:{item_id}:{repetition}:{edge}"


class BridgeGoldClient:
    """The real client: `bridge_client.TeacherClient` over an OpenAI-compatible
    endpoint. Constructed only behind --real, and never in tests.

    Identity is read from the endpoint's /models listing so the receipt
    records what the endpoint SAYS it serves, checked against the registry by
    the caller. The chat call is the bridge's own `complete`.
    """

    kind = "real"

    def __init__(self, endpoint: str, model: str, *, api_key: str = "",
                 timeout_s: float = 600.0):
        import httpx                                       # noqa: PLC0415
        from bridge_client import TeacherClient             # noqa: PLC0415
        self._httpx = httpx
        self.endpoint = endpoint.rstrip("/")
        self.client = TeacherClient(
            base_url=self.endpoint, model=model, api_key=api_key,
            timeout_s=timeout_s, max_retries=1,
            sampling={k: v for k, v in DECODING.items()})

    def identity(self) -> dict[str, Any]:
        response = self._httpx.get(f"{self.endpoint}/models",
                                   headers=self.client._headers(), timeout=30.0)
        if response.status_code != 200:
            raise GoldEvaluationError(
                f"endpoint {self.endpoint} answered {response.status_code} on "
                "/models; its identity cannot be read")
        served = [str(m.get("id")) for m in (response.json().get("data") or [])
                  if isinstance(m, dict) and m.get("id")]
        return {
            "kind": self.kind,
            "produces_real_measurements": True,
            "served": served,
            "endpoint": self.endpoint,
        }

    def answer(self, record: dict[str, Any], decoding: dict[str, Any]) -> Any:
        async def call() -> dict[str, Any]:
            async with self._httpx.AsyncClient() as http:
                return await self.client.complete(
                    http, [{"role": "user", "content": str(record["prompt"])}])
        return asyncio.run(call())["content"]


def load_fake_outcomes(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GoldEvaluationError(f"cannot read fake outcomes {path}: {error}") from error
    if not isinstance(value, dict):
        raise GoldEvaluationError("fake outcomes must be a JSON object")
    return value


# --- receipts ---------------------------------------------------------------


def _shape_output(record: dict[str, Any], output: Any) -> Any:
    """What the verifier sees.

    A reasoning item wants {answer, reasoning}. A real endpoint returns text;
    if that text is a JSON object carrying `answer`, it is used as such,
    otherwise the text is the answer with no reasoning supplied -- which the
    verifier records honestly rather than this function guessing a split.
    """
    if isinstance(output, dict):
        return output
    text = str(output)
    declared = isinstance((record.get("expected") or {}).get("reasoning_check"), dict)
    if declared:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    return text


def _bounded_result(result: verifiers.VerificationResult) -> dict[str, Any]:
    """Booleans and digests only. The free-text detail can quote the expected
    answer or a required claim, so it is digested, not copied."""
    return {
        "passed": bool(result.passed),
        "counts_toward_correctness": bool(result.counts_toward_correctness),
        "false_positive": result.false_positive,
        "reasoning_checked": bool(result.reasoning_checked),
        "problem_count": len(result.problems),
        "detail_sha256": he.digest_payload(result.detail),
    }


def receipt_digest(receipt: dict[str, Any]) -> str:
    value = dict(receipt)
    value.pop("receipt_sha256", None)
    return he.digest_payload(value)


def receipt_set_digest(receipts: list[dict[str, Any]]) -> str:
    """Order-independent: the same receipts read in any order are the same
    evidence."""
    rows = sorted((str(r["item_id"]), int(r["repetition"]), receipt_digest(r))
                  for r in receipts)
    return he.digest_payload(rows)


def receipt_path(directory: Path, run_id: str, item_id: str, repetition: int) -> Path:
    safe = _SAFE.sub("_", item_id)
    return Path(directory) / f"{_SAFE.sub('_', run_id)}__{safe}__rep{repetition}.json"


def _write_atomic(path: Path, payload: dict[str, Any]) -> Path:
    """Write-then-rename, never over an existing file. Mirrors
    harness_eval.write_receipt: evidence is written whole or not at all."""
    path = Path(path)
    if path.exists():
        raise GoldEvaluationError(
            f"{path} already exists; existing evidence is never overwritten. "
            "Choose a new --output or --run-id.")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def validate_receipt(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise GoldEvaluationError("receipt must be an object")
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("kind") != RECEIPT_KIND:
        raise GoldEvaluationError("receipt is not a gold evaluation receipt")
    missing = [k for k in _RECEIPT_REQUIRED if k not in receipt]
    if missing:
        raise GoldEvaluationError(f"receipt missing fields: {sorted(missing)}")
    if receipt["training_authorized"] is not False:
        raise GoldEvaluationError("receipt.training_authorized must be false")
    if receipt["status"] not in STATUSES:
        raise GoldEvaluationError(f"receipt status {receipt['status']!r} is unknown")
    identity = receipt["model_identity"]
    if not isinstance(identity, dict):
        raise GoldEvaluationError("receipt.model_identity must be an object")
    for key in ("expected", "endpoint", "revision"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise GoldEvaluationError(f"receipt.model_identity.{key} is required")
    if not isinstance(identity.get("served"), list) or not identity["served"]:
        raise GoldEvaluationError("receipt.model_identity.served must be a non-empty list")
    client = receipt["client"]
    if not isinstance(client, dict) or \
            not isinstance(client.get("produces_real_measurements"), bool):
        raise GoldEvaluationError(
            "receipt.client.produces_real_measurements must be a boolean")
    result = receipt["verifier_result"]
    if receipt["status"] == STATUS_OK:
        if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
            raise GoldEvaluationError(
                f"receipt {receipt['item_id']}/{receipt['repetition']} is 'ok' "
                "without a boolean verifier verdict")
    elif result is not None:
        raise GoldEvaluationError(
            f"receipt {receipt['item_id']}/{receipt['repetition']} is "
            f"{receipt['status']!r} but carries a verifier result")
    if not isinstance(receipt["repetition"], int) or receipt["repetition"] < 1:
        raise GoldEvaluationError("receipt.repetition must be a positive integer")
    for forbidden in ("output", "expected", "prompt"):
        if forbidden in receipt:
            raise GoldEvaluationError(
                f"receipt carries {forbidden!r}; a receipt is provenance, not a "
                "copy of the item or the answer")
    return receipt


def load_receipts(directory: Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    if not directory.is_dir():
        raise GoldEvaluationError(f"missing receipts directory: {directory}")
    receipts: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise GoldEvaluationError(f"malformed receipt at {path}: {error}") from error
        if not isinstance(value, dict):
            raise GoldEvaluationError(f"receipt root must be an object: {path}")
        declared = value.pop("receipt_sha256", None)
        validate_receipt(value)
        if declared != receipt_digest(value):
            raise GoldEvaluationError(
                f"receipt at {path} does not match its own receipt_sha256; it "
                "was modified after it was written or never digested")
        receipts.append(value)
    if not receipts:
        raise GoldEvaluationError(f"no receipts found under {directory}")
    return receipts


# --- plan -------------------------------------------------------------------


def plan_evaluation(*, gold_dir: Path, model_registry: str, model_dir: Path = MODEL_DIR,
                    endpoint: str, repetitions: int,
                    experiment_path: Path = EXPERIMENT_PATH,
                    allow_fixture: bool = False) -> dict[str, Any]:
    """What would run. Writes nothing, constructs no client."""
    blockers: list[str] = []
    model: dict[str, Any] | None = None
    try:
        model = resolve_model(model_dir, model_registry)
    except GoldEvaluationError as error:
        blockers.append(str(error))
    if repetitions < MINIMUM_REPETITIONS:
        blockers.append(f"repetitions must be at least {MINIMUM_REPETITIONS} "
                        "for a stability claim")

    gold = load_gold(gold_dir)
    blockers.extend(gold_blockers(gold, allow_fixture=allow_fixture))
    items = list(gold.valid if allow_fixture else gold.eligible)
    blockers.extend(executor_blockers(items))

    by_split = {s: 0 for s in gs.SPLITS}
    by_verifier: dict[str, int] = {}
    for record in items:
        by_split[record["split"]] += 1
        by_verifier[record["verifier"]["type"]] = \
            by_verifier.get(record["verifier"]["type"], 0) + 1

    return {
        "phase": "plan",
        "gold_dir": str(gold_dir),
        "gold_set_digest": gold_set_digest(gold),
        "gold_summary": gold.summary(),
        "posture": "fixture" if allow_fixture else "production",
        "items": len(items),
        "by_split": by_split,
        "by_verifier": by_verifier,
        "repetitions": repetitions,
        "calls_planned": len(items) * repetitions,
        "model_registry": model_registry,
        "model_identity": {"expected": model["model_id"] if model else None,
                           "revision": model["revision"] if model else None},
        "endpoint": endpoint,
        "serving": model["serving"] if model else {"protocol": None, "tags": []},
        "decoding": dict(DECODING),
        "teacher_models_excluded": teacher_model_ids(experiment_path),
        "blockers": blockers,
        "executable": not blockers,
        "training_authorized": False,
    }


# --- run --------------------------------------------------------------------


def run_evaluation(*, gold_dir: Path, model_registry: str, model_dir: Path = MODEL_DIR,
                   client: Any, repetitions: int, output: Path, run_id: str = "run-1",
                   allow_fixture: bool = False,
                   experiment_path: Path = EXPERIMENT_PATH) -> dict[str, Any]:
    """Evaluate every item `repetitions` times; one receipt each.

    Every refusal below happens BEFORE the client is asked anything: a wrong
    posture, an unpinned model, a blocked gold set or an executor item the
    verifier layer cannot run must never cost a model call.
    """
    if repetitions < MINIMUM_REPETITIONS:
        raise GoldEvaluationError(
            f"repetitions must be at least {MINIMUM_REPETITIONS}; the separation "
            "gate needs repeated runs to bound the noise floor")
    model = resolve_model(model_dir, model_registry)
    gold = load_gold(gold_dir)
    items = select_items(gold, allow_fixture=allow_fixture)
    set_digest = gold_set_digest(gold)
    teachers = teacher_model_ids(experiment_path)

    output = Path(output)
    planned = [(record, rep) for rep in range(1, repetitions + 1) for record in items]
    existing = [receipt_path(output, run_id, str(r["id"]), rep)
                for r, rep in planned if receipt_path(output, run_id, str(r["id"]), rep).exists()]
    if existing:
        raise GoldEvaluationError(
            f"{len(existing)} receipt(s) already exist under {output} for run "
            f"{run_id!r} (first: {existing[0].name}); existing evidence is "
            "never overwritten")

    # Only now touch the client, and only for its identity.
    identity = client.identity()
    if not isinstance(identity, dict):
        raise GoldEvaluationError("client.identity() must return an object")
    check_served(model, identity)
    recorded_identity = {
        "expected": model["model_id"],
        "served": [str(s) for s in identity["served"]],
        "endpoint": str(identity.get("endpoint") or ""),
        "revision": model["revision"],
        "protocol": str(identity.get("protocol") or PROTOCOL_OPENAI),
    }
    check_not_teacher(recorded_identity, teachers, "the served model")
    if not recorded_identity["endpoint"]:
        raise GoldEvaluationError("client identity names no endpoint")
    is_fake = identity.get("produces_real_measurements") is not True
    if not is_fake and allow_fixture:
        raise GoldEvaluationError(
            "a real client cannot run in fixture posture; drop --allow-fixture")

    client_block = {
        "kind": str(identity.get("kind")),
        "produces_real_measurements": not is_fake,
    }
    counts = {s: 0 for s in STATUSES}
    written: list[str] = []

    for record, rep in planned:
        item_id = str(record["id"])
        if is_fake:
            started = FakeGoldClient.stamp(run_id, item_id, rep, "start")
        else:
            started = datetime.now(timezone.utc).isoformat()
        status, error, output_sha, chars, verdict = STATUS_OK, None, None, 0, None
        try:
            raw = client.answer(record, dict(DECODING))
            shaped = _shape_output(record, raw)
            output_sha = he.digest_payload(shaped)
            chars = len(json.dumps(shaped, ensure_ascii=False))
            verdict = _bounded_result(verifiers.verify(record, shaped))
        except Exception as exc:                          # noqa: BLE001
            # A client or verifier failure still owes a record. A missing
            # receipt reads as "never attempted" and aggregate refuses it.
            status, error, verdict = STATUS_ERROR, f"{type(exc).__name__}: {exc}", None
        if is_fake:
            ended = FakeGoldClient.stamp(run_id, item_id, rep, "end")
        else:
            ended = datetime.now(timezone.utc).isoformat()
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "kind": RECEIPT_KIND,
            "gold_set_digest": set_digest,
            "item_id": item_id,
            "split": record["split"],
            "score_role": record["score_role"],
            "verifier_type": record["verifier"]["type"],
            "model_registry": model_registry,
            "model_identity": recorded_identity,
            "client": client_block,
            "run_id": run_id,
            "repetition": rep,
            "repetitions_planned": repetitions,
            "decoding": dict(DECODING),
            "started_at": started,
            "ended_at": ended,
            "status": status,
            "error": error,
            "output_sha256": output_sha,
            "output_chars": chars,
            "verifier_result": verdict,
            "training_authorized": False,
        }
        validate_receipt(receipt)
        stamped = dict(receipt)
        stamped["receipt_sha256"] = receipt_digest(receipt)
        written.append(str(_write_atomic(
            receipt_path(output, run_id, item_id, rep), stamped)))
        counts[status] += 1

    return {
        "phase": "run",
        "gold_set_digest": set_digest,
        "model_identity": recorded_identity,
        "client": client_block,
        "run_id": run_id,
        "repetitions": repetitions,
        "items": len(items),
        "receipts_written": len(written),
        "status_counts": counts,
        "output": str(output),
        "fixture": is_fake,
        "training_authorized": False,
    }


# --- aggregate --------------------------------------------------------------


def measurement_digest(measurement: dict[str, Any]) -> str:
    value = dict(measurement)
    value.pop("measurement_sha256", None)
    return he.digest_payload(value)


def aggregate_measurement(*, gold_dir: Path, receipts: Path, arm: str,
                          allow_fixture: bool = False,
                          experiment_path: Path = EXPERIMENT_PATH) -> dict[str, Any]:
    """The measurement `eval_harness.separation_gate` reads, from receipts only."""
    if not isinstance(arm, str) or not arm.strip():
        raise GoldEvaluationError("an arm label is required")
    gold = load_gold(gold_dir)
    items = select_items(gold, allow_fixture=allow_fixture)
    set_digest = gold_set_digest(gold)
    rows = load_receipts(receipts)

    # --- one gold set, one identity, one run, one kind of client
    foreign = [r for r in rows if r["gold_set_digest"] != set_digest]
    if foreign:
        raise GoldEvaluationError(
            f"{len(foreign)} receipt(s) name a different gold_set_digest "
            f"({foreign[0]['gold_set_digest'][:12]}... vs {set_digest[:12]}...); "
            "the receipts did not answer these items")
    for field in ("model_identity", "model_registry", "run_id",
                  "repetitions_planned", "decoding"):
        values = {json.dumps(r[field], sort_keys=True) for r in rows}
        if len(values) != 1:
            raise GoldEvaluationError(
                f"receipts mix {len(values)} values of {field!r}; one measurement "
                "is one model at one endpoint under one run")
    first = rows[0]
    identity = dict(first["model_identity"])
    fixture_rows = [r for r in rows if not r["client"]["produces_real_measurements"]]
    if fixture_rows and len(fixture_rows) != len(rows):
        raise GoldEvaluationError(
            "receipts mix fixture and real clients; a partly replayed "
            "measurement is not interpretable")
    is_fixture = bool(fixture_rows)
    if is_fixture and not allow_fixture:
        raise GoldEvaluationError(
            f"{len(fixture_rows)} receipt(s) came from a fixture client. A "
            "replayed outcome is not a measurement; pass --allow-fixture to "
            "aggregate it as an explicitly labelled fixture artifact")
    if not is_fixture and allow_fixture:
        raise GoldEvaluationError(
            "--allow-fixture was passed but the receipts are real; a real "
            "measurement is not labelled as a fixture")
    check_not_teacher(identity, teacher_model_ids(experiment_path), f"arm {arm!r}")

    # --- coverage: every (item, repetition) exactly once
    repetitions = int(first["repetitions_planned"])
    if repetitions < MINIMUM_REPETITIONS:
        raise GoldEvaluationError("receipts plan fewer than two repetitions")
    wanted = {str(r["id"]) for r in items}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for receipt in rows:
        key = (str(receipt["item_id"]), int(receipt["repetition"]))
        if key[0] not in wanted:
            raise GoldEvaluationError(
                f"receipt names item {key[0]!r}, which is not in the evaluated set")
        if key[1] > repetitions:
            raise GoldEvaluationError(
                f"receipt {key[0]}/{key[1]} exceeds the planned {repetitions} repetitions")
        if key in indexed:
            raise GoldEvaluationError(
                f"duplicate receipt for item {key[0]!r} repetition {key[1]}; the "
                "same execution counted twice is not two measurements")
        indexed[key] = receipt
    missing = [f"{i}/{rep}" for i in sorted(wanted) for rep in range(1, repetitions + 1)
               if (i, rep) not in indexed]
    if missing:
        raise GoldEvaluationError(
            f"{len(missing)} item/repetition pair(s) have no receipt (first: "
            f"{missing[0]}); an incomplete run is not a measurement")

    # --- score. per_item is the FIRST repetition (a real observed run);
    # every repetition's pass rate is listed under runs for the stability
    # check, and per_item_by_repetition keeps the paired detail.
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in gs.SPLITS}
    for record in items:
        by_split[record["split"]].append(record)

    def passed(receipt: dict[str, Any]) -> bool:
        result = receipt.get("verifier_result")
        return receipt["status"] == STATUS_OK and bool(result and result["passed"])

    splits: dict[str, Any] = {}
    false_positive_ids: list[str] = []
    quarantined = 0
    for split in gs.SPLITS:
        counted: list[str] = []
        for record in by_split[split]:
            item_id = str(record["id"])
            primary = indexed[(item_id, 1)]
            result = primary.get("verifier_result") or {}
            if record["score_role"] != "correctness" or \
                    record["verifier"]["type"] in verifiers.QUARANTINED:
                quarantined += 1
                continue
            counted.append(item_id)
            if result.get("false_positive") is True:
                false_positive_ids.append(item_id)
        if not counted:
            # A missing split is not a zero; the gate refuses it downstream and
            # this artifact must not pretend otherwise.
            raise GoldEvaluationError(
                f"split {split!r} has no correctness items; a measurement "
                "cannot pair on an empty split")
        per_rep = {
            rep: {i: passed(indexed[(i, rep)]) for i in counted}
            for rep in range(1, repetitions + 1)
        }
        splits[split] = {
            "per_item": dict(per_rep[1]),
            "per_item_by_repetition": {str(k): v for k, v in per_rep.items()},
            "items": len(counted),
            "runs": [
                {"run_id": first["run_id"], "repetition": rep,
                 "pass_rate": sum(1 for v in per_rep[rep].values() if v) / len(counted)}
                for rep in range(1, repetitions + 1)
            ],
        }

    measurement = {
        "schema_version": MEASUREMENT_SCHEMA,
        "kind": MEASUREMENT_KIND,
        "arm": arm,
        "model_registry": first["model_registry"],
        "model_identity": identity,
        "gold_dir": str(gold_dir),
        "gold_set_digest": set_digest,
        "items": len(items),
        "repetitions": repetitions,
        "run_id": first["run_id"],
        "fixture": is_fixture,
        "decoding": dict(first["decoding"]),
        "splits": splits,
        "quarantined_items": quarantined,
        "false_positive_ids": sorted(false_positive_ids),
        "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES},
        "receipts": len(rows),
        "receipt_set_digest": receipt_set_digest(rows),
        "measured": True,
        "training_authorized": False,
    }
    problems = eh.validate_measurement(measurement, label=arm)
    if problems:
        raise GoldEvaluationError(
            "the derived measurement would be refused by the separation gate: "
            + "; ".join(problems))
    measurement["measurement_sha256"] = measurement_digest(measurement)
    return measurement


def write_measurement(path: Path, measurement: dict[str, Any]) -> Path:
    return _write_atomic(Path(path), measurement)


# --- CLI --------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser, *, model: bool) -> None:
    parser.add_argument("--gold-dir", type=Path, default=gs.GOLD_DIR)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT_PATH,
                        help="experiment naming the teacher model to exclude")
    if model:
        parser.add_argument("--model-registry", required=True,
                            help="student registry name under --model-dir")
        parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
        parser.add_argument("--endpoint", required=True,
                            help="OpenAI-compatible base URL (…/v1)")
        parser.add_argument("--repetitions", type=int, default=MINIMUM_REPETITIONS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("plan", help="what would run (writes nothing)")
    _common(p, model=True)
    p.add_argument("--allow-fixture", action="store_true",
                   help="plan over a synthetic fixture set instead of production gold")

    r = commands.add_parser("run", help="write one receipt per item and repetition")
    _common(r, model=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--run-id", default="run-1")
    r.add_argument("--real", action="store_true",
                   help="contact --endpoint; requires --client and an approved "
                        "production gold set")
    r.add_argument("--client", choices=list(CLIENTS),
                   help="which wire a --real run speaks: openai "
                        "(/v1/chat/completions via bridge_client) or ollama "
                        "(/api/chat with think:false). Required with --real; "
                        "never inferred from the endpoint")
    r.add_argument("--fake-outcomes", type=Path,
                   help="deterministic fake client replaying this outcomes file")
    r.add_argument("--allow-fixture", action="store_true",
                   help="evaluate a synthetic fixture set (fake client only)")
    r.add_argument("--api-key-env", default="",
                   help="environment variable holding the endpoint key, if any")

    a = commands.add_parser("aggregate", help="derive the measurement from receipts")
    _common(a, model=False)
    a.add_argument("--receipts", type=Path, required=True)
    a.add_argument("--arm", required=True, help="label, e.g. student_9b")
    a.add_argument("--output", type=Path)
    a.add_argument("--allow-fixture", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan_evaluation(
                gold_dir=args.gold_dir, model_registry=args.model_registry,
                model_dir=args.model_dir, endpoint=args.endpoint,
                repetitions=args.repetitions, experiment_path=args.experiment,
                allow_fixture=args.allow_fixture)
        elif args.command == "run":
            if args.real and args.fake_outcomes:
                raise GoldEvaluationError("--real and --fake-outcomes are exclusive")
            if args.real and args.allow_fixture:
                raise GoldEvaluationError(
                    "--real cannot be combined with --allow-fixture")
            if not args.real and not args.fake_outcomes:
                raise GoldEvaluationError(
                    "no client selected. Pass --fake-outcomes <file> for a "
                    "deterministic fake run, or --real to contact --endpoint. "
                    "The default posture of this tool is plan.")
            # The gold gate is decided before any client exists, so a blocked
            # set never costs a connection.
            gold = load_gold(args.gold_dir)
            select_items(gold, allow_fixture=args.allow_fixture)
            model = resolve_model(args.model_dir, args.model_registry)
            if args.real:
                if not args.client:
                    raise GoldEvaluationError(
                        "--real needs --client openai|ollama; the wire is never "
                        "inferred from the endpoint")
                check_client_protocol(model, args.client)
                if args.client == CLIENT_OLLAMA:
                    from ollama_chat_client import OllamaChatClient  # noqa: PLC0415
                    client: Any = OllamaChatClient(
                        args.endpoint, model["serving"]["tags"][0])
                else:
                    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
                    client = BridgeGoldClient(args.endpoint, model["model_id"], api_key=key)
            else:
                client = FakeGoldClient(load_fake_outcomes(args.fake_outcomes),
                                        endpoint=args.endpoint)
            result = run_evaluation(
                gold_dir=args.gold_dir, model_registry=args.model_registry,
                model_dir=args.model_dir, client=client,
                repetitions=args.repetitions, output=args.output,
                run_id=args.run_id, allow_fixture=args.allow_fixture,
                experiment_path=args.experiment)
        else:
            result = aggregate_measurement(
                gold_dir=args.gold_dir, receipts=args.receipts, arm=args.arm,
                allow_fixture=args.allow_fixture, experiment_path=args.experiment)
            if args.output:
                write_measurement(args.output, result)
    except GoldEvaluationError as error:
        die(str(error))
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
