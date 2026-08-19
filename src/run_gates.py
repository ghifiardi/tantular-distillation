"""Run the training eval gates, before and after, and compare them.

    # evaluate the base STUDENT (never the teacher: see verify_served_model)
    ./.venv/bin/python src/run_gates.py run --stage before \
        --host student-serve --teacher office-student-9b \
        --expect-model Qwen/Qwen3.5-9B-Instruct --out data/gates/before.json

    # evaluate the trained adapter
    ./.venv/bin/python src/run_gates.py run --stage after \
        --host student-serve --teacher office-student-9b \
        --expect-model Qwen/Qwen3.5-9B-Instruct \
        --adapter adapters/tantular-office-9b-v2 --out data/gates/after.json

STAGE SEMANTICS, decided 2026-08-19.

  before   Everything must be present and runnable: endpoint, model identity,
           generated output, every declared gate. A score under threshold is
           recorded as BASELINE_BELOW_TARGET and the run continues — the base
           student was never trained on Indonesian office work, so requiring it
           to clear the bar meant for a PROMOTED adapter would mean the run can
           never start, and the obvious escape is to lower the bar. Exit 0.

  after    The adapter must reach the absolute threshold AND not regress against
           the baseline, with every gate executed. Only then is it promotable.
           Exit 1 on a missed threshold.

  both     Missing endpoint, wrong model, missing scorer or missing output is
           exit 2. A tolerant threshold must not become a tolerant runner.

"measured" and "passed" are separate words in the report and never merge: a
baseline below target is not a pass, and no threshold moves to make it one.

    ./.venv/bin/python src/run_gates.py compare \
        --before data/gates/before.json --after data/gates/after.json

`train/qlora_9b.yaml` says of its gates: "Run before and after, compare, and do
not promote an adapter that regresses either." This executes that instruction.

FAILS CLOSED. A missing eval set, a missing scorer, a missing gate runner, or a
gate that produces no output is a FAILURE, never a skip. A gate that silently
does not run is worse than one that fails: the run completes, the report looks
clean, and nothing was checked.

MODEL-DEPENDENT vs MODEL-INDEPENDENT — measured, and it matters:

  indonesian_voice      MODEL-DEPENDENT. Generates 40 answers from the model
                        under test and scores them. Before/after differ, so it
                        can detect a regression.

  office_json_contract  MODEL-INDEPENDENT as configured. It runs the add-in's
                        own JS suite (`node --test tests/*.test.mjs`, 382 tests)
                        with globalThis.fetch MOCKED. It never calls a model, so
                        it returns the SAME result before and after and CANNOT
                        detect an adapter regression.

That second finding is reported in every comparison rather than left implicit.
The config's stated intent — "edit-contract shape must still parse" — is about
whether the MODEL still emits parseable edit payloads, which would require
feeding model output through the add-in's parser. The gate as configured checks
that the add-in's own code still works, which is a build-health check. Both are
worth having; only one of them gates the adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

# Which gates can see the model at all. Anything not listed is treated as
# model-independent, which is the conservative reading.
MODEL_DEPENDENT = {"indonesian_voice", "edit_contract_output"}


def digest_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def digest_tree(path: Path) -> str | None:
    """Stable digest of a directory: sorted (relpath, filehash) pairs."""
    if not path.is_dir():
        return None
    parts = []
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        parts.append(f"{f.relative_to(path)}:{digest_file(f)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def fail(msg: str) -> None:
    print(f"\nGATE RUN ABORTED: {msg}", file=sys.stderr)
    sys.exit(2)


def verify_served_model(expect: str, host: str, teacher: str) -> dict:
    """Refuse to gate a model that is not the one the config names.

    The gates measure whatever endpoint they are pointed at. Nothing else in the
    pipeline notices if that endpoint is serving the TEACHER rather than the
    student: the run completes, the numbers look plausible, and the before/after
    comparison silently becomes teacher-vs-adapter, which measures nothing.

    So identity is checked against the config's base_model before any
    model-dependent gate runs.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from config import base_url, resolve
    try:
        resolved = resolve(teacher, host)
    except SystemExit as e:
        fail(f"cannot resolve host '{host}' / model '{teacher}': {e}\n"
             "The student is configs/teachers/office-student-9b.yaml served on "
             "configs/hosts/student-serve.yaml; the teachers are not it.")
    url = resolved.get("HOST_BASE_URL") or base_url(resolved)
    try:
        import httpx
        served = [m.get("id") for m in
                  httpx.get(f"{url}/models", timeout=30).json().get("data", [])]
    except Exception as e:
        fail(f"cannot reach {url} to verify model identity: {e}")

    if not any(expect == s or expect.split("/")[-1].lower() in str(s).lower()
               for s in served):
        fail(f"the endpoint is NOT serving the model the config names.\n"
             f"  config base_model : {expect}\n"
             f"  endpoint serves   : {served}\n"
             "Gating a different model would make before/after meaningless — most "
             "likely this is pointed at the teacher rather than the student.")
    print(f"  model identity OK: {expect} served at {url}")
    return {"expected": expect, "served": served, "endpoint": url,
            "quantization": resolved.get("HOST_QUANTIZATION")}


def requested_model_id(args) -> str:
    """The id the gates ASK the endpoint for. At `after` that is the adapter's
    own id — never the base, which would return base answers."""
    if args.stage == "after":
        return args.adapter_model_id
    return args.expect_model or args.teacher or ""


ADAPTER_REQUIRED_FILES = ("adapter_config.json",)
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def configured_base_aliases(base_id: str, teacher: str | None) -> set[str]:
    """Return every configured id that can address the same base weights.

    Fixture-mode runs may omit --teacher/--host, but they still write a report
    claiming that an adapter id is distinct from the base. Derive aliases from
    the model configs rather than weakening that assertion when no endpoint is
    involved.
    """
    aliases = {base_id, Path(base_id).name, teacher or ""}
    for path in (ROOT / "configs" / "teachers").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        repos = set((payload.get("repos") or {}).values())
        if base_id not in repos and payload.get("served_model_name") != base_id:
            continue
        aliases.update({
            path.stem,
            payload.get("name", ""),
            payload.get("served_model_name", ""),
        })
    return {alias for alias in aliases if alias}


def list_served(host: str, teacher: str) -> tuple[list, str]:
    sys.path.insert(0, str(ROOT / "src"))
    from config import base_url, resolve
    try:
        resolved = resolve(teacher, host)
    except SystemExit as e:
        fail(f"cannot resolve host '{host}' / model '{teacher}': {e}")
    url = resolved.get("HOST_BASE_URL") or base_url(resolved)
    try:
        import httpx
        served = [m.get("id") for m in
                  httpx.get(f"{url}/models", timeout=30).json().get("data", [])]
    except Exception as e:
        fail(f"cannot reach {url} to list served models: {e}")
    return served, url


def verify_adapter_served(adapter: Path, adapter_id: str, base_id: str,
                          host: str, teacher: str, live: bool) -> dict:
    """Refuse to call an 'after' run an adapter evaluation unless it is one.

    Until 2026-08-19 --adapter was only existence-checked and hashed: the gates
    then generated from the SAME model id as the baseline. Every after run would
    have re-measured the base model, recorded the adapter's digest beside it, and
    reported a complete before/after comparison of a model against itself. The
    digest made it look verified, which is worse than not recording one.

    An adapter evaluation now has to prove four things:

      it is an adapter    the directory holds adapter_config.json and weights,
                          not merely some directory that hashes
      it has its own id   the id requested from the endpoint differs from EVERY
                          id under which the base is served; asking for either
                          the repo id or its short alias returns BASE answers
                          even when a LoRA is loaded in the same process
      it is loaded        /v1/models lists that id
      it was used         each model-dependent gate records the id it requested

    The fourth lives in the gate results, not here.
    """
    detail = {"path": str(adapter), "model_id": adapter_id, "base_id": base_id}

    missing = [f for f in ADAPTER_REQUIRED_FILES if not (adapter / f).is_file()]
    if missing:
        fail(f"{adapter} is not a LoRA adapter directory: missing "
             f"{', '.join(missing)}.\nA directory that hashes is not an adapter; "
             "the digest would document nothing.")
    if not any((adapter / f).is_file() for f in ADAPTER_WEIGHT_FILES):
        fail(f"{adapter} has an adapter_config.json but no weights "
             f"({' or '.join(ADAPTER_WEIGHT_FILES)}). Nothing would be loaded.")

    if not adapter_id:
        fail("--stage after needs --adapter-model-id: the id the endpoint "
             "registered the adapter under.\nWithout it the gates request the "
             "base model id and measure the base model, with the adapter's "
             "digest recorded beside the result.")
    # serve_student.sh deliberately registers the base under both its repo id
    # and the short config key (`office-student-9b`). Reject every known alias.
    # Checking only config["base_model"] is insufficient: /v1/models already
    # lists the short alias for the BASE, so an adapter using that id would look
    # "loaded" while requests continued to return base answers.
    base_aliases = configured_base_aliases(base_id, teacher)
    if adapter_id in base_aliases:
        fail(f"--adapter-model-id is a BASE model id/alias ({adapter_id!r}).\n"
             f"  known base ids: {sorted(x for x in base_aliases if x)}\n"
             "vLLM serves a LoRA under its own id alongside the base; asking for "
             "any base alias returns base answers from the same process.")

    if live:
        served, url = list_served(host, teacher)
        detail["served"] = served
        detail["endpoint"] = url
        if adapter_id not in served:
            fail(f"the endpoint does not serve {adapter_id!r}.\n"
                 f"  serves: {served}\n"
                 "Serve the base with --enable-lora and register the adapter "
                 "with --lora-modules <id>=<path>, or the after run measures the "
                 "base model.")
        print(f"  adapter identity OK: {adapter_id} served at {url}")
    else:
        detail["served"] = None
        detail["fixtures"] = True
    return detail


# --- gate implementations ---------------------------------------------------

def gate_indonesian_voice(spec: dict, stage: str, args, out_dir: Path) -> dict:
    items = ROOT / spec["source"]
    scorer = ROOT / spec.get("scorer", "src/score_voice.py")
    if not items.is_file():
        fail(f"indonesian_voice source missing: {items}")
    if not scorer.is_file():
        fail(f"indonesian_voice scorer missing: {scorer}")

    traces = out_dir / f"voice.{stage}.traces.jsonl"
    if args.traces:                       # pre-generated, e.g. for tests
        traces = Path(args.traces)
        if not traces.is_file():
            fail(f"--traces given but missing: {traces}")
    else:
        cmd = [PY, str(ROOT / "src" / "generate_normalized.py"),
               "--teacher", args.teacher, "--host", args.host,
               "--model-id", requested_model_id(args),
               "--prompts", str(items), "--out", str(traces), "--resume"]
        print(f"  generating {stage} answers from {requested_model_id(args)} "
              f"-> {traces.name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"generation failed for indonesian_voice:\n{proc.stderr[-500:]}")
    if not traces.is_file():
        fail("indonesian_voice produced no traces — failing closed")

    report = out_dir / f"voice.{stage}.score.json"
    proc = subprocess.run(
        [PY, str(scorer), "--traces", str(traces), "--items", str(items),
         "--threshold", str(spec["min_pass_rate"]), "--json-out", str(report)],
        capture_output=True, text=True)
    if not report.is_file():
        fail(f"indonesian_voice scorer produced no report:\n{proc.stderr[-500:]}")
    scored = json.loads(report.read_text())
    return {
        "name": "indonesian_voice",
        "model_dependent": True,
        "rate": scored["rate"], "threshold": scored["threshold"],
        "passed": scored["verdict"] == "PASS",
        "items": scored["items"],
        "per_dimension_failures": scored.get("per_dimension_failures", {}),
        "hashes": {"eval_set": digest_file(items), "scorer": digest_file(scorer)},
        "artifacts": {"traces": str(traces), "report": str(report)},
    }


def gate_office_json_contract(spec: dict, stage: str, args, out_dir: Path) -> dict:
    suite = (ROOT / spec["source"]).resolve()
    if not suite.is_dir():
        fail(f"office_json_contract source missing: {suite}")
    project = suite.parent
    tests = sorted(suite.glob("*.test.mjs"))
    if not tests:
        fail(f"office_json_contract found no *.test.mjs under {suite}")

    proc = subprocess.run(["node", "--test", *[str(t) for t in tests]],
                          capture_output=True, text=True, cwd=project)
    out = proc.stdout + proc.stderr
    if "# tests " not in out:
        fail("office_json_contract runner produced no test summary — "
             "is node installed? failing closed rather than skipping")
    def field(key: str) -> int:
        for line in out.splitlines():
            if line.startswith(f"# {key} "):
                return int(line.split()[-1])
        fail(f"office_json_contract summary missing '{key}'")
    total, passed = field("tests"), field("pass")
    rate = passed / total if total else 0.0
    return {
        "name": "office_json_contract",
        "model_dependent": False,
        "_model_independent_note": (
            "Runs the add-in's own JS suite with globalThis.fetch mocked. It never "
            "calls a model, so before and after are identical by construction and "
            "this gate CANNOT detect an adapter regression."),
        "rate": rate, "threshold": spec["min_pass_rate"],
        "passed": rate >= spec["min_pass_rate"],
        "items": total,
        "hashes": {"suite": digest_tree(suite)},
    }


def gate_edit_contract_output(spec: dict, stage: str, args, out_dir: Path) -> dict:
    """Model-dependent counterpart to office_json_contract.

    Generates edit answers from the model under test and pushes them through the
    add-in's OWN parser (parseEditContract / resolveEdits / applyEditsToText) via
    scripts/check_edit_contract.mjs. The rate is contract_ok — parsed AND located
    AND applied — because valid JSON whose `find` appears nowhere in the document
    parses cleanly and is useless.
    """
    items = ROOT / spec["source"]
    checker = ROOT / spec.get("checker", "scripts/check_edit_contract.mjs")
    addin_src = (ROOT / spec["addin_src"]).resolve()
    if not items.is_file():
        fail(f"edit_contract_output source missing: {items}")
    if not checker.is_file():
        fail(f"edit_contract_output checker missing: {checker}")
    if not (addin_src / "chat" / "editContract.js").is_file():
        fail(f"add-in parser missing under {addin_src} — cannot check the real contract")

    cases_in = [json.loads(l) for l in
                items.read_text(encoding="utf-8").splitlines() if l.strip()]
    traces_path = out_dir / f"edit.{stage}.traces.jsonl"
    if args.edit_traces:
        traces_path = Path(args.edit_traces)
        if not traces_path.is_file():
            fail(f"--edit-traces given but missing: {traces_path}")
    else:
        cmd = [PY, str(ROOT / "src" / "generate_normalized.py"),
               "--teacher", args.teacher, "--host", args.host,
               "--model-id", requested_model_id(args),
               "--prompts", str(items), "--out", str(traces_path), "--resume"]
        print(f"  generating {stage} edit answers -> {traces_path.name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"generation failed for edit_contract_output:\n{proc.stderr[-500:]}")
    if not traces_path.is_file():
        fail("edit_contract_output produced no traces — failing closed")

    completions = {}
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completions[row.get("family") or row.get("id")] = row.get("completion", "")
    missing = [c["id"] for c in cases_in if c["id"] not in completions]
    if missing:
        fail(f"{len(missing)} edit item(s) have no model output: {missing[:3]} — "
             "a missing output is a failure, not a skip")

    cases = [{"id": c["id"], "document": c["document"],
              "completion": completions[c["id"]]} for c in cases_in]
    cases_file = out_dir / f"edit.{stage}.cases.json"
    cases_file.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(["node", str(checker), str(cases_file), str(addin_src)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        fail(f"edit contract checker failed:\n{proc.stderr[-500:]}")
    scored = json.loads(proc.stdout)
    report = out_dir / f"edit.{stage}.score.json"
    report.write_text(json.dumps(scored, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return {
        "name": "edit_contract_output",
        "model_dependent": True,
        "rate": scored["rate"], "threshold": spec["min_pass_rate"],
        "passed": scored["rate"] >= spec["min_pass_rate"],
        "items": scored["items"],
        "breakdown": {"parse_ok": scored["parse_ok"], "fields_ok": scored["fields_ok"],
                      "contract_ok": scored["contract_ok"]},
        "hashes": {"eval_set": digest_file(items), "checker": digest_file(checker),
                   "addin_parser": digest_file(addin_src / "chat" / "editContract.js")},
        "artifacts": {"traces": str(traces_path), "report": str(report)},
    }


GATES = {"indonesian_voice": gate_indonesian_voice,
         "office_json_contract": gate_office_json_contract,
         "edit_contract_output": gate_edit_contract_output}


# --- commands ---------------------------------------------------------------

def cmd_run(args) -> None:
    config = Path(args.config)
    if not config.is_file():
        fail(f"config missing: {config}")
    cfg = yaml.safe_load(config.read_text())
    specs = cfg.get("eval_gates") or []
    if not specs:
        fail("config declares no eval_gates")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = Path(args.adapter) if args.adapter else None
    if args.stage == "after" and adapter is None:
        fail("--stage after requires --adapter: an 'after' run with no adapter "
             "would re-measure the base model and report it as the trained one")
    if adapter is not None and not adapter.exists():
        fail(f"--adapter given but missing: {adapter}")

    model_identity = None
    adapter_identity = None
    needs_model = any(g.get("name") in MODEL_DEPENDENT for g in specs)
    using_fixtures = bool(args.traces or args.edit_traces)
    if args.stage == "after" and needs_model:
        # Checked even with fixtures: a fixture run still asserts, in its report,
        # that an adapter was evaluated. The directory and id checks hold in both
        # modes; only the /v1/models lookup needs a live endpoint.
        # The base id comes from the CONFIG, not from --expect-model: the check
        # has to hold in fixture runs too, where --expect-model is absent, and
        # the config is the thing that actually names the base model.
        base_id = cfg.get("base_model") or args.expect_model or args.teacher or ""
        adapter_identity = verify_adapter_served(
            adapter, args.adapter_model_id, base_id,
            args.host, args.teacher, live=not using_fixtures)
    if needs_model and not using_fixtures:
        if not args.expect_model:
            fail("--expect-model is required when model-dependent gates will call "
                 "a live endpoint. Pass the config's base_model so the gates "
                 "cannot silently measure the teacher instead of the student.")
        model_identity = verify_served_model(args.expect_model, args.host, args.teacher)

    print(f"=== GATE RUN [{args.stage}] ===")
    print(f"  config  {config}  {digest_file(config)[:16]}…")
    print(f"  model   {args.teacher} @ {args.host}")
    print(f"  adapter {adapter or '(base model, no adapter)'}")

    results = []
    for spec in specs:
        name = spec.get("name")
        if name not in GATES:
            fail(f"config declares gate '{name}' with no runner — failing closed "
                 "rather than skipping a gate someone believes is running")
        print(f"\n--- {name} ---")
        result = GATES[name](spec, args.stage, args, out_dir)
        # `passed` keeps its meaning at BOTH stages: rate >= threshold, nothing
        # else. What changes with the stage is the consequence, not the label —
        # a baseline below target is still a baseline below target.
        # Which model id produced these answers. Without this the report cannot
        # distinguish an adapter evaluation from a base evaluation carrying an
        # adapter's digest.
        if result["model_dependent"]:
            result["generated_by_model_id"] = requested_model_id(args)
            result["from_fixtures"] = bool(args.traces or args.edit_traces)
        result["status"] = (
            ("MET_TARGET" if result["passed"] else "BASELINE_BELOW_TARGET")
            if args.stage == "before" else
            ("PASS" if result["passed"] else "FAIL"))
        dep = "model-dependent" if result["model_dependent"] else "MODEL-INDEPENDENT"
        print(f"  {result['status']:<22} {result['rate']:.4f} vs threshold "
              f"{result['threshold']}  ({result['items']} items, {dep})")
        results.append(result)

    report = {
        "stage": args.stage,
        "config": {"path": str(config), "sha256": digest_file(config)},
        "model": {"teacher": args.teacher, "host": args.host,
                  "expected": args.expect_model, "identity": model_identity},
        "adapter": {"path": str(adapter) if adapter else None,
                    "sha256": digest_tree(adapter) if adapter else None,
                    "model_id": args.adapter_model_id,
                    "identity": adapter_identity,
                    # A digest alone never proved an adapter was evaluated; this
                    # says whether the endpoint was asked for the adapter's id.
                    "evaluated": bool(adapter_identity)},
        "gates": results,
        # Reaching this line means every gate EXECUTED: each implementation
        # fail()s with exit 2 on a missing endpoint, scorer, source or output,
        # so a rate exists for every gate or the run never got here.
        "measured": True,
        "all_passed": all(r["passed"] for r in results),
        "below_target": [r["name"] for r in results if not r["passed"]],
    }

    # Stage semantics, decided 2026-08-19.
    #
    # `before` measures a base model that was never trained on Indonesian office
    # work. Requiring it to clear 0.95 — the bar for a PROMOTED adapter — would
    # mean the run can never start, and the obvious escape is to lower the
    # threshold, which destroys the only number that matters after training. So
    # a low baseline is recorded, not treated as a failure.
    #
    # What is NOT weakened: "measured" and "passed" stay separate. A baseline
    # below target is BASELINE_BELOW_TARGET, never a pass, and the report says
    # so in the file as well as on the terminal. Infrastructure and identity
    # failures still exit 2 at both stages, from inside the gate runners.
    if args.stage == "before":
        report["verdict"] = ("BASELINE_MEASURED_BELOW_TARGET"
                             if report["below_target"] else "BASELINE_MEETS_TARGET")
        report["promotable"] = False      # a baseline is never promotable
        exit_code = 0
    else:
        report["verdict"] = "PASS" if report["all_passed"] else "FAIL"
        # Absolute threshold is necessary but not sufficient: `compare` also has
        # to find no regression before anything is promotable.
        report["promotable"] = None
        exit_code = 0 if report["all_passed"] else 1

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    if args.stage == "before":
        if report["below_target"]:
            print(f"\nBASELINE MEASURED, BELOW TARGET: "
                  f"{', '.join(report['below_target'])}")
            print("This is NOT a pass. It is a recorded starting point, and the "
                  "run may continue.")
            print("The same thresholds apply unchanged at --stage after.")
        else:
            print("\nBASELINE MEASURED, already at or above every threshold.")
    else:
        print("ALL GATES PASS" if report["all_passed"] else "ONE OR MORE GATES FAILED")
    sys.exit(exit_code)


def cmd_compare(args) -> None:
    before, after = Path(args.before), Path(args.after)
    for p in (before, after):
        if not p.is_file():
            fail(f"missing gate report: {p}")
    b, a = json.loads(before.read_text()), json.loads(after.read_text())

    if b.get("stage") != "before" or a.get("stage") != "after":
        fail(f"stages are wrong way round: before={b.get('stage')!r}, "
             f"after={a.get('stage')!r}")

    if b["config"]["sha256"] != a["config"]["sha256"]:
        fail("before and after ran against DIFFERENT configs — the comparison "
             "would attribute a config change to the adapter")
    if a["adapter"]["sha256"] is None:
        fail("the 'after' report has no adapter — nothing was trained to compare")

    # A digest is not evidence of an evaluation. These three refusals exist
    # because the after run used to hash the adapter and then generate from the
    # base model id, producing a complete-looking comparison of a model with
    # itself.
    if not a["adapter"].get("evaluated"):
        fail("the 'after' report records an adapter digest but no adapter "
             "identity.\nIt was produced before adapter serving was verified, or "
             "by a runner that never asked the endpoint for the adapter.")
    base_id = (a["adapter"].get("identity") or {}).get("base_id") \
        or a["model"].get("expected") or a["model"].get("teacher")
    adapter_id = a["adapter"].get("model_id")
    if not adapter_id or adapter_id == base_id:
        fail(f"the 'after' run requested model id {adapter_id!r}, which is the "
             f"base ({base_id!r}).\nThose answers came from the base model.")
    for g in a["gates"]:
        if not g["model_dependent"]:
            continue
        got = g.get("generated_by_model_id")
        if got != adapter_id:
            fail(f"gate '{g['name']}' generated from {got!r}, not from the "
                 f"adapter id {adapter_id!r}.\nThe comparison would attribute "
                 "base-model answers to the adapter.")
    for g in b["gates"]:
        if g["model_dependent"] and g.get("generated_by_model_id") == adapter_id:
            fail(f"the BASELINE gate '{g['name']}' also generated from the "
                 f"adapter id {adapter_id!r} — before and after are the same "
                 "model.")

    print("=== BEFORE / AFTER ===")
    print(f"  config  {b['config']['sha256'][:16]}…  (identical in both)")
    print(f"  adapter {a['adapter']['path']}  {a['adapter']['sha256'][:16]}…\n")
    print(f"  {'gate':<24}{'before':>9}{'after':>9}{'delta':>9}   verdict")

    bg = {g["name"]: g for g in b["gates"]}
    regressed, uninformative = [], []
    for g in a["gates"]:
        prev = bg.get(g["name"])
        if prev is None:
            fail(f"gate '{g['name']}' present after but not before")
        delta = g["rate"] - prev["rate"]
        verdict = "REGRESSED" if delta < 0 else ("same" if delta == 0 else "improved")
        if delta < 0:
            regressed.append(g["name"])
        if not g["model_dependent"]:
            uninformative.append(g["name"])
        print(f"  {g['name']:<24}{prev['rate']:>9.4f}{g['rate']:>9.4f}"
              f"{delta:>+9.4f}   {verdict}"
              + ("  [model-independent]" if not g["model_dependent"] else ""))

    if uninformative:
        print(f"\n  NOTE {', '.join(uninformative)} is model-independent: identical")
        print("       before and after by construction, so 'same' here is not")
        print("       evidence the adapter preserved anything.")

    # Promotion needs BOTH, and neither substitutes for the other:
    #   absolute    every gate at or above its own threshold after training
    #   relative    no gate below where the baseline already was
    # An adapter that improves a lot and still misses 0.95 is not promotable —
    # improvement is progress, not a product. And an adapter that clears every
    # threshold while regressing another gate is not promotable either.
    failed_after = [g["name"] for g in a["gates"] if not g["passed"]]
    baseline_low = b.get("below_target", [])
    print()
    if baseline_low:
        print(f"  baseline was below target on: {', '.join(baseline_low)} — "
              "recorded, not held against the adapter.")
    if regressed:
        print(f"DO NOT PROMOTE — regressed: {', '.join(regressed)}")
    if failed_after:
        print(f"DO NOT PROMOTE — below threshold after training: "
              f"{', '.join(failed_after)}")
        for name in failed_after:
            g = next(x for x in a["gates"] if x["name"] == name)
            prev = bg[name]["rate"]
            if g["rate"] > prev:
                print(f"  {name} improved {prev:.4f} -> {g['rate']:.4f} but the "
                      f"bar is {g['threshold']}. Improvement is not promotion.")
    promotable = not regressed and not failed_after
    if promotable:
        print("PROMOTABLE — every gate at or above threshold, and no regression.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "promotable": promotable, "regressed": regressed,
            "below_threshold_after": failed_after,
            "baseline_below_target": baseline_low,
            "gates": {g["name"]: {"before": bg[g["name"]]["rate"],
                                  "after": g["rate"],
                                  "threshold": g["threshold"],
                                  "model_dependent": g["model_dependent"]}
                      for g in a["gates"]},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    sys.exit(0 if promotable else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run every gate at one stage")
    r.add_argument("--config", default="train/qlora_9b.yaml")
    r.add_argument("--stage", choices=("before", "after"), required=True)
    # No defaults: the teacher was the default here and would have made the
    # "before" gates measure Muse Glimmer instead of the student.
    r.add_argument("--host", required=False, default=None)
    r.add_argument("--teacher", required=False, default=None)
    r.add_argument("--expect-model", default=None,
                   help="the config's base_model; gates refuse a different one")
    r.add_argument("--adapter", default=None)
    r.add_argument("--adapter-model-id", default=None,
                   help="the model id the endpoint registered the adapter under "
                        "(vLLM --lora-modules <id>=<path>). Required at --stage "
                        "after: without it the gates ask for the base model.")
    r.add_argument("--traces", default=None,
                   help="pre-generated voice traces instead of calling the model")
    r.add_argument("--edit-traces", default=None,
                   help="pre-generated edit traces instead of calling the model")
    r.add_argument("--out", required=True)

    c = sub.add_parser("compare", help="compare two stage reports")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--json-out", default=None)

    args = parser.parse_args()
    (cmd_run if args.command == "run" else cmd_compare)(args)


if __name__ == "__main__":
    main()
