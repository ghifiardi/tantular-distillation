"""Fail-closed distillation planner (DRAFT — authorizes nothing).

Reads a run plan (configs/distillation/<name>.yaml), the model registry
(configs/models/<name>.yaml), architecture profiles, and the declared fleet
(configs/hosts/<name>.yaml), then either emits a plan or REFUSES. It performs no
network call, imports no GPU library, and starts no training. It is the *how*,
still gated behind train/TRAINING_BLOCKED.md's *whether*.

    # emit or refuse a plan
    ./.venv/bin/python src/distill_plan.py plan office-v2-sequence

    # freshness is relative to a date you can pin for reproducibility
    ./.venv/bin/python src/distill_plan.py plan office-v2-sequence --today 2026-09-03

    # compute an architecture signature from a real config.json
    ./.venv/bin/python src/distill_plan.py arch-signature /path/to/config.json

WHAT IT REFUSES, and why each refusal exists:

  licence not permitted / stale     Apache 2.0 held across Qwen 3.5/3.6 and
                                    STOPPED at the 3.8 flagship; nemotron ships
                                    under NVIDIA OML. A licence recorded once and
                                    trusted forever is a hazard, so it is a GATE
                                    re-checked for freshness on every run.

  Mode C across a tokenizer split   Token-level KL needs a shared vocabulary. The
                                    repo's real teachers are non-Qwen, so the
                                    compatibility key (tokenizer sha256) usually
                                    differs. auto downgrades to B/A; an explicit
                                    on_policy_kd request is refused, never
                                    silently downgraded.

  no host can serve the teacher     The fleet tops out at one rented 48GB Ada and
                                    ai19 cannot do FP8. Rather than emit a plan
                                    for a 2x80GB machine the project lacks, the
                                    planner reports what must be procured.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # keep the failure legible
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent

BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}
# Weights are not the whole story: KV cache, activations, runtime metadata. A
# conservative multiplier keeps the planner from declaring a card "enough" when
# published deployment guidance sits well above the weight math.
SERVING_OVERHEAD = 1.35


def die(msg: str, code: int = 2) -> None:
    print(f"\nPLANNER REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


def _load(kind: str, name: str) -> dict:
    path = ROOT / "configs" / kind / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (ROOT / "configs" / kind).glob("*.yaml"))
        die(f"no such {kind} config: {name!r} (have: {', '.join(available) or 'none'})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        die(f"{kind}/{name}.yaml must be a mapping")
    return data


# --- schema validation ------------------------------------------------------

_REQUIRED_MODEL_KEYS = ("model_id", "role", "tokenizer", "license")


def validate_model(spec: dict, name: str) -> dict:
    for key in _REQUIRED_MODEL_KEYS:
        if key not in spec:
            die(f"model {name!r} is missing required field {key!r}")
    if spec.get("role") not in ("student", "teacher"):
        die(f"model {name!r} role must be 'student' or 'teacher', got {spec.get('role')!r}")
    tok = spec.get("tokenizer") or {}
    if not isinstance(tok, dict):
        die(f"model {name!r} tokenizer must be a mapping")
    lic = spec.get("license") or {}
    if not isinstance(lic, dict):
        die(f"model {name!r} license must be a mapping")
    return spec


def _digest_present(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


# --- gates ------------------------------------------------------------------

def license_status(spec: dict, today: _dt.date) -> dict:
    """Structured licence verdict, shared by the planner gate and the audit.

    A licence recorded once and trusted forever is a hazard: Apache 2.0 held
    across Qwen 3.5/3.6 and STOPPED at the 3.8 flagship. Freshness is therefore
    part of the verdict, not metadata.
    """
    lic = spec.get("license") or {}
    permitted = lic.get("output_training_permitted") is True
    reviewed = lic.get("reviewed_at")
    max_age = lic.get("recheck_max_age_days")
    evidence = _digest_present(lic.get("evidence_sha256"))
    age = None
    status = "FRESH"
    reason = None
    problems: list[str] = []
    if not permitted:
        status = "NOT_PERMITTED"
        problems.append("output_training_permitted is not true")
    else:
        if not reviewed or not isinstance(max_age, int):
            status = "UNKNOWN"
            reason = "no reviewed_at/recheck_max_age_days"
            problems.append(reason)
        else:
            try:
                rd = _dt.date.fromisoformat(str(reviewed))
            except ValueError:
                status = "UNKNOWN"
                reason = f"reviewed_at not an ISO date: {reviewed!r}"
                problems.append(reason)
            else:
                age = (today - rd).days
                if age < 0:
                    # A review dated after the day being asked about is not
                    # fresh, it is wrong — and it would otherwise sail through
                    # the max-age comparison forever.
                    status = "UNKNOWN"
                    reason = f"reviewed_at {rd.isoformat()} is after {today.isoformat()}"
                    problems.append(reason)
                elif age > max_age:
                    status = "STALE"
                    reason = f"{age} days old, limit {max_age}"
                    problems.append(reason)
        if not evidence:
            problems.append("evidence_sha256 empty")
            if status == "FRESH":
                status = "FRESH_NO_EVIDENCE"
    return {
        "identifier": lic.get("identifier"),
        "output_training_permitted": permitted,
        "reviewed_at": str(reviewed) if reviewed is not None else None,
        "age_days": age,
        "max_age_days": max_age if isinstance(max_age, int) else None,
        "evidence_present": evidence,
        "status": status,
        "reason": reason,
        "problems": problems,
    }


def license_gate(teacher: dict, name: str, today: _dt.date) -> None:
    """Teacher outputs feed training in every mode, so the licence gate always
    applies to the teacher. Single-sourced with license_status()."""
    st = license_status(teacher, today)
    if st["problems"]:
        detail = "; ".join(st["problems"])
        die(f"teacher {name!r} licence gate FAILED ({st['status']}): {detail}. "
            "This is a GATE, not a note (Apache 2.0 stopped holding at the 3.8 "
            "flagship).")


def compatibility_key(spec: dict) -> str | None:
    tok = spec.get("tokenizer") or {}
    sha = tok.get("sha256")
    return sha if _digest_present(sha) else None


def tokenizers_compatible(teacher: dict, student: dict) -> bool:
    t, s = compatibility_key(teacher), compatibility_key(student)
    return bool(t and s and t == s)


# --- registry <-> serving config reconciliation ------------------------------

def _serving_config(name: str) -> dict | None:
    path = ROOT / "configs" / "teachers" / f"{name}.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else None


def serving_mismatches(spec: dict, name: str) -> list[str]:
    """Where a registry entry and its serving config disagree about identity.

    configs/models/ holds IDENTITY — model_id, tokenizer digests, licence
    freshness, architecture profile. configs/teachers/ holds SERVING — port,
    repo per precision, sampling. They describe the same checkpoint and are read
    by DIFFERENT tools: src/config.py resolves the serving config for the gates,
    this planner reads the registry. So a disagreement does not produce an
    error; it produces a plan written for one checkpoint and gates measuring
    another. src/model_ids.py exists because Qwen3.5-9B and Qwen3.5-9B-Base are
    one suffix apart; this is that same hazard one level up.

    Only the fields that must agree are compared. A serving config's fp8 / int4 /
    gateway repos are deliberately DIFFERENT artefacts (RedHatAI's FP8 block, an
    Ollama tag, a LiteLLM id) and are not identity claims. bf16 is: it is the
    same weights, which is why configs/teachers/office-student-9b.yaml carries
    bf16 alone.
    """
    sys.path.insert(0, str(ROOT / "src"))
    import model_ids

    link = spec.get("serving_config")
    if not link:
        return []
    serving = _serving_config(str(link))
    if serving is None:
        return [f"serving_config {link!r} names no file under configs/teachers/"]

    problems: list[str] = []
    back = serving.get("registry_model")
    if back != name:
        problems.append(
            f"configs/teachers/{link}.yaml says registry_model: {back!r}, but "
            f"configs/models/{name}.yaml points at it. The link must be mutual "
            "or one of them is stale.")

    model_id = spec.get("model_id")
    served = serving.get("served_model_name")
    # A served_model_name without a "/" is a local serving alias (vLLM's
    # --served-model-name, an Ollama tag). Only a repo-shaped one is an identity
    # claim, and then it must be the registry's.
    if served and "/" in str(served) and not model_ids.matches(str(model_id), served):
        problems.append(f"served_model_name {served!r} != registry model_id {model_id!r}")

    bf16 = (serving.get("repos") or {}).get("bf16")
    if bf16 and not model_ids.matches(str(model_id), bf16):
        problems.append(f"repos.bf16 {bf16!r} != registry model_id {model_id!r}")

    tokenizer_repo = serving.get("tokenizer")
    declared = (spec.get("tokenizer") or {}).get("model_id") or model_id
    if tokenizer_repo and not model_ids.matches(str(declared), tokenizer_repo):
        problems.append(f"tokenizer {tokenizer_repo!r} != registry "
                        f"tokenizer.model_id {declared!r}")

    identifier = (spec.get("license") or {}).get("identifier")
    served_licence = serving.get("license")
    if served_licence and identifier and str(served_licence) != str(identifier):
        problems.append(f"license {served_licence!r} != registry license.identifier "
                        f"{identifier!r}")
    return problems


def reconcile_gate(spec: dict, name: str) -> None:
    problems = serving_mismatches(spec, name)
    if problems:
        die(f"registry model {name!r} and its serving config disagree:\n  - "
            + "\n  - ".join(problems)
            + "\nThe gates read the serving config and this planner reads the "
              "registry. While they differ, a plan and its measurements are "
              "about different checkpoints.")


# --- mode selection ---------------------------------------------------------

MODE_SEQUENCE = "sequence"
MODE_PREFERENCE = "preference"
MODE_C = "on_policy_kd"


def _mode_c_preflight(teacher: dict, tname: str, student: dict, arch: dict | None) -> list[str]:
    """Non-tokenizer reasons Mode C cannot run. Returned as a list so 'auto' can
    fall back and an explicit request can report all of them at once."""
    errs: list[str] = []
    if compatibility_key(teacher) is None or compatibility_key(student) is None:
        errs.append("a tokenizer.sha256 is missing on one side")
    elif not tokenizers_compatible(teacher, student):
        errs.append("tokenizer compatibility keys differ (cross-family): "
                    "token-level KL needs a shared vocabulary")
    if (teacher.get("capabilities") or {}).get("logprobs") is not True:
        errs.append(f"teacher {tname!r} does not expose logprobs")
    if not _digest_present(teacher.get("revision")):
        errs.append(f"teacher {tname!r} revision is not pinned")
    if arch is None:
        errs.append("student has no architecture_profile for Mode C")
    else:
        mc = arch.get("mode_c") or {}
        if mc.get("eligible") is not True:
            errs.append(f"architecture {arch.get('name')!r} is not Mode-C eligible")
        if mc.get("targets_lm_head") is True:
            errs.append("profile targets lm_head, which DistillationTrainer rejects")
    return errs


def choose_mode(requested: str, plan: dict, teacher: dict, tname: str,
                student: dict, arch: dict | None) -> tuple[str, list[str]]:
    """Return (mode, notes). Explicit requests are refused on failure; 'auto'
    downgrades on_policy_kd -> preference -> sequence."""
    notes: list[str] = []
    has_pairs = bool((plan.get("preference") or {}).get("scope"))

    if requested in (MODE_SEQUENCE, MODE_PREFERENCE, MODE_C):
        if requested == MODE_C:
            errs = _mode_c_preflight(teacher, tname, student, arch)
            if errs:
                die("explicit mode 'on_policy_kd' cannot run:\n  - " + "\n  - ".join(errs)
                    + "\nUse mode 'auto' to fall back to preference/sequence, or fix the above.")
        if requested == MODE_PREFERENCE and not has_pairs:
            die("explicit mode 'preference' but plan.preference.scope is empty")
        return requested, notes

    if requested != "auto":
        die(f"unknown mode {requested!r} (use sequence|preference|on_policy_kd|auto)")

    # auto
    errs = _mode_c_preflight(teacher, tname, student, arch)
    if not errs:
        return MODE_C, ["auto selected on_policy_kd (compatibility key matches)"]
    notes.append("auto: on_policy_kd unavailable -> " + "; ".join(errs))
    if has_pairs:
        notes.append("auto fell back to preference (judge-gated)")
        return MODE_PREFERENCE, notes
    notes.append("auto fell back to sequence")
    return MODE_SEQUENCE, notes


# --- hardware ---------------------------------------------------------------

def _weight_gb(spec: dict, precision: str) -> tuple[float, bool]:
    params = spec.get("params") or {}
    total = params.get("total_b")
    if not isinstance(total, (int, float)):
        die(f"model {spec.get('model_id')!r} params.total_b is required for a "
            "memory estimate")
    per = BYTES_PER_PARAM.get(precision)
    if per is None:
        die(f"unknown precision {precision!r}")
    incomplete = False
    vision = params.get("vision_b")
    if spec.get("modality", {}).get("vision") and not isinstance(vision, (int, float)):
        incomplete = True  # vision tower unaccounted
        vision = 0.0
    grand = float(total) + float(vision or 0.0)
    return grand * per, incomplete


def _fleet() -> list[dict]:
    hosts = []
    for path in sorted((ROOT / "configs" / "hosts").glob("*.yaml")):
        h = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        h["_name"] = path.stem
        hosts.append(h)
    return hosts


def _host_capacity_gb(host: dict) -> float | None:
    """Declared usable VRAM. Missing declaration -> unknown (host ineligible),
    which is the fail-closed choice."""
    per = host.get("gpu_memory_gb")
    count = host.get("gpu_count") or host.get("tensor_parallel_size") or 1
    if not isinstance(per, (int, float)):
        return None
    util = host.get("gpu_memory_utilization", 0.90)
    try:
        util = float(util)
    except (TypeError, ValueError):
        util = 0.90
    return float(per) * float(count) * util


def hardware_plan(teacher: dict, precisions: list[str]) -> dict:
    supports_fp8_only = {"fp8"}
    candidates = []
    for precision in precisions:
        need_gb, incomplete = _weight_gb(teacher, precision)
        need_gb *= SERVING_OVERHEAD
        eligible = []
        for host in _fleet():
            if host.get("teacher_serving") is False:
                continue
            if precision in supports_fp8_only and host.get("supports_fp8") is False:
                continue
            cap = _host_capacity_gb(host)
            if cap is None:
                continue
            if cap >= need_gb:
                eligible.append((host["_name"], round(cap, 1)))
        candidates.append({
            "precision": precision,
            "estimated_need_gb": round(need_gb, 1),
            "estimate_incomplete": incomplete,
            "eligible_hosts": eligible,
        })
    return {"attempts": candidates}


# --- signature --------------------------------------------------------------

def _shape_fields(config: dict) -> dict:
    """Where the language-model shape actually lives.

    Qwen3.5 ships a UNIFIED VL config: the top level holds only model_type,
    vision_config and the image/video token ids, while every load-bearing text
    field — num_hidden_layers, layer_types, hidden_size — sits under
    text_config. Reading the top level of such a config yields all-null, and a
    signature over all-null matches a 32-layer dense student and a 48-layer MoE
    alike. Verified against Qwen3.5-9B-Base's real config.json.
    """
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def describe_architecture(config: dict) -> dict:
    """The load-bearing shape of a HF config.json, flat or nested."""
    shape = _shape_fields(config)
    vision = config.get("vision_config")
    return {
        "model_type": config.get("model_type"),
        "text_model_type": shape.get("model_type") if shape is not config else None,
        "num_hidden_layers": shape.get("num_hidden_layers"),
        "layer_types": shape.get("layer_types"),
        "hidden_size": shape.get("hidden_size"),
        "num_attention_heads": shape.get("num_attention_heads"),
        "num_key_value_heads": shape.get("num_key_value_heads"),
        "num_experts": shape.get("num_experts") or shape.get("num_local_experts"),
        "num_experts_per_tok": shape.get("num_experts_per_tok"),
        # A swapped vision tower is a different model even when the text stack
        # is identical, and the student is a VL checkpoint.
        "vision": {
            "model_type": vision.get("model_type"),
            "depth": vision.get("depth"),
            "hidden_size": vision.get("hidden_size"),
            "out_hidden_size": vision.get("out_hidden_size"),
        } if isinstance(vision, dict) else None,
    }


def architecture_signature(config: dict) -> str:
    """Stable digest over the load-bearing shape fields of a HF config.json."""
    descriptor = describe_architecture(config)
    if descriptor["num_hidden_layers"] is None and descriptor["layer_types"] is None:
        die("this config declares neither num_hidden_layers nor layer_types, at "
            "the top level or under text_config. A signature over an empty shape "
            "would match every model, so none is computed.")
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# --- commands ---------------------------------------------------------------

def dry_run_sequence(plan_name: str, mode: str, teacher: dict, teacher_name: str,
                     student: dict, host: str | None) -> list[str]:
    """The existing tools a plan maps onto, in order. Prints; runs nothing.

    Every command below already exists in this repo. The point of writing them
    out is that a distillation plan is not a new pipeline — it is the generate ->
    verify -> promote -> freeze -> gate sequence the v1 run already used, with
    the teacher and mode chosen by the planner instead of by hand.
    """
    teacher_serving = teacher.get("serving_config") or teacher_name
    student_serving = student.get("serving_config") or "<student-serving-config>"
    corpus = f"data/raw/{plan_name}.r0.jsonl"
    host = host or "<host-with-declared-capacity>"
    lines = [
        "",
        "=== DRY RUN — commands only. Nothing was executed, generated, or written. ===",
        "",
        f"# mode {mode}: the teacher produces completions; no logits cross the",
        "# tokenizer boundary. <prompt-set> is not chosen here — pick a held-out",
        "# set and record it; this planner does not invent one.",
        "",
        "# 1. generate traces from the teacher, on the host the plan qualified",
        f"./.venv/bin/python src/generate.py --teacher {teacher_serving} \\",
        f"    --host {host} --prompts <prompt-set>.jsonl --out {corpus}",
        "",
        "# 2. the corpus gate. Its exit code is evidence, not a parameter.",
        f"./.venv/bin/python src/verify_corpus.py {corpus} --gate",
        "",
        "# 3. promote what passes the mechanical checks",
        "./.venv/bin/python src/promote_corpus.py \\",
        f"    --traces {corpus} --prompts <prompt-set>.jsonl \\",
        "    --out-dir data/promoted --manifest train/RUN_MANIFEST.<v>-mechanical.json \\",
        "    --frozen-at <iso-timestamp> --write",
        "",
        "# 4. measure the BASE STUDENT before anything is trained",
        f"./.venv/bin/python src/run_gates.py run --stage before \\",
        f"    --host <student-host> --teacher {student_serving} \\",
        f"    --expect-model {student.get('model_id')} --out data/gates/before.json",
        "",
        "# 5. freeze the exact inputs (records the provenance audit as evidence)",
        "./.venv/bin/python src/freeze_training_run.py \\",
        f"    --corpus {corpus} --config <training-config>.yaml \\",
        "    --promotion-manifest train/RUN_MANIFEST.<v>-mechanical.json \\",
        "    --out train/RUN_MANIFEST.<v>.json --frozen-at <iso-timestamp> --write",
        "",
        "# 6. TRAINING — BLOCKED. train/TRAINING_BLOCKED.md is controlling and",
        "#    this planner does not lift it. What reopens it is a real observed",
        "#    failure recorded in a train/TRAINING_JUSTIFIED.md, not a plan that",
        "#    validated. No training command is printed here.",
        "",
        "# 7. after the (still hypothetical) run: measure the adapter and compare",
        "./.venv/bin/python src/run_gates.py run --stage after \\",
        f"    --host <student-host> --teacher {student_serving} \\",
        f"    --expect-model {student.get('model_id')} \\",
        "    --adapter <adapter-dir> --out data/gates/after.json",
        "./.venv/bin/python src/run_gates.py compare \\",
        "    --before data/gates/before.json --after data/gates/after.json",
        "",
        "Angle-bracketed values are NOT defaults. They are the decisions this",
        "planner will not make for you: which held-out prompts, which timestamp,",
        "which training config. Filling them in with a guess is how a corpus ends",
        "up measuring something nobody chose.",
    ]
    return lines


def cmd_plan(args: argparse.Namespace) -> None:
    today = _dt.date.fromisoformat(args.today) if args.today else _dt.date.today()
    plan = _load("distillation", args.plan)

    student_name = plan.get("student") or die("plan has no 'student'")
    teacher_name = plan.get("teacher") or die("plan has no 'teacher'")
    student = validate_model(_load("models", student_name), student_name)
    teacher = validate_model(_load("models", teacher_name), teacher_name)

    if student.get("role") != "student":
        die(f"{student_name!r} is not role: student")
    if teacher.get("role") != "teacher":
        die(f"{teacher_name!r} is not role: teacher")

    # GATE 0: the registry and the gate configs must describe one checkpoint.
    reconcile_gate(student, student_name)
    reconcile_gate(teacher, teacher_name)

    # GATE 1: teacher licence (applies in every mode).
    license_gate(teacher, teacher_name, today)

    # Architecture profile for the student (needed for Mode C, checked at train).
    arch = None
    arch_name = student.get("architecture_profile")
    if arch_name:
        arch = _load("architectures", arch_name)

    # GATE 2 + mode selection (compatibility key decides Mode C).
    mode, notes = choose_mode(str(plan.get("mode", "auto")), plan, teacher,
                              teacher_name, student, arch)

    # GATE 3: hardware — refuse rather than invent a machine.
    precisions = plan.get("precision_preference") or ["fp8", "bf16"]
    hw = hardware_plan(teacher, precisions)
    servable = [a for a in hw["attempts"] if a["eligible_hosts"]]

    warnings = list(notes)
    if teacher.get("digests_verified") is not True:
        warnings.append(f"teacher {teacher_name!r} digests_verified is not true; "
                        "verify tokenizer/template sha256 before a real corpus run")
    if student.get("digests_verified") is not True:
        warnings.append(f"student {student_name!r} digests_verified is not true")
    for role, spec_, spec_name in (("student", student, student_name),
                                   ("teacher", teacher, teacher_name)):
        if not spec_.get("serving_config"):
            warnings.append(f"{role} {spec_name!r} declares no serving_config; the "
                            "gates' view of this checkpoint is unreconciled")
    for a in hw["attempts"]:
        if a["estimate_incomplete"]:
            warnings.append(f"memory estimate for {a['precision']} is INCOMPLETE "
                            "(vision tower params unknown)")

    result = {
        "plan": args.plan,
        "today": today.isoformat(),
        "student": {"name": student_name, "model_id": student["model_id"]},
        "teacher": {"name": teacher_name, "model_id": teacher["model_id"]},
        "objective": plan.get("objective"),
        "selected_mode": mode,
        "compatibility_key_match": tokenizers_compatible(teacher, student),
        "replay": plan.get("replay"),
        "hardware": hw,
        "servable": bool(servable),
        "warnings": warnings,
        "authorizes_training": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if getattr(args, "dry_run", False) and servable:
        first = next(a for a in hw["attempts"] if a["eligible_hosts"])
        print("\n".join(dry_run_sequence(
            args.plan, mode, teacher, teacher_name, student,
            first["eligible_hosts"][0][0])))

    if not servable:
        die("no declared host can serve this teacher at any preferred precision. "
            "Add gpu_memory_gb/gpu_count to a host that can, or procure hardware. "
            "The planner will not emit a plan for a machine the fleet lacks.")


def cmd_arch_sig(args: argparse.Namespace) -> None:
    path = Path(args.config)
    if not path.is_file():
        die(f"no config.json at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    print(architecture_signature(config))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        die(f"no corpus at {path}")
    rows: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            die(f"{path}:{i}: bad JSON — {e}")
    if not rows:
        die(f"{path} has no records")
    return rows


def _counter(rows: list[dict], *keys: str) -> dict:
    from collections import Counter
    c: Counter = Counter()
    for r in rows:
        d: object = r
        for k in keys[:-1]:
            d = (d or {}).get(k, {}) if isinstance(d, dict) else {}
        c[(d or {}).get(keys[-1], "?") if isinstance(d, dict) else "?"] += 1
    return dict(c)


def _resolve_teacher_specs(corpus_teachers: list, overrides: list | None):
    """Map corpus provenance teacher names -> configs/models/*.yaml specs.

    Explicit --teacher overrides win. Otherwise auto-resolve by substring against
    each spec's registry name, model_id, basename, and family.
    """
    if overrides:
        specs = {name: validate_model(_load("models", name), name) for name in overrides}
        return specs, []
    all_specs = []
    for p in sorted((ROOT / "configs" / "models").glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(d, dict):
            d["_name"] = p.stem
            all_specs.append(d)
    specs: dict = {}
    unresolved: list = []
    for t in corpus_teachers:
        tl = str(t).lower()
        match = None
        for d in all_specs:
            hay = " ".join(str(x).lower() for x in (
                d.get("_name"), d.get("model_id"), d.get("family"),
                Path(str(d.get("model_id", ""))).name))
            if tl and tl in hay:
                match = d
                break
        if match:
            specs[match["_name"]] = match
        else:
            unresolved.append(t)
    return specs, unresolved


def _harness_coverage(path: Path) -> dict:
    """Harness attribution for the combined audit: what was DECLARED, and what
    the traces actually carry.

    Both halves matter, and separately. The declaration comes from the adjacent
    pass manifest exactly as the corpus gate reads it; the coverage is counted
    from the bytes. A corpus that declares harness-aware mode and does not
    deliver it is not simply "0% covered" — it is a disagreement, and saying so
    is more useful than either number alone.

    Reuses harness_distill rather than recomputing: a second implementation of
    "is this attributed?" is a second answer waiting to differ.
    """
    sys.path.insert(0, str(ROOT / "src"))
    import harness_distill
    import verify_corpus

    declared = verify_corpus.harness_declaration([path])
    required = bool(declared)

    try:
        counted = harness_distill.audit_traces(path)
    except harness_distill.HarnessPlanError as exc:
        return {"required": required, "attributed": False, "coverage": 0.0,
                "error": str(exc)}

    block = {
        "required": required,
        "attributed": counted["harness_attributed"] > 0,
        "coverage": counted["harness_coverage"],
    }
    if counted["harness_digests"]:
        block["digests"] = counted["harness_digests"]

    # Strict in BOTH modes. Running it only when required=True meant a corpus
    # DECLARED legacy while carrying attributed traces reported attributed:
    # true with no disagreement — a contradiction stated as a fact. The
    # declaration decides which mode to check in; it does not decide whether to
    # check at all.
    try:
        observed = harness_distill.summarize_harness_attribution(
            _load_jsonl(path), required=required)
    except harness_distill.HarnessPlanError as exc:
        block["disagreement"] = str(exc)
        return block

    # The strict check RUNS in both modes; only its bookkeeping is recorded for
    # a harness-aware corpus. A legacy corpus keeps the minimal three-key block
    # it has always reported, so a reader cannot mistake extra fields for the
    # corpus having gained an attribution it does not have.
    if required:
        block["declared"] = declared
        block["observed"] = observed
        if observed != declared:
            block["disagreement"] = (
                "the pass manifest declares a different harness state than "
                "the traces carry")
    return block


def audit_corpus(corpus: str | Path, today: _dt.date,
                 teacher_overrides: list | None = None) -> dict:
    """The mechanical limits of a promoted corpus, as a dict.

    Importable so the freeze (src/freeze_training_run.py) records the same
    verdict the CLI prints, rather than a second implementation of it. Reports;
    it decides nothing and authorizes nothing.
    """
    path = Path(corpus)
    rows = _load_jsonl(path)
    n = len(rows)

    teachers = _counter(rows, "provenance", "teacher")
    repos = _counter(rows, "provenance", "repo")
    quants = _counter(rows, "provenance", "quantization")
    hosts = _counter(rows, "provenance", "host")
    licenses_recorded = _counter(rows, "provenance", "license")
    source_classes = _counter(rows, "source_class")

    # FP8 gate — single-sourced from verify_corpus so the rule cannot drift.
    sys.path.insert(0, str(ROOT / "src"))
    import verify_corpus as vc
    untrainable = {q: c for q, c in quants.items() if q in vc.UNTRAINABLE_QUANTIZATION}
    fp8_met = not untrainable

    pair_counts: dict = {}
    for r in rows:
        p = r.get("provenance", {}) or {}
        q, h = p.get("quantization"), p.get("host")
        if q in vc.UNTRAINABLE_QUANTIZATION:
            pair_counts[(q, h)] = pair_counts.get((q, h), 0) + 1
    uncovered = {f"{q}@{h}": c for (q, h), c in pair_counts.items()
                 if (q, h) not in vc.WAIVER_COVERED}
    waiver_file = ROOT / "calibration" / "INT4_WAIVER.md"

    synthetic = source_classes.get("synthetic", 0)
    supports_real = synthetic == 0 and all(k != "synthetic" for k in source_classes)

    specs, unresolved = _resolve_teacher_specs(list(teachers), teacher_overrides)
    freshness = []
    for name, spec in specs.items():
        st = license_status(spec, today)
        st["registry_model"] = name
        st["model_id"] = spec.get("model_id")
        st["digests_verified"] = spec.get("digests_verified") is True
        freshness.append(st)

    limits: list[str] = []
    if not fp8_met:
        limits.append(
            f"FP8 gate UNMET: {sum(untrainable.values())} trace(s) from a quantized "
            f"teacher {untrainable} — no FP8 claim may be made.")
        if uncovered:
            limits.append(
                f"Quantized traces NOT covered by the signed waiver: {uncovered} "
                "(waiver names int4_ollama@ai19-ollama only).")
        elif waiver_file.is_file():
            limits.append(
                "Quantized traces are within calibration/INT4_WAIVER.md scope; the "
                "waiver authorises proceeding DESPITE the failure — it does not "
                "convert it to a pass.")
        else:
            limits.append(
                "No calibration/INT4_WAIVER.md present to authorise proceeding "
                "despite the failure.")
    if synthetic:
        limits.append(
            f"{synthetic}/{n} trace(s) are source_class: synthetic — supports NO "
            "claim about real Office documents.")
    for st in freshness:
        if st["status"] != "FRESH":
            suffix = f" ({st['reason']})" if st.get("reason") else ""
            limits.append(f"Teacher licence {st['registry_model']!r}: {st['status']}{suffix}.")
        if not st["digests_verified"]:
            limits.append(
                f"Registry model {st['registry_model']!r} digests_verified is not "
                "true; tokenizer/template sha256 unverified.")
    if unresolved:
        limits.append(
            f"Could not resolve a registry model for corpus teacher(s) {unresolved}; "
            "licence freshness UNKNOWN. Pass --teacher <name>.")

    harness = _harness_coverage(path)
    harness_ready = not harness.get("disagreement") and (
        not harness.get("required") or harness.get("coverage") == 1.0)

    # Identity is a REQUIREMENT, not a footnote. An unverified tokenizer digest
    # means the compatibility key that decided the distillation mode was never
    # checked against real files, so "trainable as is" would be a claim about a
    # checkpoint nobody confirmed. See src/verify_model_identity.py.
    unverified = [s["registry_model"] for s in freshness if not s["digests_verified"]]
    identity_verified = bool(freshness) and not unverified

    # Each requirement, named and reported separately. A single boolean says a
    # corpus is not trainable; it does not say WHY, and "still false" is not
    # evidence that the reason is unchanged — a new blocker can appear while an
    # old one is fixed and the verdict never moves. trainable_as_is is derived
    # from this block rather than computed alongside it, so the two cannot drift.
    readiness = {
        "fp8_ready": fp8_met,
        "source_ready": synthetic == 0,
        "identity_ready": identity_verified,
        "license_ready": bool(freshness)
                         and all(s["status"] == "FRESH" for s in freshness)
                         and not unresolved,
        "harness_ready": harness_ready,
    }
    trainable_as_is = all(readiness.values())

    if harness.get("disagreement"):
        limits.append("harness attribution is incoherent: " + harness["disagreement"])
    elif harness.get("required") and harness["coverage"] < 1.0:
        limits.append(
            "harness-aware mode is declared but attribution is incomplete: "
            f"{harness['coverage']:.0%} coverage")
    # No limit line for an unattributed corpus. That is the state of every
    # corpus predating harness attribution, so a line here would fire on every
    # legacy audit and reshape a report that is supposed to gain a block, not a
    # complaint. The harness block already says coverage is 0.0; what earns a
    # limit is a corpus that CLAIMS attribution and does not deliver it.

    result = {
        "corpus": str(path),
        "records": n,
        "teachers": teachers,
        "repos": repos,
        "quantizations": quants,
        "hosts": hosts,
        "licenses_recorded": licenses_recorded,
        "source_classes": source_classes,
        "fp8_gate": {
            "status": "MET" if fp8_met else "UNMET",
            "untrainable_quantizations": untrainable,
            "waiver_file_present": waiver_file.is_file(),
            "uncovered_by_waiver": uncovered,
        },
        "real_office_claim": {
            "supported": supports_real,
            "synthetic_traces": synthetic,
        },
        "license_freshness": {
            "resolved": freshness,
            "unresolved_corpus_teachers": unresolved,
        },
        "harness": harness,
        "readiness": readiness,
        "identity_verification": {
            "all_verified": identity_verified,
            "unverified_models": unverified,
        },
        "limits": limits,
        "trainable_as_is": trainable_as_is,
        "authorizes_training": False,
    }
    return result


def cmd_provenance_audit(args: argparse.Namespace) -> None:
    today = _dt.date.fromisoformat(args.today) if args.today else _dt.date.today()
    report = audit_corpus(args.corpus, today, args.teacher)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="emit or refuse a distillation plan")
    p.add_argument("plan", help="name under configs/distillation/ (no .yaml)")
    p.add_argument("--today", default=None, help="ISO date for licence freshness")
    p.add_argument("--dry-run", action="store_true",
                   help="also print the existing generate -> verify -> promote -> "
                        "freeze -> gate commands this plan maps onto. Prints only: "
                        "no trace is emitted, no file is written, and training "
                        "stays blocked")
    p.set_defaults(func=cmd_plan)

    s = sub.add_parser("arch-signature", help="digest a HF config.json")
    s.add_argument("config", help="path to config.json")
    s.set_defaults(func=cmd_arch_sig)

    a = sub.add_parser("provenance-audit",
                       help="mechanical limits of a promoted corpus")
    a.add_argument("corpus", help="path to a promoted JSONL corpus")
    a.add_argument("--today", default=None, help="ISO date for licence freshness")
    a.add_argument("--teacher", action="append", default=None,
                   help="registry model name to use for licence lookup "
                        "(repeatable); default auto-resolves from provenance")
    a.set_defaults(func=cmd_provenance_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
