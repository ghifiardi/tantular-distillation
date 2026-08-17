"""Fail early when a generator change would break router artifact distinctness.

    ./.venv/bin/python -m pytest tests/ -q

Source pack v3 gives every family its own document, and the pack audit gates
that. But the audit runs on a WRITTEN pack — by then the expensive part is done,
and the failure surfaces as "two families share a document" with no indication
of which change caused it.

Router artifacts are the fragile case: they are one sentence parameterised by
topic alone, so a router kind can yield at most as many distinct documents in a
split as that split has topics. v3 sits at margin 0 on its tightest group. These
tests fail at the point of change — raising instances_per_kind, reseeding the
splits, or trimming a topic pool — instead of after a regeneration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import author_sources_v3  # noqa: E402
import splits as splits_module  # noqa: E402

BASELINE_PATH = ROOT / "calibration" / "ROUTER_POOL_BASELINE.json"


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE_PATH.exists():
        pytest.fail(f"missing baseline {BASELINE_PATH}; "
                    "run src/write_router_pool_baseline.py --write")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    return splits_module.load()


def test_split_inputs_unchanged(baseline, manifest):
    """The baseline describes one split assignment. If that moved, every margin
    in it is about a partitioning that no longer exists."""
    assert manifest["fingerprint"] == baseline["split_fingerprint"], (
        "split fingerprint changed — families were repartitioned, so the recorded "
        "router margins no longer describe this corpus. Review the new margins and "
        "regenerate the baseline deliberately."
    )
    assert manifest["instances_per_kind"] == baseline["instances_per_kind"], (
        "instances_per_kind changed — more families per kind draw on the same "
        "topic pools, which is exactly what the margin 0 group cannot absorb."
    )


def test_topic_pools_not_shrunk(baseline):
    """Trimming a topic pool removes the headroom the margins were computed against."""
    for split, recorded in baseline["topics_per_split"].items():
        actual = len(author_sources_v3.WORLDS[split]["topics"])
        assert actual >= recorded, (
            f"{split} topic pool shrank from {recorded} to {actual}; router "
            f"distinctness in that split had no room to give up"
        )


def test_no_router_group_exceeds_its_pool(manifest):
    """The hard invariant: a router kind needing more distinct documents than its
    split has topics CANNOT be distinct, whatever the index spacing does."""
    over = [r for r in author_sources_v3.router_pool_report(manifest) if r["margin"] < 0]
    assert not over, (
        "router kinds demanding more documents than their split has topics: "
        + ", ".join(f"{r['kind']}/{r['split']} needs {r['families']} "
                    f"but only {r['topics']} topics exist" for r in over)
    )


def test_margins_have_not_degraded(baseline, manifest):
    """Margins may improve; a silent decrease is the regression."""
    current = {(r["kind"], r["split"]): r["margin"]
               for r in author_sources_v3.router_pool_report(manifest)}
    for row in baseline["groups"]:
        key = (row["kind"], row["split"])
        assert key in current, f"router group {key} disappeared from the taxonomy"
        assert current[key] >= row["margin"], (
            f"{row['kind']}/{row['split']} margin fell from {row['margin']} to "
            f"{current[key]}"
        )


def test_router_documents_are_actually_distinct(manifest):
    """The property all of the above protects, asserted directly.

    Composes each router family's document in memory — no pack on disk needed —
    so this catches a collision without a regeneration.
    """
    indices = author_sources_v3.family_index(manifest)
    seen = {}
    collisions = []
    for family, kind in sorted(manifest["kinds"].items()):
        if not kind.startswith("router:"):
            continue
        scenario = author_sources_v3.scenario_for(
            indices[family], manifest["assignments"][family])
        text = author_sources_v3.build(kind, scenario)
        if text in seen:
            collisions.append((seen[text], family))
        seen[text] = family
    assert not collisions, (
        f"{len(collisions)} router families share an identical document: "
        f"{collisions[:3]}"
    )
