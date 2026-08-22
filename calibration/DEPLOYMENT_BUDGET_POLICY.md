# Deployment budget policy

**Adopted 2026-08-22, BEFORE the gate was re-run under it.** Recorded first on
purpose: a budget changed after seeing a result it then permits is not a policy,
it is a rationalisation.

## The budget

| condition | limit |
|---|---|
| latency p95 | **≤ 30 s** |
| latency, any single request | **≤ 45 s** |
| reasoning characters | **0** |
| empty answers | **0** |
| truncated (`finish_reason: length`) | **0** |

## What changed and why

The previous per-request ceiling was 30 s, matching p95. Under it the pulled
`:latest` measured a slowest request of **exactly 30.0 s** — a pass only because
the comparison is strictly greater-than, on an unloaded machine with the model
already resident.

Zero margin is not a budget. A user's laptop is slower, colder, and busier than
the machine that produced that number, so the ceiling would be crossed in normal
use while the gate reported green.

Raising it to 45 s is **not** a loosening to accommodate a failing result: the
30 s measurement PASSED. It is an acknowledgement that a limit set equal to the
observed maximum encodes no headroom, and that p95 and worst case answer
different questions and should not share a number.

## What this does not license

- p95 stays at 30 s. The tail is bounded; the typical case is not allowed to
  drift toward it.
- The three absolute conditions stay absolute. There is no acceptable non-zero
  rate for an empty answer, a truncated answer, or reasoning returned as an
  answer.
- If a future result fails 45 s, the answer is to make it faster or to ship a
  smaller model — not to raise it again.

## Product obligation attached to this budget

A 45 s worst case is only acceptable if the user can see it happening and stop
it. The add-in must:

- show progress while a Studio action is running, not a frozen pane;
- allow cancellation of an in-flight request.

Without those, 45 s is indistinguishable from a hang — which is precisely the
failure this whole sequence began with.
