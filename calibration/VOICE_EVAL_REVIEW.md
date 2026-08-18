# Product review: voice eval v1 — NOT APPROVED

**Reviewed 2026-08-19 against the six criteria.** Four pass. Two findings block
approval. The gate is **not** wired in and the trainer stays BLOCKED.

**Reviewer caveat, stated first:** I drafted these items and this rubric, and I
am reviewing my own work. That is weaker than independent review by design. The
findings below are real, but the absence of further findings should carry less
weight than it would from a reviewer who had not written the material.

---

## 1. Office context and audience — PASS, with a gap

All 40 items are Office document tasks over plausible internal material:
facilities, archiving, energy, inventory, minutes, travel and vehicle policy.

**Gap:** only **5 of 40** name an audience in the instruction (`pimpinan`,
`direksi`, `seluruh pegawai`, `mitra eksternal`, `pimpinan unit`). Register
correctness depends on audience, so 35 items test register against an *implied*
office reader rather than a stated one. Defensible — the system prompt
establishes the persona — but it means the eval mostly tests "not casual" rather
than "correct for this reader".

## 2. Professional register — PASS

Three items deliberately tempt a customer-service reply with casual user
phrasing (`makasih ya`, `halo ... apa sih intinya?`, `tolong dong jelasin`).
A model that mirrors the user's register fails, which is the behaviour we want
to catch. Verified: `"Baik Kak, ..."` fails on both `register_address` and
`register_opener`.

## 3. Allow/deny terminology — **FINDING 1: the deny-list is too aggressive**

Stress-tested against natural professional Indonesian. Five of seven realistic
answers pass, and **two fail that should not**:

| answer | flagged | assessment |
|---|---|---|
| `Silakan unduh file laporan dari portal internal.` | `file -> berkas` | **false positive** |
| `Sistem melakukan update data setiap malam.` | `update -> pembaruan` | **false positive** |

`file` and `update` are established in Indonesian office usage in a way that
`setting`, `backup` and `user` are not — the same test the approved rule applies
("keep established domain terms, flag lazy generic anglicisms"). By that rule
they belong on the allow-list, not the deny-list.

`meeting`, `deadline`, `report`, `schedule`, `budget`, `approve` are retained on
the deny-list: each has a standard equivalent in routine written use
(`rapat`, `batas waktu`, `laporan`, `jadwal`, `anggaran`, `menyetujui`).

**Proposed change, needs your decision:** move `file` and `update` (and
`di-update`) to the allow-list. This is a judgement about Indonesian usage, not
a code fix, and I should not make it unilaterally.

## 4. Responses must not fabricate — PASS

Every extractive question has its answer present in its own source. Two items
deliberately ask about information the source does **not** contain
(`voice::0025` anggaran, `voice::0026` nama vendor) and instruct *"Jika tidak
ada, katakan demikian"* — testing that the model declines rather than invents.

**FINDING 2 (minor):** the scorer does not verify the model actually declined.
`score_voice.py` checks voice, not faithfulness. A fabricated budget figure in
fluent professional Indonesian would **pass** this gate. That is arguably
correct separation of concerns — faithfulness is `office_json_contract`'s and
the mechanical checks' job — but it should be recorded so nobody reads a voice
pass as a faithfulness pass.

## 5. Held-out — PASS

Exact reuse 0 against all four corpora. Content 8-gram overlap 1/40 (memo header
layout). Instruction-template overlap 6/40 is deliberate and approved. Scenario
domains disjoint from both the survey and the training corpus.

## 6. Source-term exemption is narrowly scoped — PASS, verified

Tested on `voice::0009` (exempt: `call center`, `aplikasi`):

| answer | result |
|---|---|
| exempt term, clean register | PASS |
| exempt term + `Baik Kak` | **FAIL** — register_address, register_opener |
| exempt term + `udah` | **FAIL** — baku |
| exempt term + `backup` | **FAIL** — terminology |

The exemption touches only the listed terms and only the terminology dimension.
It does not exempt an answer from register, baku, or language checks.

---

## Verdict

**NOT APPROVED pending two decisions:**

1. **Finding 1 (blocking):** move `file` and `update`/`di-update` to the
   allow-list? Without this the gate produces false failures on natural
   professional Indonesian, and a gate that cries wolf gets loosened under
   pressure — exactly what happened to the FP8 arm's gate 2.
2. **Finding 2 (record-only):** confirm that a voice pass is not read as a
   faithfulness pass, and that fabrication is covered elsewhere.

Also worth your call, non-blocking: whether 5-of-40 items naming an audience is
sufficient, or whether more should state their reader explicitly.

Threshold unchanged at 0.95 (38/40). Not adjusted, and not to be adjusted to
accommodate findings — the fix for a false positive is a correct list, not a
lower bar.
