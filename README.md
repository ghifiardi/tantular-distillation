# tantular-distillation

Capability distillation for Tantular: open-weight teachers → curated corpus →
fine-tuned Tantular Office models.

Teachers run wherever there is a GPU. Nothing here depends on a hosted API or
a gateway key — the weights are open, so the pipeline is unblocked by default.

## Where things run

| Stage | Runs on | Why |
|---|---|---|
| Teacher serving | gateway (int4) or rented Ada/Hopper (fp8) | 30B needs 32GB+ at FP8 |
| Trace generation | Anywhere with network to the teacher | Just HTTP |
| Judge / dedup / promote | Anywhere | CPU work |
| QLoRA training | Single 24GB+ GPU | 9B 4-bit + LoRA |
| Serving the student | The user's laptop, via Ollama | The whole point |

## Hosts

- `ai19` — 2×RTX **3090** 24GB (Ampere, cc 8.6). **Cannot do FP8** (needs 8.9+).
  Shared production box: its Ollama backs the openai.ina17.com gateway and a
  face_ai_service runs alongside, so ~10GB/12GB free at survey time, not 48GB.
  **Training host, not a teacher host** — `--host gateway` already reaches this
  machine's weights.
- `gateway` — openai.ina17.com, i.e. ai19's Ollama. int4, ~40s/request,
  operator-visible. The only teacher path available without renting.
- `rented-48gb` — one RTX 6000 Ada or L40S (~$0.74–0.79/hr on RunPod
  community). Same 48GB on a single card, so no PCIe hop. Use when a run
  needs to finish fast.
- `mac-validate` — M4 Pro 24GB, int4 via Metal. Validation scale only,
  capped at 50 prompts. Checks trace *shape*, never builds a corpus.

Hardware rules the loader enforces, because each costs a 20-minute model load
to discover the hard way:

- **NVFP4 needs Blackwell.** Ada cards (4090, RTX 6000 Ada, L40S) cannot load
  those builds. Use FP8.
- **FP8 needs Ada or Hopper.** An RTX A6000 at $0.33/hr looks like the bargain
  but is Ampere — it has no FP8 path and would force int4.
- **Don't run two teachers at once.** Two 30B models don't co-resident in
  48GB at FP8, and dropping both to int4 to fit degrades every trace the
  student learns from. Teacher quality is the ceiling on everything
  downstream; it is the one loss you cannot recover later.

## Teachers

| Teacher | Arch | License | Teach it for |
|---|---|---|---|
| `muse-glimmer` | dense 30B | Apache 2.0 ✅ | long-context docs, multimodal, tool schemas |
| `nemotron` | MoE 30B-A3B | NVIDIA OML ⚠️ verify | complex reasoning, structured problems, tool calls |

`A3B` means ~3B *active* parameters — throughput, not memory. All experts stay
resident, so budget the same ~32GB at FP8 as a dense 30B.

**Do not use either teacher to teach Indonesian.** That capability is
Tantular's own and already paid for; distilling over it loses ground. Import
reasoning and agentic behaviour, keep the voice.

⚠️ Nemotron's license is the one open legal question. Confirm its
synthetic-data terms permit a commercially-shipped derivative *before*
generating a corpus from it. Muse Glimmer's Apache 2.0 is settled.

## Run

```bash
pip install -r requirements.txt

# 1. serve one teacher (GPU host)
./scripts/serve_teacher.sh muse-glimmer ai19
./scripts/serve_teacher.sh muse-glimmer rented-48gb   # identical downstream

# 2. generate traces (anywhere)
python3 src/generate.py --teacher muse-glimmer --host ai19 \
    --prompts prompts/office_edit.jsonl \
    --out data/raw/muse.office_edit.jsonl

# 3. judge / dedup / promote — reuse the existing pipeline
#    ../tantular/finetune/{judge,dedup,review_promote}.py

# 4. train
#    train/qlora_9b.yaml
```

Swap host by changing one argument. `bridge_client.py` only ever sees a URL,
so it cannot tell ai19 from a rented card from a gateway.

## Sequencing

Today's `tantular-office:0.4-9b` is a **runtime profile**, not a fine-tune —
`Modelfile.office-9b` is `FROM qwen3.5:9b` plus a SYSTEM prompt. The only
genuinely trained Tantular is `tantular:0.2-id-3b-lora` (Qwen2.5-3B base,
153 Indonesian support examples), a different base family and domain.

So there is **no Office checkpoint to continue from**. This first run starts
from Qwen3.5 base and produces v1 — the real checkpoint. Teacher distillation
onto an existing Tantular becomes v2, once v1 exists.

```
now:   Qwen3.5 9B + system prompt        (no weights trained)
v1:    Qwen3.5 9B + Office SFT           (this repo's first run)
v2:    v1 + teacher distillation         (Muse / Nemotron)
v3:    v2 → distilled into 4B            (9B becomes the teacher)
```

The 4B is trained last, from the improved 9B rather than from the 30B
teachers directly — it learns from something that already shares its target
behaviour and voice.

## Data handling

Hosts declare where data goes; prompts declare what they carry.

| Host | `data_egress` | Real Office material |
|---|---|---|
| `ai19` | internal | ✅ on-premises (training only) |
| `gateway` | operator_visible | ❌ synthetic only — operator sees and may retain prompts |
| `mac-validate` | internal | ✅ nothing leaves the machine |
| `rented-48gb` | **external** | ❌ needs explicit `--egress-approval` |

Prompts carry `source_class`. **Unclassified defaults to `internal`**, so
forgetting to label real corpus material fails closed rather than shipping it
to a rented GPU. Only `"source_class": "synthetic"` travels freely.

## Source inventory

Coverage is not a number to satisfy. It is a claim that each stratum is backed
by real, approved, redacted source material — inventing documents to turn
12/26 into 26/26 produces a model that scores well on a corpus describing a
company that does not exist.

```bash
python3 src/inventory.py refresh                  # sync rows with the manifest
python3 src/inventory.py status data/raw/*.jsonl  # what is blocked, on what
```

`inventory/sources.yaml` records per kind: the source, who approved it, whether
it is redacted, and whether egress is approved. Kinds without that are BLOCKED,
and **"blocked on data coverage" is the correct state to report** — not
synthetic coverage presented as production data.

Current: 12 kinds have traces (synthetic, from the canary), **14 blocked** on
approved source material.

## Corpus status

**int4-only, under a signed waiver** (`calibration/INT4_WAIVER.md`, accepted
2026-08-13). The FP8 gate is UNMET, not passed — `verify_corpus.py --gate`
still fails and still exits 1. No FP8 claim may be made.

| Path | Contents |
|---|---|
| `data/raw/` | authoritative corpus — **empty**, awaiting approved sources |
| `data/validation/` | pipeline-validation artifacts, never corpus |
| `data/crossover/control/` | 3 int4 replicates, control noise floor |
| `data/calibration/` | deployment baseline + parity validation |

Remaining sequence:

1. Approve and redact sources for the **14 blocked strata** (`src/inventory.py status`)
2. Write seeds only for approved kinds
3. Generate through ai19 under the waiver, **with replicates for the 3 volatile families**
4. ~~Exclude the gateway canaries~~ — done, preserved in `data/validation/`
5. **Rotate the gateway key** before production use

## Gates

Every run is gated on regression evals (`train/qlora_9b.yaml`). Distillation
reliably buys reasoning while quietly costing Indonesian voice and JSON
contract adherence — and Studio breaks outright when the edit contract
drifts. Run evals before and after; don't promote an adapter that regresses
either.
