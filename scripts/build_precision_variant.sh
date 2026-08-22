#!/usr/bin/env bash
# Rebuild the shipped Tantular profile at a different precision, changing NOTHING else.
#
#   ./scripts/build_precision_variant.sh qwen3.5:9b-q8_0 tantular-greedy-q8:0.4-9b
#
# Experiment 2 asks whether Q4_K_M quantization explains why the shipped profile
# scores below the bf16 base. That question is only answerable if precision is
# the ONLY difference, so this derives the new Modelfile FROM THE SHIPPED ONE
# rather than retyping it: the SYSTEM prompt, RENDERER and PARSER are copied
# verbatim, and only the FROM line and the sampling parameters are replaced.
#
# Sampling is pinned to the same greedy settings used for the Q4 arm
# (experiment 1), so the two arms differ in precision alone:
#   temperature 0, top_k 0, top_p 1.0, presence_penalty 0, repeat_penalty 1.0
#
# Retyping the system prompt would risk a whitespace difference nobody would
# notice, and the whole comparison rests on it being identical.
set -euo pipefail

SOURCE_MODEL="${SOURCE_MODEL:-tantular-office:0.4-9b}"
BASE="${1:?usage: build_precision_variant.sh <base-tag> <new-tag>}"
NEW="${2:?usage: build_precision_variant.sh <base-tag> <new-tag>}"

# Capture first, then match. Under `set -o pipefail`, `ollama list | grep -q`
# fails even when the tag IS present: grep -q exits on the first match, ollama
# takes SIGPIPE and exits non-zero, and pipefail propagates that. The guard then
# rejects a model that is sitting right there.
INSTALLED="$(ollama list)"
grep -q "^${BASE}" <<<"$INSTALLED" || { echo "base tag not pulled: $BASE" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Everything except FROM, PARAMETER and LICENSE — i.e. SYSTEM, TEMPLATE,
# RENDERER, PARSER — copied byte for byte.
ollama show --modelfile "$SOURCE_MODEL" \
  | awk '/^LICENSE/{exit} !/^FROM /&&!/^PARAMETER /&&!/^#/' > "$WORK/body"

{
  echo "FROM $BASE"
  cat "$WORK/body"
  echo "PARAMETER temperature 0"
  echo "PARAMETER top_k 0"
  echo "PARAMETER top_p 1.0"
  echo "PARAMETER presence_penalty 0"
  echo "PARAMETER repeat_penalty 1.0"
  echo "PARAMETER num_ctx 32768"
  echo "PARAMETER num_predict 8192"
} > "$WORK/Modelfile"

echo "=== derived Modelfile (non-LICENSE lines) ==="
grep -E "^(FROM|PARAMETER|RENDERER|PARSER|TEMPLATE)" "$WORK/Modelfile"
echo "system prompt bytes: $(grep -c . "$WORK/body") lines"

ollama create "$NEW" -f "$WORK/Modelfile"

echo
echo "=== identity record ==="
for tag in "$SOURCE_MODEL" "$NEW"; do
  printf '%s\n' "$tag"
  ollama show "$tag" 2>/dev/null | grep -iE "parameters|quantization|architecture" | sed 's/^/    /'
done
