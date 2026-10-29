#!/bin/bash
# run_l1_data.sh (Claude, 2026-10-29): batch L1 (DESIGN_LLM_LATENT_SHEAF_GBP.md section 2; D8-82) data preparation on the
# Linux GPU box, CPU only, no GPU, no training. Not run on Windows.
#   1. If hotpot_train_v1.1.json / hotpot_dev_distractor_v1.json are missing from $DATA_DIR: curl them from the official
#      host http://curtis.ml.cmu.edu/datasets/hotpot (to <name>.part, renamed only when complete). If either file is
#      still missing or empty afterwards (the official host did not resolve from uk-linux on 2026-10-29), fall back to
#      the Hugging Face mirror hotpotqa/hotpot_qa (config distractor) through l1_hotpot_from_hf.py --out $DATA_DIR,
#      with HF_HUB_CACHE / HF_DATASETS_CACHE under $HF_CACHE_ROOT/{hub,datasets}; it also writes $DATA_DIR/hf_source.json.
#   2. Print sha256 and size of both files and which source was used (existing / official / hf-mirror).
#   3. Run l1_data.py with the pre-registered sizes (its defaults: 20,000 / 1,000 / 1,000, seed 342101) into
#      $DATA_DIR/splits_l1. Never overwrites: if $DATA_DIR/splits_l1/splits_meta.json exists, print it and exit 0.
# Writes only under $DATA_DIR and $HF_CACHE_ROOT (both on /mnt/datasets). Data never enters the repo.
# Usage (uk-linux): [PYTHON=/path/to/python-with-datasets] bash scripts/run_l1_data.sh
#   PYTHON needs the `datasets` package only if the mirror fallback is taken. L1_DATA_EXTRA_ARGS is for the unit test
#   (smaller sizes on a synthetic fixture) and must stay empty for the real run.
set -euo pipefail
DATA_DIR="${DATA_DIR:-/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/hotpot}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache}"
BASE_URL="http://curtis.ml.cmu.edu/datasets/hotpot"
PYTHON="${PYTHON:-python3}"
CURL="${CURL:-curl}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CONVERTER="${HF_CONVERTER:-$HERE/l1_hotpot_from_hf.py}"
EXTRA_ARGS="${L1_DATA_EXTRA_ARGS:-}"
FILES="hotpot_train_v1.1.json hotpot_dev_distractor_v1.json"
OUT="$DATA_DIR/splits_l1"
log(){ echo "$(date -u +%FT%TZ) $*"; }
sha(){ sha256sum "$1" | cut -d' ' -f1; }

mkdir -p "$DATA_DIR"
log "run_l1_data started pid $$ host $(hostname) DATA_DIR=$DATA_DIR l1_data.py sha256 $(sha "$HERE/l1_data.py")"
SOURCE=existing
for f in $FILES; do
  if [ -s "$DATA_DIR/$f" ]; then
    log "$f present"
    continue
  fi
  log "downloading $f from $BASE_URL"
  rm -f "$DATA_DIR/$f.part"
  if "$CURL" -fL --retry 2 --connect-timeout 30 -o "$DATA_DIR/$f.part" "$BASE_URL/$f" && [ -s "$DATA_DIR/$f.part" ]; then
    mv "$DATA_DIR/$f.part" "$DATA_DIR/$f"
    SOURCE=official
  else
    log "official download of $f failed"
    rm -f "$DATA_DIR/$f.part"
  fi
done
MISSING=0
for f in $FILES; do
  [ -s "$DATA_DIR/$f" ] || MISSING=1
done
if [ "$MISSING" = 1 ]; then
  log "falling back to the Hugging Face mirror hotpotqa/hotpot_qa (distractor) via $HF_CONVERTER"
  mkdir -p "$HF_CACHE_ROOT/hub" "$HF_CACHE_ROOT/datasets"
  HF_HUB_CACHE="$HF_CACHE_ROOT/hub" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets" "$PYTHON" -B "$HF_CONVERTER" --out "$DATA_DIR"
  SOURCE=hf-mirror
fi
for f in $FILES; do
  if [ ! -s "$DATA_DIR/$f" ]; then
    log "ERROR: $DATA_DIR/$f missing after the official download and the mirror fallback"
    exit 1
  fi
  log "$f sha256 $(sha "$DATA_DIR/$f") bytes $(stat -c %s "$DATA_DIR/$f")"
done
log "source: $SOURCE"
if [ -s "$DATA_DIR/hf_source.json" ]; then
  log "hf_source.json: $(tr -d '\n' < "$DATA_DIR/hf_source.json")"
fi
if [ -s "$OUT/splits_meta.json" ]; then
  log "splits already exist at $OUT (not overwritten)"
  cat "$OUT/splits_meta.json"
  exit 0
fi
mkdir -p "$OUT"
log "running l1_data.py -> $OUT"
"$PYTHON" -B "$HERE/l1_data.py" --train-json "$DATA_DIR/hotpot_train_v1.1.json" --dev-json "$DATA_DIR/hotpot_dev_distractor_v1.json" --out "$OUT" $EXTRA_ARGS
log "done: $OUT"
