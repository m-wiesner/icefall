#!/usr/bin/env bash
#SBATCH --job-name=mtasr-decode-synth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh
export HF_HOME=${HF_HOME:-$recipe_dir/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${MODEL_KIND:?Set MODEL_KIND to wavlm or parakeet}"
: "${EXP_DIR:?Set EXP_DIR}"
: "${ITER:?Set ITER to the selected checkpoint update}"

suffix=${SUFFIX:-argmax_latest}
max_duration=${MAX_DURATION:-80}
avg=${AVG:-1}

common_args=(
  --iter "$ITER"
  --avg "$avg"
  --use-averaged-model false
  --exp-dir "$EXP_DIR"
  --lang-dir data/lang_bpe_5000
  --test-sets synth
  --suffix "$suffix"
  --max-num-spks 4
  --factorized-token-speaker-output true
  --speaker-weight 1.0
  --sort-strategy speech_time
  --max-duration "$max_duration"
  --num-workers 2
)

if [[ "$MODEL_KIND" == wavlm ]]; then
  python wavlm_spashl_ctc/decode.py --method greedy -- \
    "${common_args[@]}" \
    --use-large true \
    --downsample false \
    --use-layer-norm false
elif [[ "$MODEL_KIND" == parakeet ]]; then
  speaker_layer=${SPEAKER_LAYER:-18}
  convolution_norm=${PARAKEET_CONVOLUTION_NORM:-masked_batch_norm}
  python parakeet_spashl_ctc/decode.py --method greedy \
    --speaker-layer "$speaker_layer" \
    --parakeet-convolution-norm "$convolution_norm" -- \
    "${common_args[@]}" \
    --downsample false \
    --use-layer-norm false
else
  echo "Unsupported MODEL_KIND=$MODEL_KIND" >&2
  exit 2
fi

hyp="$EXP_DIR/decode/hyps_chkpt${ITER}_avg${avg}_synth_${suffix}.stm"
ref="$EXP_DIR/decode/refs_chkpt${ITER}_avg${avg}_synth_${suffix}.stm"
score="$EXP_DIR/decode/hyps_chkpt${ITER}_avg${avg}_synth_${suffix}_tcpwer.json"
per_reco="$EXP_DIR/decode/hyps_chkpt${ITER}_avg${avg}_synth_${suffix}_tcpwer_per_reco.json"

python -m meeteval.wer tcpwer \
  --reference "$ref" \
  --hypothesis "$hyp" \
  --collar 5 \
  --average-out "$score" \
  --per-reco-out "$per_reco"

python - "$score" <<'PY'
import json
import sys

path = sys.argv[1]
score = json.load(open(path, encoding="utf-8"))
print(json.dumps({"score_file": path, **score}, indent=2))
PY
