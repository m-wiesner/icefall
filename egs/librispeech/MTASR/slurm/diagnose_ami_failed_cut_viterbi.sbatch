#!/usr/bin/env bash
#SBATCH --job-name=par-ami-cut-diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

manifest=${MANIFEST:?MANIFEST is required}
set_name=${SET_NAME:?SET_NAME is required}
cut_id=${CUT_ID:?CUT_ID is required}
variant=${VARIANT:?VARIANT is required}

case "$variant" in
  nomask) mask_window=0; mask_collar=5; keep_fraction=0.3 ;;
  mask10) mask_window=2; mask_collar=10; keep_fraction=0.3 ;;
  keep50) mask_window=2; mask_collar=5; keep_fraction=0.5 ;;
  *) echo "Unknown VARIANT=$variant" >&2; exit 2 ;;
esac

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter 200000 --avg 5 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --test-cut-id "$cut_id" --suffix "ami_failed_${cut_id}_${variant}" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar 32000 --total-max-overlaps 10000000 \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-keep-fraction "$keep_fraction" \
  --blank-frame-filter-min-token-ratio 2.0 \
  --transcript-specific-dense-remap true \
  --token-speaker-mask-window-duration "$mask_window" \
  --token-speaker-mask-boundary-collar "$mask_collar" \
  --alignment-chunk-duration 290 --alignment-chunk-overlap 10 \
  --alignment-chunk-context 0 --alignment-diagnostics true \
  --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments true --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false
