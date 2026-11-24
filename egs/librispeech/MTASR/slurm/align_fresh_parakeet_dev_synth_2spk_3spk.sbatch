#!/usr/bin/env bash
#SBATCH --job-name=par-synth-fresh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=2-3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

num_speakers=${SLURM_ARRAY_TASK_ID}
set_name=synth${num_speakers}
exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
suffix=fresh_baseline_v1_training_topo_allframes_dense

python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter 200000 --avg 5 --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --ignore-speaker false \
  --max-num-spks 8 \
  --oracle-num-spks "$num_speakers" \
  --sort-strategy speech_time \
  --speaker-weight 1.0 \
  --blank-weight 0.0 \
  --collar 32000 \
  --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false \
  --single-state-ctc-topo false \
  --intersection-method dense \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --alignment-diagnostics true \
  --beam-size 40 \
  --max-states 25000000 \
  --skip-empty-alignments false \
  --max-duration 400 \
  --max-cuts 1 \
  --num-buckets 1 \
  --num-workers 4 \
  --downsample false \
  --use-layer-norm false \
  --frame-duration 0.02 \
  --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt200000_avg5_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$(dirname "$ali")/word_$(basename "$ali")"
