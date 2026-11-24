#!/usr/bin/env bash
#SBATCH --job-name=viterbi-stack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
iter=200000
avg=5
collar=32000
set_name=notsofar_sdm_dev30_word_aligned
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized_word_aligned.jsonl.gz
cut_id=nsf-dev-MTG_30503-sc_plaza_0-g0000-w0004

run_case() {
  local stage=$1
  local single_state=$2
  local keep_fraction=$3
  local compact_remap=$4
  local chunk_duration=$5
  local mask_window=$6

  local blank_args=()
  if [[ -n "$keep_fraction" ]]; then
    blank_args=(
      --blank-frame-filter-keep-fraction "$keep_fraction"
      --blank-frame-filter-min-token-ratio 2.0
    )
  fi

  echo "BEGIN_ABLATION_STAGE $stage"
  python parakeet_native_spashl_ctc/align.py --method token -- \
    --iter "$iter" --avg "$avg" --use-averaged-model false \
    --exp-dir "$exp_dir" \
    --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
    --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
    --test-cut-id "$cut_id" --suffix "viterbi_stack_${stage}" \
    --factorized-token-speaker-output true --factorized-alignment false \
    --ignore-speaker false --oracle-num-spks 0 \
    --max-num-spks 8 --sort-strategy speech_time \
    --speaker-weight 1.0 --blank-weight 0.0 \
    --collar "$collar" --total-max-overlaps 10000000 \
    --original-topo false --single-state-ctc-topo "$single_state" \
    --alignment-chunk-duration "$chunk_duration" \
    --alignment-chunk-overlap 5 \
    --transcript-specific-dense-remap "$compact_remap" \
    --token-speaker-mask-window-duration "$mask_window" \
    --token-speaker-mask-boundary-collar 2 \
    --intersection-method factorized_viterbi \
    --alignment-diagnostics true \
    "${blank_args[@]}" \
    --skip-empty-alignments true \
    --max-duration 2000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
    --downsample false --use-layer-norm false --frame-duration 0.02 \
    --normalize-loudness false
  echo "END_ABLATION_STAGE $stage"
}

# Add one memory optimization per stage. Topology reduction deliberately
# precedes frame removal, matching the previous dense-intersection ablation.
run_case 00_baseline_training_topo_allframes false "" false 0 0
run_case 01_single_state_allframes true "" false 0 0
run_case 02_single_state_keep030 true 0.30 false 0 0
run_case 03_single_state_keep030_compact true 0.30 true 0 0
run_case 04_single_state_keep030_compact_chunk25 true 0.30 true 25 0
run_case 05_single_state_keep030_compact_chunk25_mask2 true 0.30 true 25 2
