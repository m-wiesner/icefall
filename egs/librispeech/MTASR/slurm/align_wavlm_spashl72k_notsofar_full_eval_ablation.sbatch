#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-lfeval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-3%3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
set_name=notsofar_eval_sc_plaza_0_full_wavlm_spashl72k
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
collar=32000
margin=1.0

single_state=true
blockwise=false
blank_args=()
case "$SLURM_ARRAY_TASK_ID" in
  1)
    label=02_1state_allframes_fullgraph
    ;;
  2)
    label=03_1state_corpus30_fullgraph
    blank_args=(
      --blank-frame-filter-threshold 0.9991292357444763
      --blank-frame-filter-min-token-ratio 2
    )
    ;;
  3)
    label=04_1state_corpus30_graph20_audio30
    blockwise=true
    blank_args=(
      --blank-frame-filter-threshold 0.9991292357444763
      --blank-frame-filter-min-token-ratio 2
    )
    ;;
  *)
    echo "Unexpected array task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

suffix=longform_eval_time_mask1_${label}_20260923

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02 --max-num-spks 8 \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo "$single_state" \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin "$margin" \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment "$blockwise" \
  --blockwise-whole-recording-inference false \
  --blockwise-graph-chunk-duration 20 \
  --blockwise-token-time-mask true \
  --blockwise-speaker-ordering training \
  "${blank_args[@]}" \
  --alignment-diagnostics true --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

python local/score_alignment_mean_median.py \
  --relabel --hyp "$word_ali" --ref "$reference" \
  --output "${word_ali%.json}_uncalibrated_mean_median.json"

dev_hyp="$exp_dir/align/word_alignments_chkpt${iter}_avg${avg}_notsofar_dev_sc_plaza_0_full_wavlm_spashl72k_collar${collar}_longform_time_mask1_${label}_20260922.json"
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
python local/score_alignment_mean_median.py \
  --relabel --hyp "$word_ali" --ref "$reference" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --output "${word_ali%.json}_dev_calibrated_mean_median.json"

echo "LABEL=$label"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
echo "CALIBRATED_METRICS_JSON=${word_ali%.json}_dev_calibrated_mean_median.json"
