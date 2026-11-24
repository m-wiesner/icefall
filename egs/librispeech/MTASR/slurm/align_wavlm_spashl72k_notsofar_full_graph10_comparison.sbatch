#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-lfg10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-2%2
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
collar=32000
margin=1.0

dev_set=notsofar_dev_sc_plaza_0_full_wavlm_spashl72k
eval_set=notsofar_eval_sc_plaza_0_full_wavlm_spashl72k
dev_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
eval_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json

blank_args=()
bpe_max_piece_characters=0
case "$SLURM_ARRAY_TASK_ID" in
  1)
    label=05_1state_corpus30_graph10_audio30
    blank_args=(
      --blank-frame-filter-target-average-fraction 0.30
      --blank-frame-filter-calibration-set "$dev_set"
      --blank-frame-filter-threshold-output "$exp_dir/align/blank_threshold_${dev_set}_decoupled_${label}.json"
      --blank-frame-filter-min-token-ratio 2
    )
    ;;
  2)
    label=06_1state_allframes_graph10_audio30_bpechars4
    bpe_max_piece_characters=4
    ;;
  *)
    echo "Unexpected array task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

suffix=longform_time_mask1_decoupled_${label}_20260924

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$dev_set $eval_set" \
  --test-cut-manifest "$dev_set=$dev_manifest" \
  --test-cut-manifest "$eval_set=$eval_manifest" \
  --suffix "$suffix" \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02 --max-num-spks 8 \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin "$margin" \
  --alignment-bpe-max-piece-characters "$bpe_max_piece_characters" \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment true \
  --blockwise-whole-recording-inference false \
  --blockwise-graph-chunk-duration 10 \
  --blockwise-token-time-mask true \
  --blockwise-speaker-ordering training \
  "${blank_args[@]}" \
  --alignment-diagnostics true --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --normalize-loudness false

word_alignment() {
  local set_name=$1
  local ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
  python local/ali_to_word_ali.py "$ali"
}

word_alignment "$dev_set"
word_alignment "$eval_set"

dev_ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${dev_set}_collar${collar}_${suffix}.json"
eval_ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${eval_set}_collar${collar}_${suffix}.json"
dev_hyp="$(dirname "$dev_ali")/word_$(basename "$dev_ali")"
eval_hyp="$(dirname "$eval_ali")/word_$(basename "$eval_ali")"

python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --output "${dev_hyp%.json}_uncalibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --output "${dev_hyp%.json}_full_dev_calibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --output "${eval_hyp%.json}_uncalibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --output "${eval_hyp%.json}_dev_calibrated_mean_median.json"

echo "LABEL=$label"
echo "DEV_HYP=$dev_hyp"
echo "EVAL_HYP=$eval_hyp"
