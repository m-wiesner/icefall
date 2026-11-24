#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-nsf-bprm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Percentages here mean frames removed.  The corresponding corpus-level
# target mean fractions retained are 100%, 80%, 60%, and 40%.
case "$SLURM_ARRAY_TASK_ID" in
  0) removal_percent=0;  keep_fraction=1.00 ;;
  1) removal_percent=20; keep_fraction=0.80 ;;
  2) removal_percent=40; keep_fraction=0.60 ;;
  3) removal_percent=60; keep_fraction=0.40 ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
collar=32000
label="center20_graph20_remove${removal_percent}_1state_spkattr_bpechars4"
suffix="longform_${label}_20260923"

dev_set=notsofar_dev_sc_plaza_0_full_wavlm_spashl72k_bpechars4
eval_set=notsofar_eval_sc_plaza_0_full_wavlm_spashl72k_bpechars4
dev_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
eval_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json

threshold_json="$exp_dir/align/blank_threshold_notsofar_${label}.json"

common_args=(
  --iter "$iter" --avg "$avg" --use-averaged-model false
  --exp-dir "$exp_dir"
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en
  --suffix "$suffix"
  --use-large true --downsample false --use-layer-norm false
  --frame-duration 0.02 --max-num-spks 8
  --factorized-token-speaker-output true --factorized-alignment false
  --ignore-speaker false --oracle-num-spks 0
  --blockwise-speaker-recovery oracle_map
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0
  --collar "$collar" --total-max-overlaps 10000000
  --allow-self-overlap false
  --original-topo false --single-state-ctc-topo true
  --intersection-method factorized_viterbi
  --segment-time-window-margin 1.0
  --alignment-bpe-max-piece-characters 4
  --single-token-duration-per-character 0.08
  --transcript-specific-dense-remap true
  --alignment-chunk-duration 30 --alignment-chunk-context 5
  --alignment-chunk-overlap 0
  --token-speaker-mask-window-duration 0
  --blockwise-alignment true
  --blockwise-whole-recording-inference false
  --blockwise-graph-chunk-duration 20
  --blockwise-token-time-mask true
  --blockwise-speaker-ordering training
  --alignment-diagnostics true --max-states 25000000
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1
  --num-buckets 1 --num-workers 4 --normalize-loudness false
)

if [[ "$removal_percent" == 0 ]]; then
  # Omit all blank-filter options so every encoder frame is retained.
  python wavlm_spashl_ctc/align.py --method token -- \
    "${common_args[@]}" \
    --test-sets "$dev_set" --test-cut-manifest "$dev_set=$dev_manifest"
  python wavlm_spashl_ctc/align.py --method token -- \
    "${common_args[@]}" \
    --test-sets "$eval_set" --test-cut-manifest "$eval_set=$eval_manifest"
else
  # Estimate one threshold over full DEV recordings and reuse it on EVAL.
  python wavlm_spashl_ctc/align.py --method token -- \
    "${common_args[@]}" \
    --test-sets "$dev_set" --test-cut-manifest "$dev_set=$dev_manifest" \
    --blank-frame-filter-target-average-fraction "$keep_fraction" \
    --blank-frame-filter-calibration-set "$dev_set" \
    --blank-frame-filter-threshold-output "$threshold_json" \
    --blank-frame-filter-min-token-ratio 2

  blank_threshold=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["blank_probability_threshold"])' "$threshold_json")
  python wavlm_spashl_ctc/align.py --method token -- \
    "${common_args[@]}" \
    --test-sets "$eval_set" --test-cut-manifest "$eval_set=$eval_manifest" \
    --blank-frame-filter-threshold "$blank_threshold" \
    --blank-frame-filter-min-token-ratio 2
fi

alignment_path() {
  local set_name=$1
  echo "$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
}

dev_ali=$(alignment_path "$dev_set")
eval_ali=$(alignment_path "$eval_set")
python local/ali_to_word_ali.py "$dev_ali"
python local/ali_to_word_ali.py "$eval_ali"
dev_hyp="$(dirname "$dev_ali")/word_$(basename "$dev_ali")"
eval_hyp="$(dirname "$eval_ali")/word_$(basename "$eval_ali")"
offsets="$exp_dir/align/calibration_notsofar_${label}.json"

python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --output "${dev_hyp%.json}_uncalibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-output "$offsets" \
  --output "${dev_hyp%.json}_full_dev_calibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --output "${eval_hyp%.json}_uncalibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-offsets "$offsets" \
  --output "${eval_hyp%.json}_dev_calibrated_mean_median.json"

echo "BLANK_REMOVAL_PERCENT=$removal_percent"
echo "TARGET_MEAN_KEEP_FRACTION=$keep_fraction"
if [[ "$removal_percent" != 0 ]]; then
  echo "BLANK_THRESHOLD_JSON=$threshold_json"
fi
echo "DEV_WORD_ALIGNMENT=$dev_hyp"
echo "EVAL_WORD_ALIGNMENT=$eval_hyp"
