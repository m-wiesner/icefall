#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-sfabl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=1-5%5
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned}
iter=72000
avg=1
collar=32000
margin=1.0

synth_set=synth3_spkattr_wavlm_spashl72k_sfabl
synth_manifest=data/manifests/cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz
synth_ref=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
dev_set=notsofar_dev30_sc_plaza_0_spkattr_wavlm_spashl72k_sfabl
dev_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_set=notsofar_eval30_sc_plaza_0_spkattr_wavlm_spashl72k_sfabl
eval_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json

single_state=false
blockwise=false
use_blank=false
bpe_max_piece_characters=0
case "$SLURM_ARRAY_TASK_ID" in
  1) label=01_training_allframes_fullgraph ;;
  2) label=02_1state_allframes_fullgraph; single_state=true ;;
  3) label=03_1state_corpus30_fullgraph; single_state=true; use_blank=true ;;
  4) label=04_1state_corpus30_graph10; single_state=true; use_blank=true; blockwise=true ;;
  5) label=05_1state_allframes_graph10_bpechars4; single_state=true; blockwise=true; bpe_max_piece_characters=4 ;;
  *) echo "Unexpected array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
suffix=shortform_time_mask1_${label}_20260923

common_args=(
  --iter "$iter" --avg "$avg" --use-averaged-model false
  --exp-dir "$exp_dir"
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en
  --suffix "$suffix"
  --use-large true --downsample false --use-layer-norm false
  --frame-duration 0.02 --max-num-spks 8
  --factorized-token-speaker-output true --factorized-alignment false
  --ignore-speaker false --oracle-num-spks 0
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0
  --collar "$collar" --total-max-overlaps 10000000
  --allow-self-overlap false
  --original-topo false --single-state-ctc-topo "$single_state"
  --intersection-method factorized_viterbi
  --segment-time-window-margin "$margin"
  --alignment-bpe-max-piece-characters "$bpe_max_piece_characters"
  --transcript-specific-dense-remap false
  --alignment-chunk-duration 0 --alignment-chunk-context 0
  --alignment-chunk-overlap 0
  --token-speaker-mask-window-duration 0
  --blockwise-alignment "$blockwise"
  --blockwise-whole-recording-inference "$blockwise"
  --blockwise-graph-chunk-duration 10
  --blockwise-token-time-mask true
  --blockwise-speaker-ordering training
  --alignment-diagnostics true --max-states 25000000
  --skip-empty-alignments false --max-duration 200 --max-cuts 1
  --num-buckets 1 --num-workers 4 --normalize-loudness false
)

run_alignment() {
  local sets=$1
  shift
  local blank_args=()
  if [[ "$use_blank" == true ]]; then
    local calibration_set=$1
    local threshold_output=$2
    shift 2
    blank_args=(
      --blank-frame-filter-target-average-fraction 0.30
      --blank-frame-filter-calibration-set "$calibration_set"
      --blank-frame-filter-threshold-output "$threshold_output"
      --blank-frame-filter-min-token-ratio 2
    )
  fi
  python wavlm_spashl_ctc/align.py --method token -- \
    "${common_args[@]}" --test-sets "$sets" "$@" "${blank_args[@]}"
}

synth_threshold="$exp_dir/align/blank_threshold_${synth_set}_${label}.json"
nsf_threshold="$exp_dir/align/blank_threshold_${dev_set}_${label}.json"
run_synth=${RUN_SYNTH:-true}
if [[ "$use_blank" == true ]]; then
  if [[ "$run_synth" == true ]]; then
    run_alignment "$synth_set" "$synth_set" "$synth_threshold" \
      --test-cut-manifest "$synth_set=$synth_manifest"
  fi
  run_alignment "$dev_set $eval_set" "$dev_set" "$nsf_threshold" \
    --test-cut-manifest "$dev_set=$dev_manifest" \
    --test-cut-manifest "$eval_set=$eval_manifest"
else
  if [[ "$run_synth" == true ]]; then
    run_alignment "$synth_set" \
      --test-cut-manifest "$synth_set=$synth_manifest"
  fi
  run_alignment "$dev_set $eval_set" \
    --test-cut-manifest "$dev_set=$dev_manifest" \
    --test-cut-manifest "$eval_set=$eval_manifest"
fi

word_alignment() {
  local set_name=$1
  local ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
  python local/ali_to_word_ali.py "$ali"
}

if [[ "$run_synth" == true ]]; then
  word_alignment "$synth_set"
fi
word_alignment "$dev_set"
word_alignment "$eval_set"
synth_ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${synth_set}_collar${collar}_${suffix}.json"
dev_ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${dev_set}_collar${collar}_${suffix}.json"
eval_ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${eval_set}_collar${collar}_${suffix}.json"
synth_hyp="$(dirname "$synth_ali")/word_$(basename "$synth_ali")"
dev_hyp="$(dirname "$dev_ali")/word_$(basename "$dev_ali")"
eval_hyp="$(dirname "$eval_ali")/word_$(basename "$eval_ali")"

if [[ "$run_synth" == true ]]; then
  python local/score_alignment_mean_median.py --relabel \
    --hyp "$synth_hyp" --ref "$synth_ref" \
    --output "${synth_hyp%.json}_uncalibrated_mean_median.json"
  python local/score_alignment_mean_median.py --relabel --calibrate-half \
    --hyp "$synth_hyp" --ref "$synth_ref" \
    --output "${synth_hyp%.json}_firsthalf_calibrated_secondhalf_mean_median.json"
fi

offsets="$exp_dir/align/calibration_${dev_set}_${label}.json"
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

echo "LABEL=$label"
echo "SYNTH_HYP=$synth_hyp"
echo "DEV_HYP=$dev_hyp"
echo "EVAL_HYP=$eval_hyp"
