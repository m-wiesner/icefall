#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-ami-bp40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
collar=32000
label=center20_graph20_remove40_1state_spkattr_bpechars4
suffix=longform_${label}_20260923

dev_set=ami_sdm_dev_full_wavlm_spashl72k_bpechars4
eval_set=ami_sdm_test_full_wavlm_spashl72k_bpechars4
dev_manifest=data/manifests/real_meetings/ami/ami_sdm_dev_full_normalized_word_aligned.jsonl.gz
eval_manifest=data/manifests/real_meetings/ami/ami_sdm_test_full_normalized_word_aligned.jsonl.gz
dev_ref=data/alignments/ami_sdm_dev_full_normalized_word_alignments.json
eval_ref=data/alignments/ami_sdm_test_full_normalized_word_alignments.json
threshold_json="$exp_dir/align/blank_threshold_ami_${label}.json"

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

# DEV estimates the threshold producing 60% mean retained frames (40%
# removal).  The same threshold is applied to TEST.
python wavlm_spashl_ctc/align.py --method token -- \
  "${common_args[@]}" \
  --test-sets "$dev_set" --test-cut-manifest "$dev_set=$dev_manifest" \
  --blank-frame-filter-target-average-fraction 0.60 \
  --blank-frame-filter-calibration-set "$dev_set" \
  --blank-frame-filter-threshold-output "$threshold_json" \
  --blank-frame-filter-min-token-ratio 2

blank_threshold=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["blank_probability_threshold"])' "$threshold_json")
python wavlm_spashl_ctc/align.py --method token -- \
  "${common_args[@]}" \
  --test-sets "$eval_set" --test-cut-manifest "$eval_set=$eval_manifest" \
  --blank-frame-filter-threshold "$blank_threshold" \
  --blank-frame-filter-min-token-ratio 2

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
offsets="$exp_dir/align/calibration_ami_${label}.json"

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

for split in dev eval; do
  if [[ "$split" == dev ]]; then
    hyp=$dev_hyp; ref=$dev_ref; manifest=$dev_manifest
  else
    hyp=$eval_hyp; ref=$eval_ref; manifest=$eval_manifest
  fi
  speaker_metrics="${hyp%.json}_speaker_metrics.txt"
  {
    echo "UNCALIBRATED_SUPERVISION_BOUNDARY_DER"
    python local/score_alignment_der.py --relabel --collar 0.25 \
      --hyp "$hyp" --ref "$ref" --ref-cut-manifest "$manifest"
    echo "DEV_CALIBRATED_SUPERVISION_BOUNDARY_DER"
    python local/score_alignment_der.py --relabel --collar 0.25 \
      --hyp "$hyp" --ref "$ref" --ref-cut-manifest "$manifest" \
      --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
      --calibration-statistic median
  } | tee "$speaker_metrics"
done

echo "CORPUS=ami"
echo "BLANK_REMOVAL_PERCENT=40"
echo "TARGET_MEAN_KEEP_FRACTION=0.60"
echo "BLANK_THRESHOLD_JSON=$threshold_json"
echo "DEV_WORD_ALIGNMENT=$dev_hyp"
echo "EVAL_WORD_ALIGNMENT=$eval_hyp"
