#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-tok40
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

corpus=${CORPUS:?Set CORPUS=notsofar or ami}
split=${SPLIT:?Set SPLIT=dev or test}
attribution_unit=${ATTRIBUTION_UNIT:-segment}
if [[ "$attribution_unit" != segment && "$attribution_unit" != word ]]; then
  echo "Unexpected ATTRIBUTION_UNIT=$attribution_unit" >&2
  exit 2
fi

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
collar=32000
label=center20_graph20_remove40_1state_tokenonly_oraclemap_${attribution_unit}_bpechars4
suffix=longform_${label}_20260923

case "$corpus:$split" in
  notsofar:dev)
    set_name=notsofar_dev_sc_plaza_0_full_wavlm_spashl72k_bpechars4_tokoracle
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
    ;;
  notsofar:test)
    set_name=notsofar_eval_sc_plaza_0_full_wavlm_spashl72k_bpechars4_tokoracle
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
    ;;
  ami:dev)
    set_name=ami_sdm_dev_full_wavlm_spashl72k_bpechars4_tokoracle
    manifest=data/manifests/real_meetings/ami/ami_sdm_dev_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/ami_sdm_dev_full_normalized_word_alignments.json
    ;;
  ami:test)
    set_name=ami_sdm_test_full_wavlm_spashl72k_bpechars4_tokoracle
    manifest=data/manifests/real_meetings/ami/ami_sdm_test_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/ami_sdm_test_full_normalized_word_alignments.json
    ;;
  *)
    echo "Unexpected CORPUS:SPLIT=$corpus:$split" >&2
    exit 2
    ;;
esac

if [[ "$corpus" == notsofar ]]; then
  dev_set=notsofar_dev_sc_plaza_0_full_wavlm_spashl72k_bpechars4_tokoracle
  dev_reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
  threshold_json="$exp_dir/align/blank_threshold_notsofar_center20_graph20_remove40_1state_spkattr_bpechars4.json"
else
  dev_set=ami_sdm_dev_full_wavlm_spashl72k_bpechars4_tokoracle
  dev_reference=data/alignments/ami_sdm_dev_full_normalized_word_alignments.json
  threshold_json="$exp_dir/align/blank_threshold_ami_center20_graph20_remove40_1state_spkattr_bpechars4.json"
fi

if [[ ! -s "$threshold_json" ]]; then
  echo "Missing existing 40%-removal threshold: $threshold_json" >&2
  exit 1
fi
blank_threshold=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["blank_probability_threshold"])' "$threshold_json")

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02 --max-num-spks 8 \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker true --max-speaker-score-for-token-graph false \
  --oracle-num-spks 0 --blockwise-speaker-recovery oracle_map \
  --blockwise-speaker-attribution-unit "$attribution_unit" \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin 1.0 \
  --alignment-bpe-max-piece-characters 4 \
  --single-token-duration-per-character 0.08 \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment true \
  --blockwise-whole-recording-inference false \
  --blockwise-graph-chunk-duration 20 \
  --blockwise-token-time-mask true \
  --blockwise-speaker-ordering training \
  --blank-frame-filter-threshold "$blank_threshold" \
  --blank-frame-filter-min-token-ratio 2 \
  --alignment-diagnostics true --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
hyp="$(dirname "$ali")/word_$(basename "$ali")"
dev_hyp="$exp_dir/align/word_alignments_chkpt${iter}_avg${avg}_${dev_set}_collar${collar}_${suffix}.json"
offsets="$exp_dir/align/calibration_${corpus}_${label}.json"

# --relabel computes the oracle recording-level speaker permutation. The
# alignment additionally uses oracle_map to reconcile chunk-local slots before
# accumulating speaker scores for each segment or word.
python local/score_alignment_mean_median.py --relabel \
  --hyp "$hyp" --ref "$reference" \
  --output "${hyp%.json}_uncalibrated_mean_median.json"

if [[ "$split" == dev ]]; then
  calibrated_metrics="${hyp%.json}_full_dev_calibrated_mean_median.json"
  python local/score_alignment_mean_median.py --relabel \
    --hyp "$hyp" --ref "$reference" \
    --calibration-hyp "$hyp" --calibration-ref "$reference" \
    --calibration-output "$offsets" \
    --output "$calibrated_metrics"
else
  if [[ ! -s "$dev_hyp" || ! -s "$offsets" ]]; then
    echo "Missing DEV alignment/calibration required by TEST: $dev_hyp or $offsets" >&2
    exit 1
  fi
  calibrated_metrics="${hyp%.json}_dev_calibrated_mean_median.json"
  python local/score_alignment_mean_median.py --relabel \
    --hyp "$hyp" --ref "$reference" \
    --calibration-offsets "$offsets" \
    --output "$calibrated_metrics"
fi

speaker_metrics="${hyp%.json}_speaker_metrics.txt"
if [[ "$attribution_unit" == word ]]; then
  hyp_region_unit=word
else
  hyp_region_unit=segment
fi
{
  echo "ORACLE_PERMUTATION_UNCALIBRATED_DER_COLLAR_0.25"
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$hyp" --ref "$reference" --ref-cut-manifest "$manifest" \
    --hyp-region-unit "$hyp_region_unit"
  echo "ORACLE_PERMUTATION_CALIBRATED_DER_COLLAR_0.25"
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$hyp" --ref "$reference" --ref-cut-manifest "$manifest" \
    --calibration-offsets "$offsets" --hyp-region-unit "$hyp_region_unit"
  echo "ORACLE_PERMUTATION_UNCALIBRATED_DER_COLLAR_0"
  python local/score_alignment_der.py --relabel --collar 0 \
    --hyp "$hyp" --ref "$reference" --ref-cut-manifest "$manifest" \
    --hyp-region-unit "$hyp_region_unit"
  echo "ORACLE_PERMUTATION_CALIBRATED_DER_COLLAR_0"
  python local/score_alignment_der.py --relabel --collar 0 \
    --hyp "$hyp" --ref "$reference" --ref-cut-manifest "$manifest" \
    --calibration-offsets "$offsets" --hyp-region-unit "$hyp_region_unit"
} | tee "$speaker_metrics"

if [[ "$attribution_unit" == word ]]; then
  majority_segment_metrics="${hyp%.json}_majority_segment_speaker_metrics.txt"
  {
    echo "ORACLE_PERMUTATION_CALIBRATED_MAJORITY_SEGMENT_DER_COLLAR_0.25"
    python local/score_alignment_der.py --relabel --collar 0.25 \
      --hyp "$hyp" --ref "$reference" --ref-cut-manifest "$manifest" \
      --calibration-offsets "$offsets" --hyp-region-unit segment_majority
    echo "ORACLE_PERMUTATION_CALIBRATED_MAJORITY_SEGMENT_DER_COLLAR_0"
    python local/score_alignment_der.py --relabel --collar 0 \
      --hyp "$hyp" --ref "$reference" --ref-cut-manifest "$manifest" \
      --calibration-offsets "$offsets" --hyp-region-unit segment_majority
  } | tee "$majority_segment_metrics"
fi

echo "CORPUS=$corpus"
echo "SPLIT=$split"
echo "TOKEN_GRAPH=true"
echo "SPEAKER_RECOVERY=oracle_map"
echo "SPEAKER_ATTRIBUTION_UNIT=$attribution_unit"
echo "SCORING_PERMUTATION=oracle_hungarian_per_recording"
echo "BLANK_THRESHOLD_JSON=$threshold_json"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$hyp"
echo "CALIBRATED_METRICS_JSON=$calibrated_metrics"
echo "SPEAKER_METRICS=$speaker_metrics"
