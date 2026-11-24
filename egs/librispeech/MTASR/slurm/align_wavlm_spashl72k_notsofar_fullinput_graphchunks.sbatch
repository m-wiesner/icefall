#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-lffull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

split=${SPLIT:?Set SPLIT=dev or SPLIT=eval}
exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
collar=32000
label=05_1state_corpus30_graph20_fullinput
threshold_json="$exp_dir/align/blank_threshold_notsofar_dev_sc_plaza_0_full_wavlm_spashl72k_${label}.json"

dev_set=notsofar_dev_sc_plaza_0_full_wavlm_spashl72k
dev_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
dev_suffix=longform_time_mask1_${label}_20260923
dev_word="$exp_dir/align/word_alignments_chkpt${iter}_avg${avg}_${dev_set}_collar${collar}_${dev_suffix}.json"

case "$split" in
  dev)
    set_name=$dev_set
    manifest=$dev_manifest
    reference=$dev_ref
    suffix=$dev_suffix
    blank_args=(
      --blank-frame-filter-target-average-fraction 0.30
      --blank-frame-filter-calibration-set "$dev_set"
      --blank-frame-filter-threshold-output "$threshold_json"
      --blank-frame-filter-min-token-ratio 2
    )
    ;;
  eval)
    set_name=notsofar_eval_sc_plaza_0_full_wavlm_spashl72k
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
    suffix=longform_eval_time_mask1_${label}_20260923
    blank_threshold=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["blank_probability_threshold"])' "$threshold_json")
    blank_args=(
      --blank-frame-filter-threshold "$blank_threshold"
      --blank-frame-filter-min-token-ratio 2
    )
    ;;
  *)
    echo "Unexpected SPLIT=$split (expected dev or eval)" >&2
    exit 2
    ;;
esac

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
  --original-topo false --single-state-ctc-topo true \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin 1.0 \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration 0 --alignment-chunk-context 0 \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment true \
  --blockwise-whole-recording-inference true \
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

if [[ "$split" == dev ]]; then
  python local/score_alignment_mean_median.py \
    --relabel --hyp "$word_ali" --ref "$reference" \
    --calibration-hyp "$word_ali" --calibration-ref "$reference" \
    --calibration-output "$exp_dir/align/calibration_${set_name}_${label}.json" \
    --output "${word_ali%.json}_full_dev_calibrated_mean_median.json"
else
  python local/score_alignment_mean_median.py \
    --relabel --hyp "$word_ali" --ref "$reference" \
    --calibration-hyp "$dev_word" --calibration-ref "$dev_ref" \
    --output "${word_ali%.json}_dev_calibrated_mean_median.json"
fi

echo "SPLIT=$split"
echo "LABEL=$label"
echo "BLANK_THRESHOLD_JSON=$threshold_json"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
