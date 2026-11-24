#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-full-orc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

corpus=${CORPUS:?Set CORPUS=ami or notsofar}
mode=${MODE:?Set MODE=attributed or token}
split=${SPLIT:?Set SPLIT=dev or test}
if [[ "$mode" != attributed && "$mode" != token ]]; then
  echo "Unexpected MODE=$mode" >&2
  exit 2
fi

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned}
iter=72000
avg=1
collar=32000

case "$corpus:$split" in
  ami:dev)
    set_name=ami_sdm_dev_full_wavlm_spashl72k
    manifest=data/manifests/real_meetings/ami/ami_sdm_dev_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/ami_sdm_dev_full_normalized_word_alignments.json
    ;;
  ami:test)
    set_name=ami_sdm_test_full_wavlm_spashl72k
    manifest=data/manifests/real_meetings/ami/ami_sdm_test_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/ami_sdm_test_full_normalized_word_alignments.json
    ;;
  notsofar:dev)
    set_name=notsofar_dev_sc_plaza_0_full_wavlm_spashl72k
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
    ;;
  notsofar:test)
    set_name=notsofar_eval_sc_plaza_0_full_wavlm_spashl72k
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
    ;;
  *)
    echo "Unexpected CORPUS:SPLIT=$corpus:$split" >&2
    exit 2
    ;;
esac

if [[ "$corpus" == ami ]]; then
  dev_set=ami_sdm_dev_full_wavlm_spashl72k
  dev_ref=data/alignments/ami_sdm_dev_full_normalized_word_alignments.json
  threshold_json="$exp_dir/align/blank_threshold_${dev_set}_center20_graph20_corpus30.json"
else
  dev_set=notsofar_dev_sc_plaza_0_full_wavlm_spashl72k
  dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
  # Already calibrated on this model, corpus, microphone, and 30-second/
  # center-20-second acoustic inference configuration.
  threshold_json="$exp_dir/align/blank_threshold_${dev_set}_03_1state_corpus30_fullgraph.json"
fi

label=center20_graph20_corpus30_1state_oracle_${mode}
suffix=longform_${label}_20260923
if [[ "$mode" == token ]]; then
  ignore_speaker=true
else
  ignore_speaker=false
fi

blank_args=()
if [[ "$corpus:$mode:$split" == ami:attributed:dev ]]; then
  blank_args=(
    --blank-frame-filter-target-average-fraction 0.30
    --blank-frame-filter-calibration-set "$set_name"
    --blank-frame-filter-threshold-output "$threshold_json"
    --blank-frame-filter-min-token-ratio 2
  )
else
  blank_threshold=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["blank_probability_threshold"])' "$threshold_json")
  blank_args=(
    --blank-frame-filter-threshold "$blank_threshold"
    --blank-frame-filter-min-token-ratio 2
  )
fi

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02 --max-num-spks 8 \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker "$ignore_speaker" --oracle-num-spks 0 \
  --blockwise-speaker-recovery oracle_map \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin 1.0 \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment true \
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

dev_suffix=longform_center20_graph20_corpus30_1state_oracle_${mode}_20260923
dev_word="$exp_dir/align/word_alignments_chkpt${iter}_avg${avg}_${dev_set}_collar${collar}_${dev_suffix}.json"
if [[ "$split" == dev ]]; then
  calibration_hyp=$word_ali
  calibrated_metrics="${word_ali%.json}_full_dev_calibrated_mean_median.json"
else
  calibration_hyp=$dev_word
  calibrated_metrics="${word_ali%.json}_dev_calibrated_mean_median.json"
fi

python local/score_alignment_mean_median.py \
  --relabel --hyp "$word_ali" --ref "$reference" \
  --calibration-hyp "$calibration_hyp" --calibration-ref "$dev_ref" \
  --calibration-output "$exp_dir/align/calibration_${corpus}_${mode}_${label}.json" \
  --output "$calibrated_metrics"

speaker_metrics="${word_ali%.json}_speaker_metrics.txt"
{
  echo "UNCALIBRATED_SUPERVISION_BOUNDARY_DER"
  python local/score_alignment_der.py \
    --relabel --collar 0.25 --hyp "$word_ali" --ref "$reference" \
    --ref-cut-manifest "$manifest"
  echo "CALIBRATED_SUPERVISION_BOUNDARY_DER"
  python local/score_alignment_der.py \
    --relabel --collar 0.25 --hyp "$word_ali" --ref "$reference" \
    --ref-cut-manifest "$manifest" \
    --calibration-hyp "$calibration_hyp" --calibration-ref "$dev_ref" \
    --calibration-statistic median
} | tee "$speaker_metrics"

echo "CORPUS=$corpus"
echo "MODE=$mode"
echo "SPLIT=$split"
echo "LABEL=$label"
echo "BLANK_THRESHOLD_JSON=$threshold_json"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
echo "CALIBRATED_METRICS_JSON=$calibrated_metrics"
echo "SPEAKER_METRICS=$speaker_metrics"
