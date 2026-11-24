#!/usr/bin/env bash
#SBATCH --job-name=wl50-hs1-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4
iter=50000
avg=1
collar=32000
margin=1.00
suffix=training_topo_allframes_exact_hardseg100

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
  set_name=synth3
  manifest_args=()
  reference=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
  oracle_num_spks=3
  max_duration=400
else
  set_name=notsofar_eval30_sc_plaza_0
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
  manifest_args=(--test-cut-manifest "$set_name=$manifest")
  reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
  oracle_num_spks=0
  max_duration=80
fi

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --test-sets "$set_name" "${manifest_args[@]}" \
  --suffix "$suffix" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --ignore-speaker false \
  --max-num-spks 8 --oracle-num-spks "$oracle_num_spks" \
  --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin "$margin" \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration "$max_duration" --max-cuts 4 \
  --num-buckets 1 --num-workers 4 \
  --normalize-loudness false \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

python local/score_alignments.py --relabel \
  --hyp "$word_ali" --ref "$reference" \
  | tee "${word_ali%.json}_uncalibrated_stats.txt"

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
  # The scorer estimates timing biases on the first half and reports metrics
  # only on the disjoint second half.
  python local/score_alignments.py --relabel --calibrate \
    --hyp "$word_ali" --ref "$reference" \
    | tee "${word_ali%.json}_firsthalf_calibrated_secondhalf_stats.txt"
else
  dev_hyp="$exp_dir/align/word_alignments_chkpt${iter}_avg${avg}_notsofar_dev30_sc_plaza_0_collar${collar}_${suffix}_full_dev.json"
  dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
  offsets="$exp_dir/align/wavlm_spashl50k_hardseg100_notsofar_dev30_timing_calibration_median.json"

  # Estimate one set of median offsets from the complete DEV set.  The first
  # scoring target is DEV only so that the stored calibration is auditable.
  python local/score_alignments.py --relabel \
    --hyp "$dev_hyp" --ref "$dev_ref" \
    --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
    --calibration-statistic median --calibration-output "$offsets" \
    | tee "${dev_hyp%.json}_full_dev_external_calibrated_stats.txt"

  python local/score_alignments.py --relabel \
    --hyp "$word_ali" --ref "$reference" \
    --calibration-offsets "$offsets" \
    | tee "${word_ali%.json}_dev_calibrated_stats.txt"
  echo "CALIBRATION_OFFSETS=$offsets"
fi

echo "SEGMENT_TIME_WINDOW_MARGIN=$margin seconds on each side"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
