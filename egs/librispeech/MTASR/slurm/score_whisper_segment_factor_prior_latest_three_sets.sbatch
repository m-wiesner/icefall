#!/usr/bin/env bash
#SBATCH --job-name=whseg-prior-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
cd "$recipe_dir"
source activate.sh

exp_dir=exp/fastmss_yodas_en_whisper_large_v3_turbo_shuffle_segment_factor_softmask_a100x2_accum4_maxlr5e-6
iter=${ITER:-172000}
avg=${AVG:-1}
run_tag=${RUN_TAG:-segmentprior172k}
collar=32000
suffix=segment_factor_prior_training_topo_exact_viterbi
tag=chkpt${iter}_avg${avg}

synth_hyp="$exp_dir/align/word_alignments_${tag}_synth3_collar${collar}_${suffix}.json"
dev_set=notsofar_dev30_sc_plaza_0_token_only_whisper_${run_tag}
eval_set=notsofar_eval30_sc_plaza_0_token_only_whisper_${run_tag}
dev_hyp="$exp_dir/align/word_alignments_${tag}_${dev_set}_collar${collar}_${suffix}.json"
eval_hyp="$exp_dir/align/word_alignments_${tag}_${eval_set}_collar${collar}_${suffix}.json"
synth_ref=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
offsets="$exp_dir/align/whisper_${run_tag}_notsofar_dev30_timing_offsets.json"

python local/score_alignment_mean_median.py --relabel \
  --hyp "$synth_hyp" --ref "$synth_ref" \
  --output "${synth_hyp%.json}_uncalibrated_metrics.json"
python local/score_alignment_mean_median.py --relabel --calibrate-half \
  --hyp "$synth_hyp" --ref "$synth_ref" \
  --output "${synth_hyp%.json}_firsthalf_calibrated_secondhalf_metrics.json"

python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --output "${dev_hyp%.json}_uncalibrated_metrics.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-output "$offsets" \
  --output "${dev_hyp%.json}_full_dev_calibrated_metrics.json"

python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --output "${eval_hyp%.json}_uncalibrated_metrics.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-offsets "$offsets" \
  --output "${eval_hyp%.json}_dev_calibrated_metrics.json"

echo "CALIBRATION_OFFSETS=$offsets"

