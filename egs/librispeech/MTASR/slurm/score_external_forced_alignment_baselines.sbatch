#!/usr/bin/env bash
#SBATCH --job-name=ext-fa-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
baseline_dir=/expscratch/mwiesner/shuffle/baselines
result_dir="$baseline_dir/results"
metric_dir="$result_dir/metrics"
cd "$recipe_dir"
source activate.sh
mkdir -p "$metric_dir"

synth_ref=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json

for system in mfa wav2vec2_base960h; do
  synth_hyp="$result_dir/${system}_dev_synth_3spk.json"
  dev_hyp="$result_dir/${system}_notsofar_dev30.json"
  eval_hyp="$result_dir/${system}_notsofar_eval30.json"
  offsets="$metric_dir/${system}_notsofar_dev_timing_offsets.json"

  python local/score_alignment_mean_median.py --relabel \
    --hyp "$synth_hyp" --ref "$synth_ref" \
    --output "$metric_dir/${system}_dev_synth_3spk_uncalibrated.json"
  python local/score_alignment_mean_median.py --relabel --calibrate-half \
    --hyp "$synth_hyp" --ref "$synth_ref" \
    --output "$metric_dir/${system}_dev_synth_3spk_firsthalf_calibrated_secondhalf.json"

  python local/score_alignment_mean_median.py --relabel \
    --hyp "$dev_hyp" --ref "$dev_ref" \
    --output "$metric_dir/${system}_notsofar_dev30_uncalibrated.json"
  python local/score_alignment_mean_median.py --relabel \
    --hyp "$dev_hyp" --ref "$dev_ref" \
    --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
    --calibration-output "$offsets" \
    --output "$metric_dir/${system}_notsofar_dev30_full_dev_calibrated.json"

  python local/score_alignment_mean_median.py --relabel \
    --hyp "$eval_hyp" --ref "$eval_ref" \
    --output "$metric_dir/${system}_notsofar_eval30_uncalibrated.json"
  python local/score_alignment_mean_median.py --relabel \
    --hyp "$eval_hyp" --ref "$eval_ref" \
    --calibration-offsets "$offsets" \
    --output "$metric_dir/${system}_notsofar_eval30_dev_calibrated.json"
done
