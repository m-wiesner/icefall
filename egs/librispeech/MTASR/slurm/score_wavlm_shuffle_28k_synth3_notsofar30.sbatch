#!/usr/bin/env bash
#SBATCH --job-name=wavsh28k-tok-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

exp_dir=exp/fastmss_yodas_en_wavlm_large_shuffle_v100x4_accum4
tag=chkpt28000_avg1
collar=32000
suffix=token_only_training_topo_allframes_exact_batched_viterbi_batch4

synth_hyp="$exp_dir/align/word_alignments_${tag}_synth3_collar${collar}_${suffix}.json"
synth_ref=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
dev_set=notsofar_dev30_sc_plaza_0_token_only_wavlm_shuffle_28k
eval_set=notsofar_eval30_sc_plaza_0_token_only_wavlm_shuffle_28k
dev_hyp="$exp_dir/align/word_alignments_${tag}_${dev_set}_collar${collar}_${suffix}.json"
eval_hyp="$exp_dir/align/word_alignments_${tag}_${eval_set}_collar${collar}_${suffix}.json"
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
offsets="$exp_dir/align/wavlm_shuffle_28k_notsofar_dev30_timing_calibration_median.json"

python local/score_alignments.py --relabel \
  --hyp "$synth_hyp" --ref "$synth_ref" \
  | tee "${synth_hyp%.json}_uncalibrated_stats.txt"
python local/score_alignments.py --relabel --calibrate \
  --hyp "$synth_hyp" --ref "$synth_ref" \
  | tee "${synth_hyp%.json}_calibrated_stats.txt"

python local/score_alignments.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  | tee "${dev_hyp%.json}_uncalibrated_stats.txt"
python local/score_alignments.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-statistic median --calibration-output "$offsets" \
  | tee "${dev_hyp%.json}_full_dev_calibrated_stats.txt"

python local/score_alignments.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  | tee "${eval_hyp%.json}_uncalibrated_stats.txt"
python local/score_alignments.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-offsets "$offsets" \
  | tee "${eval_hyp%.json}_dev_calibrated_stats.txt"

echo "CALIBRATION_OFFSETS=$offsets"
