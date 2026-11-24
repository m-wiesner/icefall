#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-tok-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4
iter=${ITER:-72000}
avg=${AVG:-1}
collar=32000
suffix=token_only_training_topo_allframes_exact_hardseg100_spashl72k_spkweight1_20260922
align_dir="$exp_dir/align"

synth_hyp="$align_dir/word_alignments_chkpt${iter}_avg${avg}_synth3_collar${collar}_${suffix}.json"
dev_set=notsofar_dev30_sc_plaza_0_token_only_wavlm_spashl72k
eval_set=notsofar_eval30_sc_plaza_0_token_only_wavlm_spashl72k
dev_hyp="$align_dir/word_alignments_chkpt${iter}_avg${avg}_${dev_set}_collar${collar}_${suffix}.json"
eval_hyp="$align_dir/word_alignments_chkpt${iter}_avg${avg}_${eval_set}_collar${collar}_${suffix}.json"

synth_ref=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
offsets="$align_dir/wavlm_spashl72k_token_only_hardseg100_notsofar_dev_timing_offsets.json"

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
