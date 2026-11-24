#!/usr/bin/env bash
#SBATCH --job-name=cmp3-hs1-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

collar=32000
suffix=training_topo_allframes_exact_hardseg100_compare20260921
synth_ref=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json

for spec in \
  'parakeet_avg5|exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe|200000|5' \
  'wavlm_spashl56k|exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4|56000|1' \
  'wavlm_shuffle70k|exp/fastmss_yodas_en_wavlm_large_shuffle_v100x4_accum4|70000|1'; do
  IFS='|' read -r model exp_dir iter avg <<< "$spec"
  align_dir="$exp_dir/align"
  synth_hyp="$align_dir/word_alignments_chkpt${iter}_avg${avg}_synth3_collar${collar}_${suffix}.json"
  dev_hyp="$align_dir/word_alignments_chkpt${iter}_avg${avg}_notsofar_dev30_sc_plaza_0_collar${collar}_${suffix}.json"
  eval_hyp="$align_dir/word_alignments_chkpt${iter}_avg${avg}_notsofar_eval30_sc_plaza_0_collar${collar}_${suffix}.json"
  offsets="$align_dir/${model}_hardseg100_notsofar_dev_timing_offsets.json"

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
done
