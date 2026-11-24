#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-calib
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
hyp=$exp_dir/align/word_alignments_chkpt200000_avg5_notsofar_sc_plaza_full_collar32000_row7_vbx_tokensel_c080_n2_gap075_clean144_fulldev_uncal.json
ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
output=$exp_dir/align/notsofar_dev_parakeet_row7_timing_calibration_median.json

python local/score_alignments.py \
  --relabel --hyp "$hyp" --ref "$ref" \
  --calibration-hyp "$hyp" --calibration-ref "$ref" \
  --calibration-statistic median --calibration-output "$output"
test -s "$output"
echo "Stored calibration: $output"
