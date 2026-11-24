#!/usr/bin/env bash
#SBATCH --job-name=nsf30-par-spk-score
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
checkpoint_tag=chkpt200000_avg5
collar=32000
max_cuts=${MAX_CUTS:-4}
suffix=speaker_attributed_training_topo_allframes_exact_batched_viterbi_batch${max_cuts}
dev_set=notsofar_dev30_sc_plaza_0_parakeet_avg5
eval_set=notsofar_eval30_sc_plaza_0_parakeet_avg5
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
dev_hyp=$exp_dir/align/word_alignments_${checkpoint_tag}_${dev_set}_collar${collar}_${suffix}.json
eval_hyp=$exp_dir/align/word_alignments_${checkpoint_tag}_${eval_set}_collar${collar}_${suffix}.json
offsets=$exp_dir/align/parakeet_avg5_notsofar_dev30_sc_plaza_0_speaker_attributed_full_dev_timing_calibration_median.json

python local/score_alignments.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-statistic median --calibration-output "$offsets" \
  | tee "${dev_hyp%.json}_full_dev_calibrated_stats.txt"

python local/score_alignments.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-offsets "$offsets" \
  | tee "${eval_hyp%.json}_dev_calibrated_stats.txt"

echo "CALIBRATION_OFFSETS=$offsets"
echo "DEV_HYP=$dev_hyp"
echo "EVAL_HYP=$eval_hyp"
