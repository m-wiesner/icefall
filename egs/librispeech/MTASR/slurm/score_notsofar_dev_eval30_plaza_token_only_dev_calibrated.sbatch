#!/usr/bin/env bash
#SBATCH --job-name=nsf30-tok-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

collar=32000
max_cuts=${MAX_CUTS:-4}
suffix=training_topo_allframes_exact_batched_viterbi_batch${max_cuts}
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json

if [[ ${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  model=wavlm_paper
  exp_dir=exp/shuffle_large_length_paper
  checkpoint_tag=chkpt134000_avg1
else
  model=parakeet_avg5
  exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
  checkpoint_tag=chkpt200000_avg5
fi

dev_set=notsofar_dev30_sc_plaza_0_token_only_${model}
eval_set=notsofar_eval30_sc_plaza_0_token_only_${model}
dev_hyp=$exp_dir/align/word_alignments_${checkpoint_tag}_${dev_set}_collar${collar}_${suffix}.json
eval_hyp=$exp_dir/align/word_alignments_${checkpoint_tag}_${eval_set}_collar${collar}_${suffix}.json
offsets=$exp_dir/align/${model}_notsofar_dev30_sc_plaza_0_full_dev_timing_calibration_median.json

# Estimate one pair of offsets from every aligned token in the complete Dev
# set, and apply it back to the complete Dev set (not a held-out half).
python local/score_alignments.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-statistic median --calibration-output "$offsets" \
  | tee "${dev_hyp%.json}_full_dev_calibrated_stats.txt"

# Reuse the Dev offsets unchanged for Eval.
python local/score_alignments.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-offsets "$offsets" \
  | tee "${eval_hyp%.json}_dev_calibrated_stats.txt"

echo "MODEL=$model"
echo "CALIBRATION_OFFSETS=$offsets"
echo "DEV_HYP=$dev_hyp"
echo "EVAL_HYP=$eval_hyp"
