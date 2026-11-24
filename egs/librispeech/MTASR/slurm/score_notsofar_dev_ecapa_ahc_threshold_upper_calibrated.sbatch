#!/usr/bin/env bash
#SBATCH --job-name=nsf-ahc-upcal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
offsets=$exp_dir/align/notsofar_dev_whisper_token_only_timing_calibration_median.json
tags=(0p75 0p80 0p85 0p90 0p95 1p00 1p10 1p20 1p40 1p60)

for tag in "${tags[@]}"; do
  suffix=token_only_center20_graphchunks_mask_corpus30_remap_ecapa_localdiar_b04_s045_smooth500_pyannote21_ahc_estnum_thr${tag}
  hyp=$exp_dir/align/word_alignments_chkpt82000_avg1_notsofar_sc_plaza_full_collar32000_${suffix}.json
  stats=${hyp%.json}_calibrated_stats.txt
  {
    python local/score_alignments.py --relabel --hyp "$hyp" --ref "$ref" \
      --calibration-offsets "$offsets"
    echo WORD_BOUNDARY_DER
    python local/score_alignment_der.py --relabel --collar 0.25 \
      --hyp "$hyp" --ref "$ref" --calibration-offsets "$offsets"
    echo SUPERVISION_BOUNDARY_DER
    python local/score_alignment_der.py --relabel --collar 0.25 \
      --hyp "$hyp" --ref "$ref" --ref-cut-manifest "$manifest" \
      --calibration-offsets "$offsets"
  } | tee "$stats"
done
