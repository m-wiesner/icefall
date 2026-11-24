#!/usr/bin/env bash
#SBATCH --job-name=nsf-eval-embseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
export SPLIT=eval
export ORACLE_NUM_SPEAKERS=true
export TIMING_CALIBRATION="$exp_dir/align/notsofar_dev_whisper_token_only_timing_calibration_median.json"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    export SEGMENTATION_SOURCE=posterior
    export SUFFIX=token_only_center20_graphchunks_mask_corpus30_remap_ecapa_localdiar_b04_s045_smooth500_pyannote21_ahc_oraclenum_posterior_eval
    ;;
  1)
    export SEGMENTATION_SOURCE=token_alignment
    export TOKEN_GAP=0.25
    export SUFFIX=token_only_center20_graphchunks_mask_corpus30_remap_ecapa_localdiar_b04_s045_smooth500_pyannote21_ahc_oraclenum_tokenali_eval
    ;;
  *)
    echo "Unexpected array task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

exec bash slurm/align_whisper_notsofar_dev_embedding_stitch.sbatch
