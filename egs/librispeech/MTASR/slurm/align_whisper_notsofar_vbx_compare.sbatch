#!/usr/bin/env bash
#SBATCH --job-name=nsf-vbx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
export SPEAKER_RECOVERY=vbx
export TIMING_CALIBRATION="$exp_dir/align/notsofar_dev_whisper_token_only_timing_calibration_median.json"

case "$SLURM_ARRAY_TASK_ID" in
  0) export SPLIT=dev;  export SEGMENTATION_SOURCE=posterior;       export ORACLE_NUM_SPEAKERS=true;  tag=dev_posterior_oraclenum ;;
  1) export SPLIT=dev;  export SEGMENTATION_SOURCE=token_alignment; export ORACLE_NUM_SPEAKERS=true;  tag=dev_tokenali_oraclenum ;;
  2) export SPLIT=eval; export SEGMENTATION_SOURCE=posterior;       export ORACLE_NUM_SPEAKERS=true;  tag=eval_posterior_oraclenum ;;
  3) export SPLIT=eval; export SEGMENTATION_SOURCE=token_alignment; export ORACLE_NUM_SPEAKERS=true;  tag=eval_tokenali_oraclenum ;;
  4) export SPLIT=dev;  export SEGMENTATION_SOURCE=posterior;       export ORACLE_NUM_SPEAKERS=false; tag=dev_posterior_estnum ;;
  5) export SPLIT=dev;  export SEGMENTATION_SOURCE=token_alignment; export ORACLE_NUM_SPEAKERS=false; tag=dev_tokenali_estnum ;;
  6) export SPLIT=eval; export SEGMENTATION_SOURCE=posterior;       export ORACLE_NUM_SPEAKERS=false; tag=eval_posterior_estnum ;;
  7) export SPLIT=eval; export SEGMENTATION_SOURCE=token_alignment; export ORACLE_NUM_SPEAKERS=false; tag=eval_tokenali_estnum ;;
  *) echo "Unexpected array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export TOKEN_GAP=0.25
export SUFFIX="token_only_center20_graphchunks_mask_corpus30_remap_vbx_resnet101_plda_occupancyfix_${tag}"
exec bash slurm/align_whisper_notsofar_dev_embedding_stitch.sbatch
