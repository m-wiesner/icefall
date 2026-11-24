#!/usr/bin/env bash
#SBATCH --job-name=nsf-ahc-upper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%10
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

thresholds=(0.75 0.80 0.85 0.90 0.95 1.00 1.10 1.20 1.40 1.60)
threshold=${thresholds[$SLURM_ARRAY_TASK_ID]}
tag=${threshold//./p}

export ORACLE_NUM_SPEAKERS=false
export CLUSTERING_THRESHOLD=$threshold
export SUFFIX="token_only_center20_graphchunks_mask_corpus30_remap_ecapa_localdiar_b04_s045_smooth500_pyannote21_ahc_estnum_thr${tag}"

exec bash slurm/align_whisper_notsofar_dev_embedding_stitch.sbatch
