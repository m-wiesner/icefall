#!/usr/bin/env bash
#SBATCH --job-name=nsf-ahc-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

thresholds=(0.20 0.30 0.40 0.50 0.55 0.60 0.65 0.7153814381597874)
threshold=${thresholds[$SLURM_ARRAY_TASK_ID]}
tag=${threshold//./p}

export ORACLE_NUM_SPEAKERS=false
export CLUSTERING_THRESHOLD=$threshold
export SUFFIX="token_only_center20_graphchunks_mask_corpus30_remap_ecapa_localdiar_b04_s045_smooth500_pyannote21_ahc_estnum_thr${tag}"

exec bash slurm/align_whisper_notsofar_dev_embedding_stitch.sbatch
