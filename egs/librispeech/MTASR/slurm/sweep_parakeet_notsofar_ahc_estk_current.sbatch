#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-ahc-estk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-17%8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

thresholds=(0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.10)
i=$SLURM_ARRAY_TASK_ID
split_i=$((i / 9))
threshold=${thresholds[$((i % 9))]}
if (( split_i == 0 )); then split=dev; else split=eval; fi

export SPLIT=$split
export SEGMENTATION_SOURCE=token_alignment
export SPEAKER_RECOVERY=embedding_cluster
export ORACLE_NUM_SPEAKERS=false
export TOKEN_SPEAKER_CONFIDENCE=0.80
export TOKEN_NEIGHBORS=2
export TOKEN_GAP=0.75
export CLUSTERING_THRESHOLD=$threshold
export SUFFIX="row7_tokenali_ahc_estK_c080_n2_gap075_${split}_thr${threshold//./p}"
exec bash slurm/align_parakeet_notsofar_embedding_stitch.sbatch
