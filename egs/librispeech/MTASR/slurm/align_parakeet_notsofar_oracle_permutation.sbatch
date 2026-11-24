#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-oracleperm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
if (( SLURM_ARRAY_TASK_ID == 0 )); then export SPLIT=dev; else export SPLIT=eval; fi
export SEGMENTATION_SOURCE=token_alignment
export SPEAKER_RECOVERY=identity_map
export ORACLE_NUM_SPEAKERS=true
export SUFFIX="row7_diar_${SPLIT}_tokenali_fixed_global_permutation"
exec bash slurm/align_parakeet_notsofar_embedding_stitch.sbatch
