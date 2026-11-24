#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-gap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
gaps=(0.10 0.15 0.25 0.40 0.50)
i=$SLURM_ARRAY_TASK_ID
method_i=$((i / 5)); gap=${gaps[$((i % 5))]}
export SPLIT=dev SEGMENTATION_SOURCE=token_alignment ORACLE_NUM_SPEAKERS=true
export TOKEN_GAP=$gap
if (( method_i == 0 )); then export SPEAKER_RECOVERY=embedding_cluster; method=ahc; else export SPEAKER_RECOVERY=vbx; method=vbx; fi
export SUFFIX="row7_diar_pkthr_b07_s055_dev_tokenali_${method}_oracleK_gap${gap//./p}"
exec bash slurm/align_parakeet_notsofar_embedding_stitch.sbatch
