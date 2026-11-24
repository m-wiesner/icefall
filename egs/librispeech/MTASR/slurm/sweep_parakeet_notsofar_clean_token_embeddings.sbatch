#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-cleanemb
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

confidences=(0.60 0.70 0.80 0.90)
i=$SLURM_ARRAY_TASK_ID
method_i=$((i / 4))
confidence=${confidences[$((i % 4))]}

export SPLIT=dev
export SEGMENTATION_SOURCE=token_alignment
export ORACLE_NUM_SPEAKERS=true
export TOKEN_GAP=0.50
export TOKEN_SPEAKER_CONFIDENCE=$confidence
if (( method_i == 0 )); then
  export SPEAKER_RECOVERY=embedding_cluster
  method=ahc
else
  export SPEAKER_RECOVERY=vbx
  method=vbx
fi
export SUFFIX="row7_clean_tokenemb_${method}_oracleK_conf${confidence//./p}"
exec bash slurm/align_parakeet_notsofar_embedding_stitch.sbatch
