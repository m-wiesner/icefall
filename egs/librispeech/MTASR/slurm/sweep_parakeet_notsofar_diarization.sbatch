#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-diar-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-47%8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

# 2 splits x 2 segment sources x [AHC oracle-K + 9 AHC estimated-K thresholds
# + VBx oracle-K + VBx estimated-K] = 48 matched experiments.
thresholds=(0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.10)
i=$SLURM_ARRAY_TASK_ID
split_i=$((i / 24)); i=$((i % 24))
seg_i=$((i / 12)); variant=$((i % 12))
if (( split_i == 0 )); then export SPLIT=dev; else export SPLIT=eval; fi
if (( seg_i == 0 )); then export SEGMENTATION_SOURCE=posterior; seg=posterior; else export SEGMENTATION_SOURCE=token_alignment; seg=tokenali; fi

if (( variant == 0 )); then
  export SPEAKER_RECOVERY=embedding_cluster ORACLE_NUM_SPEAKERS=true CLUSTERING_THRESHOLD=0.85
  method=ahc_oracleK
elif (( variant <= 9 )); then
  export SPEAKER_RECOVERY=embedding_cluster ORACLE_NUM_SPEAKERS=false
  export CLUSTERING_THRESHOLD=${thresholds[$((variant-1))]}
  method="ahc_estK_t${CLUSTERING_THRESHOLD//./p}"
elif (( variant == 10 )); then
  export SPEAKER_RECOVERY=vbx ORACLE_NUM_SPEAKERS=true CLUSTERING_THRESHOLD=0.85
  method=vbx_oracleK
else
  export SPEAKER_RECOVERY=vbx ORACLE_NUM_SPEAKERS=false CLUSTERING_THRESHOLD=0.85
  method=vbx_estK
fi
export SUFFIX="row7_diar_pkthr_b07_s055_${SPLIT}_${seg}_${method}"
exec bash slurm/align_parakeet_notsofar_embedding_stitch.sbatch
