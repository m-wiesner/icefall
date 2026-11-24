#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-vbxsel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-27%8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

# Two compact Cartesian sweeps around the best full-Dev setting:
#   0--15: confidence x neighboring-token radius (gap=.50, clean=1.44s)
#  16--27: token gap x minimum-clean duration (confidence=.80, radius=2)
confidences=(0.70 0.80 0.90 0.95)
neighbors=(0 1 2 3)
gaps=(0.25 0.50 0.75)
clean_seconds=(0.50 1.00 1.44 2.50)
i=$SLURM_ARRAY_TASK_ID
if (( i < 16 )); then
  confidence=${confidences[$((i / 4))]}
  neighbor=${neighbors[$((i % 4))]}
  gap=0.50
  clean=1.44
  tag="conf${confidence//./p}_nbr${neighbor}"
else
  j=$((i - 16))
  gap=${gaps[$((j / 4))]}
  clean=${clean_seconds[$((j % 4))]}
  confidence=0.80
  neighbor=2
  tag="gap${gap//./p}_clean${clean//./p}"
fi

export SPLIT=dev
export SET_NAME=notsofar_sc_plaza_dev_vbx_sweep6
export MANIFEST=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_vbx_sweep6.jsonl.gz
export SEGMENTATION_SOURCE=token_alignment
export SPEAKER_RECOVERY=vbx
export ORACLE_NUM_SPEAKERS=true
export TOKEN_SPEAKER_CONFIDENCE=$confidence
export TOKEN_NEIGHBORS=$neighbor
export TOKEN_GAP=$gap
export VBX_MIN_CLEAN_SECONDS=$clean
export SUFFIX="row7_vbx_tokensel_sweep6_${tag}"
exec bash slurm/align_parakeet_notsofar_embedding_stitch.sbatch
