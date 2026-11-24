#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-vbxnbr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"

neighbors=(3 4 5 6)
neighbor=${neighbors[$SLURM_ARRAY_TASK_ID]}
export SPLIT=dev
export SET_NAME=notsofar_sc_plaza_dev_vbx_sweep6
export MANIFEST=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_vbx_sweep6.jsonl.gz
export SEGMENTATION_SOURCE=token_alignment
export SPEAKER_RECOVERY=vbx
export ORACLE_NUM_SPEAKERS=true
export TOKEN_SPEAKER_CONFIDENCE=0.95
export TOKEN_NEIGHBORS=$neighbor
export TOKEN_GAP=0.75
export VBX_MIN_CLEAN_SECONDS=1.44
export SUFFIX="row7_vbx_tokensel_sweep6_conf0p95_gap0p75_clean1p44_nbr${neighbor}"
exec bash slurm/align_parakeet_notsofar_embedding_stitch.sbatch
