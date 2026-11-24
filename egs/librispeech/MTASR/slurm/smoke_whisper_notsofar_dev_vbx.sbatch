#!/usr/bin/env bash
#SBATCH --job-name=nsf-vbx-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
export SPEAKER_RECOVERY=vbx
export SEGMENTATION_SOURCE=posterior
export ORACLE_NUM_SPEAKERS="${ORACLE_NUM_SPEAKERS:-true}"
export RUN_SCORING=false
export TEST_CUT_ID=nsf-dev-MTG_30500-sc_plaza_0
export SUFFIX="${SUFFIX:-token_only_center20_graphchunks_mask_corpus30_remap_vbx_posterior_oracle_smoke}"
exec bash slurm/align_whisper_notsofar_dev_embedding_stitch.sbatch
