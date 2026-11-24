#!/usr/bin/env bash
#SBATCH --job-name=mtasr-wavlm-fifo-diar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

# The implementation and complete data/optimizer configuration live in the
# common launcher.  A max duration of 600 completed an unfrozen A100 probe at
# 25.77 GiB, leaving useful headroom on the 40-GB cards.
export WORLD_SIZE=${WORLD_SIZE:-4}
export MAX_DURATION=${MAX_DURATION:-600}
export MASTER_PORT=${MASTER_PORT:-15395}

exec bash "${SLURM_SUBMIT_DIR}/slurm/train_fastmss_yodas_wavlm_large_fifo_diarization_h200x2.sbatch"
