#!/usr/bin/env bash
#SBATCH --job-name=parakeet-upsample-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh
export HF_HOME=${HF_HOME:-$recipe_dir/.cache/huggingface}
export HF_HUB_OFFLINE=1

python parakeet_spashl_ctc/smoke_test.py
python parakeet_spashl_ctc/recipe_smoke_test.py
PYTHONPATH="$recipe_dir/parakeet_shuffle_ctc:$PYTHONPATH" \
  python parakeet_shuffle_ctc/recipe_smoke_test.py
