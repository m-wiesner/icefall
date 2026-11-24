#!/usr/bin/env bash
#SBATCH --job-name=alignment-browser
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
host=$(hostname -f)
port=${PORT:-8765}
echo "SERVER_HOST=$host"
echo "SERVER_PORT=$port"
python local/alignment_browser.py --host 0.0.0.0 --port "$port"
