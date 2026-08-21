#!/usr/bin/env bash
#SBATCH --job-name=mtasr-hlg-training-topo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

hlg=data/lang_bpe_5000/HLG_3_gram_1e_7_training_topology.pt
if [[ ! -s "$hlg" ]]; then
  python local/prepare_hlg.py \
    --lm-dir data/lm \
    --arpa 3-gram.pruned.1e-7.arpa \
    --lang-dir data/lang_bpe_5000 \
    --training-topology \
    --output "$hlg"
fi

export EXP_DIR=${EXP_DIR:-exp/shuffle_large_length_brownian}
export ITER=${ITER:-72000}
export AVG=${AVG:-4}
export HLG=$hlg
export SUFFIX=${SUFFIX:-hlg_3gram_training_topology_acwt2_bw2}
exec bash slurm/decode_synth_hlg.sbatch
