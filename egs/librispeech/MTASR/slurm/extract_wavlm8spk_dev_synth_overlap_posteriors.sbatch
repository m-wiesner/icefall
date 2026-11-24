#!/usr/bin/env bash
#SBATCH --job-name=wl8-overlap-viz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export TORCH_HOME=${TORCH_HOME:-/expscratch/mwiesner/shuffle/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

out_dir=/expscratch/mwiesner/shuffle/visualizations/dev_synth_wavlm8spk_overlap

python local/extract_wavlm_overlap_posteriors.py \
  --manifest data/manifests/cuts_librispeech_dev_synth.jsonl.gz \
  --cut-id 88e7fb40-60e0-41bb-bef8-023ae10c3c0d \
  --checkpoint exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/checkpoint-100000.pt \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --output-json "$out_dir/posteriors.json" \
  --output-wav "$out_dir/overlap_example.wav" \
  --max-num-spks 8 \
  --speaker-weight 1.0 \
  --frame-duration 0.02
