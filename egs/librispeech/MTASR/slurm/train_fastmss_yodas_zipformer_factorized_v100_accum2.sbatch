#!/usr/bin/env bash
#SBATCH --job-name=mtasr-zip-yodas-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_zipformer_factorized_v100_accum2_combined_bpe}
start_batch=${START_BATCH:-0}
mkdir -p exp/slurm "$exp_dir"

python zipformer_spashl_ctc/train.py \
  --world-size 4 \
  --master-port 15323 \
  --exp-dir "$exp_dir" \
  --start-batch "$start_batch" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause1.0.jsonl.gz \
  --train-yodas-scheme-dirs \
    /expscratch/mwiesner/shuffle/yodas_shar_filtered/en_judge8_der0069_doverlap \
  --yodas-hypothesis doverlap \
  --yodas-group-max-pause 1.0 \
  --yodas-group-max-duration 60 \
  --yodas-min-speakers 2 \
  --yodas-max-speakers 8 \
  --yodas-mix-weight 0.5 \
  --bpe-model data/lang_bpe_5000_librispeech_yodas_en/bpe.model \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --factorized-token-speaker-output true \
  --factorized-ctc-loss true \
  --speaker-branch true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 8 \
  --max-overlap 4 \
  --max-train-cut-duration 100 \
  --speaker-weight 1.0 \
  --collar 32000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 10 \
  --total-max-overlaps 6000 \
  --optimizer-name scaled_adam \
  --lr-scheduler-name eden2 \
  --base-lr 0.02 \
  --lr-batches 7500 \
  --lr-warmup-updates 5000 \
  --lr-warmup-start 0.0 \
  --total-steps 200000 \
  --freeze-iters 0 \
  --valid-interval 3000 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration 1800 \
  --max-cuts "${MAX_CUTS:-2}" \
  --accum-grad 2 \
  --num-buckets 10 \
  --num-cuts-for-bins-estimate 100 \
  --sampler-buffer-size 200 \
  --num-workers 6 \
  --use-fp16 true \
  --save-every-n 2000 \
  --keep-last-k 5 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
