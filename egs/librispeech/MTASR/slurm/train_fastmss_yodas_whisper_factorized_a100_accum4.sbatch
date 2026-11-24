#!/usr/bin/env bash
#SBATCH --job-name=mtasr-whisper-yodas-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_whisper_turbo_factorized_a100_accum4}
start_batch=${START_BATCH:-0}
max_cuts=${MAX_CUTS:-2}
mkdir -p exp/slurm "$exp_dir"

python whisper_spashl_ctc/train.py \
  --world-size 1 \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/real_meetings/ami/ami_sdm_train_max30_normalized.jsonl.gz \
    data/manifests/real_meetings/notsofar/notsofar_sdm_train_240825.1_train_max30_normalized.jsonl.gz \
  --train-yodas-scheme-dirs \
    /expscratch/mwiesner/shuffle/yodas_shar_filtered/en_judge8_der0069_doverlap \
  --yodas-hypothesis doverlap \
  --yodas-group-max-pause 1.0 \
  --yodas-group-max-duration 30 \
  --yodas-min-speakers 2 \
  --yodas-max-speakers 8 \
  --yodas-mix-weight 0.5 \
  --bpe-model data/lang_bpe_5000_librispeech_yodas_en/bpe.model \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --whisper-model-id openai/whisper-large-v3-turbo \
  --whisper-valid-cut-manifest data/manifests/cuts_librispeech_dev_synth_max30.jsonl.gz \
  --factorized-token-speaker-output true \
  --factorized-ctc-loss true \
  --speaker-branch true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 8 \
  --max-overlap 4 \
  --max-train-cut-duration 30 \
  --speaker-weight 1.0 \
  --collar 32000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 10 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps 200000 \
  --freeze-iters 2000 \
  --freeze-lr 0.005 \
  --valid-interval 3000 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration 60 \
  --max-cuts "$max_cuts" \
  --accum-grad 4 \
  --num-buckets 10 \
  --num-cuts-for-bins-estimate 100 \
  --sampler-buffer-size 200 \
  --num-workers 4 \
  --use-fp16 true \
  --save-every-n 2000 \
  --keep-last-k 5 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
