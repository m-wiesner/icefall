#!/usr/bin/env bash
#SBATCH --job-name=mtasr-wh-fm-yod-a100x2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
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

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6}
start_batch=${START_BATCH:-0}
max_cuts=${MAX_CUTS:-2}
max_states=${MAX_STATES:-25000000}
bpe_model=${BPE_MODEL:-data/lang_bpe_5000_librispeech_yodas_en/bpe.model}
lang_dir=${LANG_DIR:-data/lang_bpe_5000_librispeech_yodas_en}
mkdir -p exp/slurm "$exp_dir"

python whisper_spashl_ctc/train.py \
  --world-size 2 \
  --master-port 15347 \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause1.0_max30.jsonl.gz \
  --train-yodas-scheme-dirs \
    /expscratch/mwiesner/shuffle/yodas_shar_filtered/en_judge8_der0069_doverlap \
  --yodas-hypothesis doverlap \
  --yodas-group-max-pause 1.0 \
  --yodas-group-max-duration 30 \
  --yodas-min-speakers 2 \
  --yodas-max-speakers 8 \
  --yodas-mix-weight 0.5 \
  --bpe-model "$bpe_model" \
  --lang-dir "$lang_dir" \
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
  --collar 64000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 30 \
  --max-states "$max_states" \
  --total-max-overlaps 6000 \
  --base-lr 0.000005 \
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
  --num-workers 6 \
  --use-fp16 true \
  --save-every-n 2000 \
  --keep-last-k 5 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
