#!/usr/bin/env bash
#SBATCH --job-name=repro-wh-factorized
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
export SHUFFLE_CTC_CAPTURE_PATH="$recipe_dir/exp/factorized_repro/whisper_rank{rank}.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exp_dir=exp/whisper_factorized_repro_82000
mkdir -p exp/slurm exp/factorized_repro "$exp_dir"
source_checkpoint="$recipe_dir/exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6/checkpoint-82000.pt"
replay_checkpoint="$recipe_dir/$exp_dir/checkpoint-82000.pt"
if [[ ! -e "$replay_checkpoint" ]]; then
  ln -s "$source_checkpoint" "$replay_checkpoint"
fi

python whisper_spashl_ctc/train.py \
  --world-size 2 --master-port 15337 \
  --start-batch 82000 --exp-dir "$exp_dir" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause1.0_max30.jsonl.gz \
    data/manifests/real_meetings/ami/ami_sdm_train_max30_normalized.jsonl.gz \
    data/manifests/real_meetings/notsofar/notsofar_sdm_train_240825.1_train_max30_normalized.jsonl.gz \
  --bpe-model data/lang_bpe_5000/bpe.model --lang-dir data/lang_bpe_5000 \
  --whisper-model-id openai/whisper-large-v3-turbo \
  --whisper-valid-cut-manifest data/manifests/cuts_librispeech_dev_synth_max30.jsonl.gz \
  --factorized-token-speaker-output true --factorized-ctc-loss true \
  --speaker-branch true --downsample false --use-layer-norm false \
  --max-num-spks 8 --max-overlap 4 --max-train-cut-duration 30 \
  --speaker-weight 1.0 --collar 64000 --edge-collar 0 \
  --interval-model brownian_bridge --beam-size 30 \
  --total-max-overlaps 6000 --base-lr 0.000005 --pct-start 0.15 \
  --total-steps 84050 --freeze-iters 2000 --freeze-lr 0.005 \
  --valid-interval 3000 --original-topo false --sort-strategy speech_time \
  --max-duration 60 --max-cuts 2 --accum-grad 4 \
  --num-buckets 10 --num-cuts-for-bins-estimate 100 \
  --sampler-buffer-size 200 --fixed-sampler-skip-batches 328000 \
  --num-workers 6 --use-fp16 true --save-every-n 2000 --keep-last-k 5 \
  --average-period 200 --tensorboard false --seed 42
