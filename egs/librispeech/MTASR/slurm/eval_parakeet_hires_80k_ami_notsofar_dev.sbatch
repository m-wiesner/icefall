#!/usr/bin/env bash
#SBATCH --job-name=eval-par80k-realdev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_parakeet_hires_local_factorized_pause1_a100_accum4
iteration=80000
average=1
suffix=greedy_norm20ms_dev30
ami_name=ami_sdm_dev30
nsf_name=notsofar_sdm_dev30
ami_manifest=data/manifests/real_meetings/ami/ami_sdm_dev_max30_normalized.jsonl.gz
nsf_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized.jsonl.gz

python parakeet_native_spashl_ctc/decode.py --method greedy -- \
  --iter "$iteration" \
  --avg "$average" \
  --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000 \
  --test-sets "$ami_name $nsf_name" \
  --test-cut-manifest "$ami_name=$ami_manifest" \
  --test-cut-manifest "$nsf_name=$nsf_manifest" \
  --suffix "$suffix" \
  --max-num-spks 8 \
  --factorized-token-speaker-output true \
  --speaker-weight 1.0 \
  --sort-strategy speech_time \
  --max-duration 80 \
  --num-workers 4 \
  --downmix-multichannel true \
  --input-sampling-rate 16000 \
  --downsample false \
  --use-layer-norm false \
  --frame-duration 0.02

for test_set in "$ami_name" "$nsf_name"; do
  stem="$exp_dir/decode/hyps_chkpt${iteration}_avg${average}_${test_set}_${suffix}"
  ref="$exp_dir/decode/refs_chkpt${iteration}_avg${average}_${test_set}_${suffix}.stm"
  hyp="${stem}.stm"
  python -m meeteval.wer tcpwer \
    --reference "$ref" \
    --hypothesis "$hyp" \
    --collar 5 \
    --average-out "${stem}_tcpwer.json" \
    --per-reco-out "${stem}_tcpwer_per_reco.json"
done
