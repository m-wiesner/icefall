#!/usr/bin/env bash
#SBATCH --job-name=wl8-nsfdev30-greedy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export TORCH_HOME=${TORCH_HOME:-/expscratch/mwiesner/shuffle/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4
iteration=100000
average=1
test_set=notsofar_dev30_sc_plaza_0_argmax_wavlm8spk100k
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
suffix=greedy_shortform_sc_plaza_wavlm8spk100k_wordstm_tcpwer5

python wavlm_spashl_ctc/decode.py --method greedy -- \
  --iter "$iteration" \
  --avg "$average" \
  --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$test_set" \
  --test-cut-manifest "$test_set=$manifest" \
  --suffix "$suffix" \
  --use-large true \
  --max-num-spks 8 \
  --factorized-token-speaker-output true \
  --speaker-weight 1.0 \
  --sort-strategy speech_time \
  --max-duration 100 \
  --max-cuts 4 \
  --num-buckets 1 \
  --num-workers 4 \
  --downmix-multichannel true \
  --input-sampling-rate 16000 \
  --word-level-stm true \
  --downsample false \
  --use-layer-norm false \
  --frame-duration 0.02

stem="$exp_dir/decode/hyps_chkpt${iteration}_avg${average}_${test_set}_${suffix}"
ref="$exp_dir/decode/refs_chkpt${iteration}_avg${average}_${test_set}_${suffix}.stm"
hyp="${stem}.stm"

python -m meeteval.wer tcpwer \
  --reference "$ref" \
  --hypothesis "$hyp" \
  --collar 5 \
  --average-out "${stem}_tcpwer5.json" \
  --per-reco-out "${stem}_tcpwer5_per_reco.json"

echo "tcpWER with 5-second collar:"
cat "${stem}_tcpwer5.json"
