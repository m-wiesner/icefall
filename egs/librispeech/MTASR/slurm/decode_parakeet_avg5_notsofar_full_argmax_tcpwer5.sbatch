#!/usr/bin/env bash
#SBATCH --job-name=par-nsf-full-greedy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
iteration=200000
average=5
suffix=greedy_full_sc_plaza_factorized_avg5_wordstm_tcpwer5

case "$SLURM_ARRAY_TASK_ID" in
  0)
    test_set=notsofar_dev_sc_plaza_full_argmax
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ;;
  1)
    test_set=notsofar_eval_sc_plaza_full_argmax
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

python parakeet_native_spashl_ctc/decode.py --method greedy -- \
  --iter "$iteration" \
  --avg "$average" \
  --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$test_set" \
  --test-cut-manifest "$test_set=$manifest" \
  --suffix "$suffix" \
  --max-num-spks 8 \
  --factorized-token-speaker-output true \
  --speaker-weight 1.0 \
  --sort-strategy speech_time \
  --max-duration 1000 \
  --max-cuts 1 \
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
  --average-out "${stem}_tcpwer.json" \
  --per-reco-out "${stem}_tcpwer_per_reco.json"

echo "tcpWER result ($test_set):"
cat "${stem}_tcpwer.json"
