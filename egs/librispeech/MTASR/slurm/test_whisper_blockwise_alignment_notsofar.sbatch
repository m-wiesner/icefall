#!/usr/bin/env bash
#SBATCH --job-name=wh-nsf-blockwise-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
set_name=notsofar_sc_plaza_full
suffix=${SUFFIX:-blockwise_cc20_ctx5_collar2_keep30_smoke}
cut_id=${TEST_CUT_ID:-nsf-dev-MTG_30500-sc_plaza_0}
blank_args=(--blank-frame-filter-keep-fraction 0.30)
if [[ -n "${BLANK_THRESHOLD:-}" ]]; then
  blank_args=(--blank-frame-filter-threshold "$BLANK_THRESHOLD")
fi
python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter 82000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" \
  --test-cut-manifest "$set_name=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz" \
  --test-cut-id "$cut_id" --suffix "$suffix" \
  --blockwise-alignment true --blockwise-token-time-mask true \
  --factorized-token-speaker-output true --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar 32000 --allow-self-overlap false \
  --single-state-ctc-topo true --transcript-specific-dense-remap true \
  "${blank_args[@]}" \
  --blank-frame-filter-min-token-ratio 2.0 \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 --alignment-diagnostics true \
  --intersection-method factorized_viterbi \
  --max-duration 10000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false --skip-empty-alignments false

ali="$exp_dir/align/alignments_chkpt82000_avg1_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
python local/score_alignments.py --relabel \
  --hyp "$(dirname "$ali")/word_$(basename "$ali")" \
  --ref data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_normalized_word_alignments.json
