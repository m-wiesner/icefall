#!/usr/bin/env bash
#SBATCH --job-name=wh-nsf-block-r14
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
dev=notsofar_sc_plaza_full
test=notsofar_sc_plaza_test_full
suffix=blockwise_r14_cc20_ctx5_collar2_calblank_mean30
threshold_json="$exp_dir/align/blank_threshold_${dev}_${suffix}.json"

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter 82000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$dev $test" \
  --test-cut-manifest "$dev=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz" \
  --test-cut-manifest "$test=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz" \
  --suffix "$suffix" \
  --blockwise-alignment true --blockwise-token-time-mask true \
  --factorized-token-speaker-output true --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar 32000 --allow-self-overlap false \
  --single-state-ctc-topo true --transcript-specific-dense-remap true \
  --blank-frame-filter-target-average-fraction 0.30 \
  --blank-frame-filter-calibration-set "$dev" \
  --blank-frame-filter-threshold-output "$threshold_json" \
  --blank-frame-filter-min-token-ratio 2.0 \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 --alignment-diagnostics true \
  --intersection-method factorized_viterbi \
  --max-duration 10000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false --skip-empty-alignments false

echo "CALIBRATED_THRESHOLD_ARTIFACT $threshold_json"
cat "$threshold_json"
for set_name in "$dev" "$test"; do
  ali="$exp_dir/align/alignments_chkpt82000_avg1_${set_name}_collar32000_${suffix}.json"
  python local/ali_to_word_ali.py "$ali"
  word_ali="$(dirname "$ali")/word_$(basename "$ali")"
  if [[ "$set_name" == "$dev" ]]; then
    ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
  else
    ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
  fi
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref"
done
