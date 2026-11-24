#!/usr/bin/env bash
#SBATCH --job-name=wh-nsf-block-c20
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
set_name=notsofar_sc_plaza_full
suffix=blockwise_r14_cc20_ctx5_collar20_calblank_mean30
threshold_json="$exp_dir/align/blank_threshold_${set_name}_${suffix}.json"

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter 82000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" \
  --test-cut-manifest "$set_name=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz" \
  --suffix "$suffix" \
  --blockwise-alignment true --blockwise-token-time-mask true \
  --blockwise-speaker-ordering training \
  --factorized-token-speaker-output true --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar 320000 --allow-self-overlap false \
  --single-state-ctc-topo true --transcript-specific-dense-remap true \
  --blank-frame-filter-target-average-fraction 0.30 \
  --blank-frame-filter-calibration-set "$set_name" \
  --blank-frame-filter-threshold-output "$threshold_json" \
  --blank-frame-filter-min-token-ratio 2.0 \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 --alignment-diagnostics true \
  --intersection-method factorized_viterbi \
  --max-duration 10000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false --skip-empty-alignments false

alignment="$exp_dir/align/alignments_chkpt82000_avg1_${set_name}_collar320000_${suffix}.json"
python local/ali_to_word_ali.py "$alignment"
python local/score_alignments.py --relabel \
  --hyp "$(dirname "$alignment")/word_$(basename "$alignment")" \
  --ref data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json

echo "CALIBRATED_THRESHOLD_ARTIFACT $threshold_json"
cat "$threshold_json"
