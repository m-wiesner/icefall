#!/usr/bin/env bash
#SBATCH --job-name=nsf-wh-shuf-ali
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

iter=${ITER:-96000}
exp_dir=exp/fastmss_yodas_en_whisper_large_v3_turbo_shuffle_a100x2_accum4_maxlr5e-6
dev=notsofar_sc_plaza_full
eval_set=notsofar_sc_plaza_test_full
suffix=shuffle_trained_latest_corpus30_remap_center20_mask_graphchunks
threshold_json="$exp_dir/align/blank_threshold_${dev}_${suffix}_iter${iter}.json"

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter "$iter" --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$dev $eval_set" \
  --test-cut-manifest "$dev=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz" \
  --test-cut-manifest "$eval_set=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz" \
  --suffix "$suffix" \
  --factorized-token-speaker-output false --factorized-alignment false \
  --ignore-speaker true --max-speaker-score-for-token-graph false \
  --oracle-num-spks 0 --max-num-spks 8 --sort-strategy speech_time \
  --speaker-weight 0 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-target-average-fraction 0.30 \
  --blank-frame-filter-calibration-set "$dev" \
  --blank-frame-filter-threshold-output "$threshold_json" \
  --blank-frame-filter-min-token-ratio 2 \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 2 \
  --token-speaker-mask-boundary-collar 5 \
  --blockwise-alignment true --blockwise-token-time-mask false \
  --alignment-diagnostics true --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false

for set_name in "$dev" "$eval_set"; do
  ali="$exp_dir/align/alignments_chkpt${iter}_avg1_${set_name}_collar32000_${suffix}.json"
  python local/ali_to_word_ali.py "$ali"
done

dev_hyp="$exp_dir/align/word_alignments_chkpt${iter}_avg1_${dev}_collar32000_${suffix}.json"
eval_hyp="$exp_dir/align/word_alignments_chkpt${iter}_avg1_${eval_set}_collar32000_${suffix}.json"
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
stats="${eval_hyp%.json}_calibrated_stats.txt"
python local/score_alignments.py --relabel --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-statistic median | tee "$stats"

