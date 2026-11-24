#!/usr/bin/env bash
#SBATCH --job-name=par-nsf-whole-center
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
set_name=notsofar_sc_plaza_full
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json

case "$SLURM_ARRAY_TASK_ID" in
  0)
    mode=whole
    whole_recording=true
    chunk_duration=0
    chunk_context=0
    ;;
  1)
    mode=center20_input30
    whole_recording=false
    chunk_duration=30
    chunk_context=5
    ;;
  *) exit 2 ;;
esac

suffix="token_only_recover_${mode}_graphchunks_mask_corpus30_remap"
threshold_json="$exp_dir/align/blank_threshold_${set_name}_${suffix}.json"

python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter 200000 --avg 5 --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker true --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-target-average-fraction 0.30 \
  --blank-frame-filter-calibration-set "$set_name" \
  --blank-frame-filter-threshold-output "$threshold_json" \
  --blank-frame-filter-min-token-ratio 2 \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration "$chunk_duration" \
  --alignment-chunk-context "$chunk_context" \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 2 \
  --token-speaker-mask-boundary-collar 5 \
  --blockwise-alignment true --blockwise-token-time-mask false \
  --blockwise-whole-recording-inference "$whole_recording" \
  --blockwise-graph-chunk-duration 20 \
  --alignment-diagnostics true --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers "${NUM_WORKERS:-4}" --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt200000_avg5_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
stats="${word_ali%.json}_stats.txt"
{
  echo UNCALIBRATED
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref"
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$word_ali" --ref "$ref"
  echo CALIBRATED
  python local/score_alignments.py --relabel --calibrate \
    --hyp "$word_ali" --ref "$ref"
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$word_ali" --ref "$ref" \
    --calibration-hyp "$word_ali" --calibration-ref "$ref" \
    --calibration-statistic median
} | tee "$stats"
