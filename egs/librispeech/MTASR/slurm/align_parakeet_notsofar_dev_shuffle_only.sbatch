#!/usr/bin/env bash
#SBATCH --job-name=align-par-nsf-shuf
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

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
iter=${ITER:-136000}
avg=${AVERAGE:-1}
collar=${COLLAR:-128000}
suffix=${SUFFIX:-spk_attr_factorized_single_state_blank05_ratio2_w0004_collar8s}
single_state_topo=${SINGLE_STATE_TOPO:-true}
original_topo=${ORIGINAL_TOPO:-false}
factorized_alignment=${FACTORIZED_ALIGNMENT:-false}
ignore_speaker=${IGNORE_SPEAKER:-false}
max_speaker_score_for_token_graph=${MAX_SPEAKER_SCORE_FOR_TOKEN_GRAPH:-false}
intersection_method=${INTERSECTION_METHOD:-dense_pruned}
online_intersection_block_frames=${ONLINE_INTERSECTION_BLOCK_FRAMES:-500}
online_intersection_profile_only=${ONLINE_INTERSECTION_PROFILE_ONLY:-false}
beam_size=${BEAM_SIZE:-10}
search_beam=${SEARCH_BEAM:-100}
blank_filter_threshold=${BLANK_FILTER_THRESHOLD-0.5}
blank_filter_keep_fraction=${BLANK_FILTER_KEEP_FRACTION-}
blank_filter_min_token_ratio=${BLANK_FILTER_MIN_TOKEN_RATIO:-2.0}
alignment_chunk_duration=${ALIGNMENT_CHUNK_DURATION:-0}
alignment_chunk_overlap=${ALIGNMENT_CHUNK_OVERLAP:-5}
alignment_chunk_context=${ALIGNMENT_CHUNK_CONTEXT:-0}
transcript_specific_remap=${TRANSCRIPT_SPECIFIC_REMAP:-false}
token_speaker_mask_window=${TOKEN_SPEAKER_MASK_WINDOW:-0}
token_speaker_mask_collar=${TOKEN_SPEAKER_MASK_COLLAR:-2}
alignment_diagnostics=${ALIGNMENT_DIAGNOSTICS:-false}
diagnostic_uniform_scores=${DIAGNOSTIC_UNIFORM_SCORES:-false}
diagnostic_frame_delta=${DIAGNOSTIC_FRAME_DELTA:-0}
pruned_min_active_states=${PRUNED_MIN_ACTIVE_STATES:-199999}
pruned_max_active_states=${PRUNED_MAX_ACTIVE_STATES:-200000}
max_duration=${MAX_DURATION:-80}
max_states=${MAX_STATES:-25000000}
run_scoring=${RUN_SCORING:-true}
test_cut_id=${TEST_CUT_ID-nsf-dev-MTG_30503-sc_plaza_0-g0000-w0004}
test_cut_args=()
if [[ -n "$test_cut_id" ]]; then
  test_cut_args=(--test-cut-id "$test_cut_id")
fi
blank_filter_args=()
if [[ -n "$blank_filter_threshold" || -n "$blank_filter_keep_fraction" ]]; then
  blank_filter_args=(
    --blank-frame-filter-min-token-ratio "$blank_filter_min_token_ratio"
  )
  if [[ -n "$blank_filter_threshold" ]]; then
    blank_filter_args+=(--blank-frame-filter-threshold "$blank_filter_threshold")
  fi
  if [[ -n "$blank_filter_keep_fraction" ]]; then
    blank_filter_args+=(--blank-frame-filter-keep-fraction "$blank_filter_keep_fraction")
  fi
fi
set_name=${SET_NAME:-notsofar_sdm_dev30_word_aligned}
manifest=${MANIFEST:-data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized_word_aligned.jsonl.gz}
ref=${REFERENCE:-data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized_word_alignments.json}

python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  "${test_cut_args[@]}" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment "$factorized_alignment" \
  --ignore-speaker "$ignore_speaker" \
  --max-speaker-score-for-token-graph "$max_speaker_score_for_token_graph" \
  --oracle-num-spks 0 \
  --max-num-spks 8 --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --original-topo "$original_topo" \
  --single-state-ctc-topo "$single_state_topo" \
  --alignment-chunk-duration "$alignment_chunk_duration" \
  --alignment-chunk-overlap "$alignment_chunk_overlap" \
  --alignment-chunk-context "$alignment_chunk_context" \
  --transcript-specific-dense-remap "$transcript_specific_remap" \
  --token-speaker-mask-window-duration "$token_speaker_mask_window" \
  --token-speaker-mask-boundary-collar "$token_speaker_mask_collar" \
  --alignment-diagnostics "$alignment_diagnostics" \
  --diagnostic-uniform-scores "$diagnostic_uniform_scores" \
  --diagnostic-frame-delta "$diagnostic_frame_delta" \
  --intersection-method "$intersection_method" \
  --online-intersection-block-frames "$online_intersection_block_frames" \
  --online-intersection-profile-only "$online_intersection_profile_only" \
  --search-beam "$search_beam" --beam-size "$beam_size" \
  --max-states "$max_states" \
  --pruned-min-active-states "$pruned_min_active_states" \
  --pruned-max-active-states "$pruned_max_active_states" \
  "${blank_filter_args[@]}" \
  --skip-empty-alignments true \
  --max-duration "$max_duration" --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false

if [[ "$run_scoring" == true ]]; then
  ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
  python local/ali_to_word_ali.py "$ali"
  word_ali="$(dirname "$ali")/word_$(basename "$ali")"
  stats="$exp_dir/align/$(basename "${word_ali%.json}")_stats.txt"
  python local/score_alignments.py --relabel --calibrate \
    --hyp "$word_ali" --ref "$ref" > "$stats"
  cat "$stats"
fi
