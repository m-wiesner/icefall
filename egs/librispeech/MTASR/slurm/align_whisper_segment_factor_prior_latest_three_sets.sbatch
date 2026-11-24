#!/usr/bin/env bash
#SBATCH --job-name=whseg-prior-ali
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail

recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
cd "$recipe_dir"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_whisper_large_v3_turbo_shuffle_segment_factor_softmask_a100x2_accum4_maxlr5e-6
iter=${ITER:-172000}
avg=${AVG:-1}
run_tag=${RUN_TAG:-segmentprior172k}
collar=32000
suffix=segment_factor_prior_training_topo_exact_viterbi
alignment_chunk_duration=0
transcript_specific_dense_remap=false
max_cuts=2
max_duration=60

common_args=(
  --iter "$iter" --avg "$avg" --use-averaged-model false
  --exp-dir "$exp_dir"
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en
  --factorized-token-speaker-output false
  --factorized-alignment false
  --ignore-speaker true
  --token-only-segment-factor-alignment true
  --segment-factor-collar 48000
  --segment-factor-decay-scale 16000
  --segment-factor-floor-prob 0.0001
  --sort-strategy speech_time
  --speaker-weight 0.0 --blank-weight 0.0
  --collar "$collar" --edge-collar 0 --interval-model brownian_bridge
  --total-max-overlaps 6000 --allow-self-overlap false
  --original-topo false --single-state-ctc-topo false
  --intersection-method factorized_viterbi
  --token-speaker-mask-window-duration 0
  --blockwise-alignment false
  --beam-size 10
  --num-buckets 1 --num-workers 4
  --downsample false --use-layer-norm false --frame-duration 0.02
  --normalize-loudness false
  --max-num-spks 8 --oracle-num-spks 0
)

case "$SLURM_ARRAY_TASK_ID" in
  0)
    set_name=synth3
    test_args=(--test-sets "$set_name")
    reference=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
    # The synthetic cuts may exceed Whisper's 30-second input window.
    alignment_chunk_duration=30
    transcript_specific_dense_remap=true
    max_cuts=1
    max_duration=60
    ;;
  1)
    set_name=notsofar_dev30_sc_plaza_0_token_only_whisper_${run_tag}
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  2)
    set_name=notsofar_eval30_sc_plaza_0_token_only_whisper_${run_tag}
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

if [[ -n ${TEST_CUT_ID:-} ]]; then
  test_args+=(--test-cut-id "$TEST_CUT_ID")
  suffix="${suffix}_smoke"
  max_cuts=1
fi

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  "${common_args[@]}" "${test_args[@]}" \
  --suffix "$suffix" \
  --max-duration "$max_duration" --max-cuts "$max_cuts" \
  --alignment-chunk-duration "$alignment_chunk_duration" \
  --alignment-chunk-overlap 0 \
  --transcript-specific-dense-remap "$transcript_specific_dense_remap"

tag=chkpt${iter}_avg${avg}
ali="$exp_dir/align/alignments_${tag}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

echo "SET=$set_name"
echo "REFERENCE=$reference"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
