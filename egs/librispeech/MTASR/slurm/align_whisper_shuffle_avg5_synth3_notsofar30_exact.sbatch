#!/usr/bin/env bash
#SBATCH --job-name=whshavg5-tok-ali
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail

cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_whisper_large_v3_turbo_shuffle_a100x2_accum4_maxlr5e-6
iter=200000
avg=5
collar=32000
max_cuts=${MAX_CUTS:-2}
base_suffix=token_only_training_topo_allframes_exact_batched_viterbi_batch${max_cuts}
suffix=$base_suffix
alignment_chunk_duration=0
alignment_chunk_overlap=0
transcript_specific_dense_remap=false

common_args=(
  --iter "$iter" --avg "$avg" --use-averaged-model false
  --exp-dir "$exp_dir"
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en
  --factorized-token-speaker-output false
  --factorized-alignment false
  --ignore-speaker true
  --max-speaker-score-for-token-graph false
  --sort-strategy speech_time --speaker-weight 0.0 --blank-weight 0.0
  --collar "$collar" --total-max-overlaps 10000000
  --allow-self-overlap false
  --original-topo false --single-state-ctc-topo false
  --intersection-method factorized_viterbi
  --token-speaker-mask-window-duration 0
  --blockwise-alignment false
  --beam-size 40 --max-cuts "$max_cuts"
  --num-buckets 1 --num-workers 4
  --downsample false --use-layer-norm false --frame-duration 0.02
  --normalize-loudness false
)

case "$SLURM_ARRAY_TASK_ID" in
  0)
    set_name=synth3
    test_args=(--test-sets "$set_name")
    # Some original synth3 cuts exceed Whisper's hard 30-second input limit.
    # Keep each full cut/graph, but run its acoustic inference in contiguous,
    # non-overlapping 30-second pieces.  Compact transcript remapping is exact
    # here because the alignment graph cannot consume out-of-transcript tokens.
    max_cuts=1
    max_duration=60
    alignment_chunk_duration=30
    transcript_specific_dense_remap=true
    suffix=token_only_training_topo_allframes_exact_batched_viterbi_batch1_whisper_chunks30
    oracle_num_spks=3
    reference=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
    ;;
  1)
    set_name=notsofar_dev30_sc_plaza_0_token_only_whisper_shuffle_avg5
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    max_duration=60
    oracle_num_spks=0
    reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  2)
    set_name=notsofar_eval30_sc_plaza_0_token_only_whisper_shuffle_avg5
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    max_duration=60
    oracle_num_spks=0
    reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  "${common_args[@]}" "${test_args[@]}" --suffix "$suffix" \
  --oracle-num-spks "$oracle_num_spks" --max-num-spks 8 \
  --max-duration "$max_duration" --max-cuts "$max_cuts" \
  --alignment-chunk-duration "$alignment_chunk_duration" \
  --alignment-chunk-overlap "$alignment_chunk_overlap" \
  --transcript-specific-dense-remap "$transcript_specific_dense_remap"

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

echo "SET=$set_name"
echo "REFERENCE=$reference"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
