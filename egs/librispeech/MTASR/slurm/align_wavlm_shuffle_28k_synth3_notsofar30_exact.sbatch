#!/usr/bin/env bash
#SBATCH --job-name=wavsh28k-tok-ali
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_shuffle_v100x4_accum4
iter=28000
avg=1
collar=32000
max_cuts=${MAX_CUTS:-4}
suffix=token_only_training_topo_allframes_exact_batched_viterbi_batch${max_cuts}

common_args=(
  --iter "$iter" --avg "$avg" --use-averaged-model false
  --exp-dir "$exp_dir"
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en
  --use-large true --downsample false --use-layer-norm false
  --frame-duration 0.02 --max-num-spks 8
  --factorized-token-speaker-output true
  --factorized-alignment false
  --ignore-speaker true
  --max-speaker-score-for-token-graph false
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0
  --collar "$collar" --total-max-overlaps 10000000
  --allow-self-overlap false
  --original-topo false --single-state-ctc-topo false
  --intersection-method factorized_viterbi
  --transcript-specific-dense-remap false
  --alignment-chunk-duration 0
  --token-speaker-mask-window-duration 0
  --blockwise-alignment false
  --beam-size 40 --max-cuts "$max_cuts"
  --num-buckets 1 --num-workers 4
  --normalize-loudness false
)

case "$SLURM_ARRAY_TASK_ID" in
  0)
    set_name=synth3_wavlm_shuffle_28k
    # The standard synth3 manifest is selected by this built-in test-set name.
    test_args=(--test-sets synth3)
    max_duration=400
    reference=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
    ;;
  1)
    set_name=notsofar_dev30_sc_plaza_0_token_only_wavlm_shuffle_28k
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    max_duration=160
    reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  2)
    set_name=notsofar_eval30_sc_plaza_0_token_only_wavlm_shuffle_28k
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    max_duration=160
    reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

# synth3 is a built-in name, but use a unique output name by supplying its
# standard manifest explicitly when available. Otherwise rename the outputs
# below after alignment.
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
  python wavlm_spashl_ctc/align.py --method token -- \
    "${common_args[@]}" --test-sets synth3 --suffix "$suffix" \
    --oracle-num-spks 3 --max-duration "$max_duration"
  source_set=synth3
else
  python wavlm_spashl_ctc/align.py --method token -- \
    "${common_args[@]}" "${test_args[@]}" --suffix "$suffix" \
    --oracle-num-spks 0 --max-duration "$max_duration"
  source_set=$set_name
fi

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${source_set}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

echo "SET=$source_set"
echo "REFERENCE=$reference"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"

