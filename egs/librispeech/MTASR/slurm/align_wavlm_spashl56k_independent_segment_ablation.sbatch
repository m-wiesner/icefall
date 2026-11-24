#!/usr/bin/env bash
#SBATCH --job-name=w8-indseg-ali
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4
iter=56000
avg=1
collar=32000
margin=1.0

variant_index=$((SLURM_ARRAY_TASK_ID / 3))
set_index=$((SLURM_ARRAY_TASK_ID % 3))

case "$variant_index" in
  0)
    variant=known_segment_speaker
    ignore_speaker=false
    ;;
  1)
    variant=token_only
    ignore_speaker=true
    ;;
  *) exit 2 ;;
esac

case "$set_index" in
  0)
    set_name=synth3
    test_args=(--test-sets synth3)
    oracle_num_spks=3
    max_duration=200
    ;;
  1)
    set_name=notsofar_dev30_sc_plaza_0
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    oracle_num_spks=0
    max_duration=80
    ;;
  2)
    set_name=notsofar_eval30_sc_plaza_0
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    test_args=(--test-sets "$set_name" --test-cut-manifest "$set_name=$manifest")
    oracle_num_spks=0
    max_duration=80
    ;;
  *) exit 2 ;;
esac

suffix="independent_segment_${variant}_training_topo_allframes_hardseg100_20260921"

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  "${test_args[@]}" --suffix "$suffix" \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --independent-segment-alignment true \
  --ignore-speaker "$ignore_speaker" \
  --max-speaker-score-for-token-graph false \
  --max-num-spks 8 --oracle-num-spks "$oracle_num_spks" \
  --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin "$margin" \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration "$max_duration" --max-cuts 1 \
  --num-buckets 1 --num-workers 4 \
  --normalize-loudness false \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

echo "VARIANT=$variant"
echo "SET=$set_name"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"

