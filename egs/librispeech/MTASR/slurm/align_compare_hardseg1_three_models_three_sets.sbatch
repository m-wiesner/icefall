#!/usr/bin/env bash
#SBATCH --job-name=cmp3-hs1-ali
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_index=$((SLURM_ARRAY_TASK_ID / 3))
set_index=$((SLURM_ARRAY_TASK_ID % 3))
collar=32000
margin=1.00
suffix=training_topo_allframes_exact_hardseg100_compare20260921

case "$model_index" in
  0)
    model=parakeet_avg5
    exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
    aligner=parakeet_native_spashl_ctc/align.py
    iter=200000
    avg=5
    ignore_speaker=false
    model_args=(--downsample false --use-layer-norm false --frame-duration 0.02)
    ;;
  1)
    model=wavlm_spashl56k
    exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4
    aligner=wavlm_spashl_ctc/align.py
    iter=56000
    avg=1
    ignore_speaker=false
    model_args=(--use-large true --downsample false --use-layer-norm false --frame-duration 0.02)
    ;;
  2)
    model=wavlm_shuffle70k
    exp_dir=exp/fastmss_yodas_en_wavlm_large_shuffle_v100x4_accum4
    aligner=wavlm_spashl_ctc/align.py
    iter=70000
    avg=1
    ignore_speaker=true
    model_args=(--use-large true --downsample false --use-layer-norm false --frame-duration 0.02)
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

speaker_args=()
if [[ $ignore_speaker == true ]]; then
  speaker_args+=(--max-speaker-score-for-token-graph false)
fi

python "$aligner" --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  "${test_args[@]}" --suffix "$suffix" \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --ignore-speaker "$ignore_speaker" "${speaker_args[@]}" \
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
  --beam-size 40 --max-duration "$max_duration" --max-cuts 2 \
  --num-buckets 1 --num-workers 4 \
  --normalize-loudness false \
  "${model_args[@]}"

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

echo "MODEL=$model"
echo "SET=$set_name"
echo "SEGMENT_TIME_WINDOW_MARGIN=$margin seconds on each side"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
