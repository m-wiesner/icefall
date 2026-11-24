#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-bpedur
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
collar=32000
manifest=data/manifests/cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz
reference=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json

case "$SLURM_ARRAY_TASK_ID" in
  0)
    # Isolate BPE decomposition: use max-four-character pieces while
    # retaining the former fixed 200 ms singleton-token duration.
    label=bpechars4_only
    max_piece_characters=4
    duration_args=(--single-token-fixed-duration 0.2)
    ;;
  1)
    # Isolate the duration change: retain canonical SentencePiece encoding
    # while using the new 80 ms-per-visible-character fallback.
    label=char_duration_only
    max_piece_characters=0
    duration_args=(--single-token-duration-per-character 0.08)
    ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

set_name="synth3_spkattr_wavlm_spashl72k_${label}"
suffix="speaker_attributed_training_topo_allframes_exact_hardseg100_${label}_spashl72k_20260923"

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02 --max-num-spks 8 \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 3 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin 1.0 \
  --alignment-bpe-max-piece-characters "$max_piece_characters" \
  "${duration_args[@]}" \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration 200 --max-cuts 2 \
  --num-buckets 1 --num-workers 4 --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
hyp="$(dirname "$ali")/word_$(basename "$ali")"

python local/score_alignment_mean_median.py --relabel \
  --hyp "$hyp" --ref "$reference" \
  --output "${hyp%.json}_uncalibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel --calibrate-half \
  --hyp "$hyp" --ref "$reference" \
  --output "${hyp%.json}_firsthalf_calibrated_secondhalf_mean_median.json"

echo "ABLATION=$label"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$hyp"
