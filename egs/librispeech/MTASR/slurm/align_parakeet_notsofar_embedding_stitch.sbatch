#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-diar
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
split=${SPLIT:-dev}
case "$split" in
  dev)
    set_name=notsofar_sc_plaza_full
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json ;;
  eval|test)
    set_name=notsofar_sc_plaza_test_full
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json ;;
  *) echo "Unsupported SPLIT=$split" >&2; exit 2 ;;
esac
set_name=${SET_NAME:-$set_name}
manifest=${MANIFEST:-$manifest}

suffix=${SUFFIX:?Set SUFFIX}
oracle_num_speakers=${ORACLE_NUM_SPEAKERS:-true}
clustering_threshold=${CLUSTERING_THRESHOLD:-0.85}
segmentation_source=${SEGMENTATION_SOURCE:-posterior}
speaker_recovery=${SPEAKER_RECOVERY:-embedding_cluster}
blank_threshold=${STITCHING_BLANK_THRESHOLD:-0.7}
speaker_threshold=${STITCHING_SPEAKER_THRESHOLD:-0.55}
token_gap=${TOKEN_GAP:-0.25}
token_speaker_confidence=${TOKEN_SPEAKER_CONFIDENCE:-0.75}
token_neighbors=${TOKEN_NEIGHBORS:-2}
vbx_min_clean_seconds=${VBX_MIN_CLEAN_SECONDS:-1.44}
timing_calibration=${TIMING_CALIBRATION:-}
calibration_arg=()
alignment_calibration_arg=()
if [[ -n "$timing_calibration" ]]; then
  calibration_arg=(--calibration-offsets "$timing_calibration")
  alignment_calibration_arg=(--speaker-stitching-timing-calibration "$timing_calibration")
fi

python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter 200000 --avg 5 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker true --oracle-num-spks 0 --max-num-spks 8 \
  --blockwise-speaker-recovery "$speaker_recovery" \
  --speaker-embedding-min-seconds 0 --speaker-embedding-max-seconds 0 \
  --speaker-clustering-threshold "$clustering_threshold" \
  --speaker-clustering-min-cluster-size 15 \
  --speaker-clustering-oracle-num-speakers "$oracle_num_speakers" \
  --speaker-stitching-blank-threshold "$blank_threshold" \
  --speaker-stitching-speaker-threshold "$speaker_threshold" \
  --speaker-stitching-smoothing 0.5 \
  --speaker-stitching-segmentation-source "$segmentation_source" \
  --speaker-stitching-token-gap "$token_gap" \
  --speaker-stitching-token-neighbors "$token_neighbors" \
  --speaker-stitching-token-speaker-confidence "$token_speaker_confidence" \
  --speaker-vbx-min-clean-seconds "$vbx_min_clean_seconds" \
  "${alignment_calibration_arg[@]}" \
  --sort-strategy speech_time --speaker-weight 1 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-threshold 0.9998317360877991 \
  --blank-frame-filter-min-token-ratio 2 \
  --transcript-specific-dense-remap true \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration 2 --token-speaker-mask-boundary-collar 5 \
  --blockwise-alignment true --blockwise-token-time-mask false \
  --alignment-diagnostics true --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt200000_avg5_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
stats="${word_ali%.json}_stats.txt"
{
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref" "${calibration_arg[@]}"
  echo WORD_BOUNDARY_DER
  python local/score_alignment_der.py --relabel --collar 0.25 --hyp "$word_ali" --ref "$ref" "${calibration_arg[@]}"
  echo SUPERVISION_BOUNDARY_DER
  python local/score_alignment_der.py --relabel --collar 0.25 --hyp "$word_ali" --ref "$ref" --ref-cut-manifest "$manifest" "${calibration_arg[@]}"
} | tee "$stats"
