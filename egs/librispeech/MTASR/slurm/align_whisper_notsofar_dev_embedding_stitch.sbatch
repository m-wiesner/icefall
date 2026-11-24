#!/usr/bin/env bash
#SBATCH --job-name=wh-nsf-emb-stitch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
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

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
split=${SPLIT:-dev}
case "$split" in
  dev)
    set_name=notsofar_sc_plaza_full
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
    ;;
  eval|test)
    set_name=notsofar_sc_plaza_test_full
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
    ;;
  *)
    echo "Unsupported SPLIT=$split (expected dev, eval, or test)" >&2
    exit 2
    ;;
esac
suffix=${SUFFIX:-token_only_center20_graphchunks_mask_corpus30_remap_ecapa_localdiar_b04_s045_smooth500_pyannote21_ahc}
oracle_num_speakers=${ORACLE_NUM_SPEAKERS:-true}
clustering_threshold=${CLUSTERING_THRESHOLD:-0.7153814381597874}
segmentation_source=${SEGMENTATION_SOURCE:-posterior}
token_gap=${TOKEN_GAP:-0.25}
timing_calibration=${TIMING_CALIBRATION:-}
speaker_recovery=${SPEAKER_RECOVERY:-embedding_cluster}
test_cut_args=()
if [[ -n "${TEST_CUT_ID:-}" ]]; then
  test_cut_args=(--test-cut-id "$TEST_CUT_ID")
fi

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter 82000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  "${test_cut_args[@]}" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker true --oracle-num-spks 0 --max-num-spks 8 \
  --blockwise-speaker-recovery "$speaker_recovery" \
  --speaker-embedding-min-seconds 0 \
  --speaker-embedding-max-seconds 0 \
  --speaker-clustering-threshold "$clustering_threshold" \
  --speaker-clustering-min-cluster-size 15 \
  --speaker-clustering-oracle-num-speakers "$oracle_num_speakers" \
  --speaker-stitching-blank-threshold 0.4 \
  --speaker-stitching-speaker-threshold 0.45 \
  --speaker-stitching-smoothing 0.5 \
  --speaker-stitching-segmentation-source "$segmentation_source" \
  --speaker-stitching-token-gap "$token_gap" \
  --speaker-stitching-timing-calibration "$timing_calibration" \
  --sort-strategy speech_time --speaker-weight 1 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-threshold 0.9968393445014954 \
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

if [[ "${RUN_SCORING:-true}" != true ]]; then
  exit 0
fi
ali="$exp_dir/align/alignments_chkpt82000_avg1_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
stats="${word_ali%.json}_stats.txt"
{
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref"
  echo WORD_BOUNDARY_DER
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$word_ali" --ref "$ref"
  echo SUPERVISION_BOUNDARY_DER
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$word_ali" --ref "$ref" --ref-cut-manifest "$manifest"
} | tee "$stats"

if [[ -n "${TIMING_CALIBRATION:-}" ]]; then
  {
    echo CALIBRATED
    python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref" \
      --calibration-offsets "$TIMING_CALIBRATION"
    echo WORD_BOUNDARY_DER
    python local/score_alignment_der.py --relabel --collar 0.25 \
      --hyp "$word_ali" --ref "$ref" \
      --calibration-offsets "$TIMING_CALIBRATION"
    echo SUPERVISION_BOUNDARY_DER
    python local/score_alignment_der.py --relabel --collar 0.25 \
      --hyp "$word_ali" --ref "$ref" --ref-cut-manifest "$manifest" \
      --calibration-offsets "$TIMING_CALIBRATION"
  } | tee "${word_ali%.json}_calibrated_stats.txt"
fi
