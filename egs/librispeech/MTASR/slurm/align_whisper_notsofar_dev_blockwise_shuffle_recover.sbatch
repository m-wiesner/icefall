#!/usr/bin/env bash
#SBATCH --job-name=wh-nsf-shuf-block
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

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
split=${SPLIT:-dev}
if [[ "$split" == dev ]]; then
  set_name=notsofar_sc_plaza_full
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
  ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
elif [[ "$split" == test || "$split" == eval ]]; then
  set_name=notsofar_sc_plaza_test_full
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
  ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
else
  echo "Unsupported SPLIT=$split" >&2
  exit 2
fi
suffix=ablation_r8_shuffle_recover_corpus30_center20_mask_graphchunks
# Reuse row 8's corpus-level threshold: acoustic inference and blank scores are identical.
blank_threshold=0.9968393445014954

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter 82000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker true --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-threshold "$blank_threshold" \
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

ali="$exp_dir/align/alignments_chkpt82000_avg1_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
stats="${word_ali%.json}_stats.txt"
if [[ "$split" == dev ]]; then
  calibration_hyp="$word_ali"
else
  calibration_hyp="$exp_dir/align/word_alignments_chkpt82000_avg1_notsofar_sc_plaza_full_collar32000_${suffix}.json"
fi
calibration_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
{
  echo UNCALIBRATED
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref"
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$word_ali" --ref "$ref"
  echo CALIBRATED
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref" \
    --calibration-hyp "$calibration_hyp" --calibration-ref "$calibration_ref" \
    --calibration-statistic median
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$word_ali" --ref "$ref" \
    --calibration-hyp "$calibration_hyp" --calibration-ref "$calibration_ref" \
    --calibration-statistic median
} | tee "$stats"
