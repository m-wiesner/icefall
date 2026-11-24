#!/usr/bin/env bash
#SBATCH --job-name=wh-ami-block
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

mode=${MODE:?MODE must be attributed or token}
split=${SPLIT:?SPLIT must be dev or test}
if [[ "$mode" != attributed && "$mode" != token ]]; then exit 2; fi
if [[ "$split" == dev ]]; then
  set_name=ami_sdm_dev_full
  manifest=data/manifests/real_meetings/ami/ami_sdm_dev_full_normalized_word_aligned.jsonl.gz
  ref=data/alignments/ami_sdm_dev_full_normalized_word_alignments.json
elif [[ "$split" == test ]]; then
  set_name=ami_sdm_test_full
  manifest=data/manifests/real_meetings/ami/ami_sdm_test_full_normalized_word_aligned.jsonl.gz
  ref=data/alignments/ami_sdm_test_full_normalized_word_alignments.json
else
  exit 2
fi

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
suffix=ami_latest_blockwise_corpus30_center20_mask_${mode}
threshold_json="$exp_dir/align/blank_threshold_ami_sdm_dev_full_${suffix}.json"
if [[ "$mode" == token ]]; then ignore_speaker=true; else ignore_speaker=false; fi

blank_args=()
if [[ "$split" == dev ]]; then
  blank_args+=(
    --blank-frame-filter-target-average-fraction 0.30
    --blank-frame-filter-calibration-set "$set_name"
    --blank-frame-filter-threshold-output "$threshold_json"
  )
else
  blank_threshold=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["blank_probability_threshold"])' "$threshold_json")
  blank_args+=(--blank-frame-filter-threshold "$blank_threshold")
fi

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter 82000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker "$ignore_speaker" --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  "${blank_args[@]}" --blank-frame-filter-min-token-ratio 2 \
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
dev_hyp="$exp_dir/align/word_alignments_chkpt82000_avg1_ami_sdm_dev_full_collar32000_${suffix}.json"
dev_ref=data/alignments/ami_sdm_dev_full_normalized_word_alignments.json
{
  echo UNCALIBRATED
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref"
  python local/score_alignment_der.py --relabel --collar 0.25 --hyp "$word_ali" --ref "$ref"
  echo CALIBRATED
  python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref" \
    --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref"
  python local/score_alignment_der.py --relabel --collar 0.25 \
    --hyp "$word_ali" --ref "$ref" \
    --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref"
} | tee "$stats"
