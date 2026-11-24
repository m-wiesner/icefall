#!/usr/bin/env bash
#SBATCH --job-name=nsf-vit-corpus
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=3-8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

row=${SLURM_ARRAY_TASK_ID}
set_name=notsofar_sc_plaza_full
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json

remap=false
chunk_duration=0
chunk_context=0
mask_window=0
mask_collar=0
blockwise=false
case "$row" in
  3) label=single_corpus30 ;;
  4) label=single_corpus30_remap; remap=true ;;
  5) label=single_corpus30_remap_center20; remap=true; chunk_duration=30; chunk_context=5 ;;
  6) label=single_corpus30_remap_center20_mask; remap=true; chunk_duration=30; chunk_context=5; mask_window=2; mask_collar=5 ;;
  7) label=single_corpus30_remap_center20_mask_graphchunks; remap=true; chunk_duration=30; chunk_context=5; mask_window=2; mask_collar=5; blockwise=true ;;
  8) label=whisper_single_corpus30_remap_center20_mask_graphchunks; remap=true; chunk_duration=30; chunk_context=5; mask_window=2; mask_collar=5; blockwise=true ;;
esac

if [[ "$row" == 8 ]]; then
  exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
  entry=(python whisper_spashl_ctc/align.py --method token --whisper-model-id openai/whisper-large-v3-turbo --)
  checkpoint=(--iter 82000 --avg 1)
  lang_dir=data/lang_bpe_5000
else
  exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
  entry=(python parakeet_native_spashl_ctc/align.py --method token --)
  checkpoint=(--iter 200000 --avg 5)
  lang_dir=data/lang_bpe_5000_librispeech_yodas_en
fi

suffix=ablation_r${row}_${label}
threshold_json="$exp_dir/align/blank_threshold_${set_name}_${suffix}.json"

"${entry[@]}" \
  "${checkpoint[@]}" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir "$lang_dir" \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-target-average-fraction 0.30 \
  --blank-frame-filter-calibration-set "$set_name" \
  --blank-frame-filter-threshold-output "$threshold_json" \
  --blank-frame-filter-min-token-ratio 2 \
  --transcript-specific-dense-remap "$remap" \
  --alignment-chunk-duration "$chunk_duration" --alignment-chunk-context "$chunk_context" \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration "$mask_window" \
  --token-speaker-mask-boundary-collar "$mask_collar" \
  --blockwise-alignment "$blockwise" --blockwise-token-time-mask false \
  --alignment-diagnostics true --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false

echo "CALIBRATED_THRESHOLD_ARTIFACT $threshold_json"
cat "$threshold_json"
ali="$exp_dir/align/alignments_chkpt${checkpoint[1]}_avg${checkpoint[3]}_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$ref"
