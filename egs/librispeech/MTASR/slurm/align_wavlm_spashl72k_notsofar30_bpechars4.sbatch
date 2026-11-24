#!/usr/bin/env bash
#SBATCH --job-name=wlsp72-nsf30-b4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4/pinned
iter=72000
avg=1
collar=32000
dev_set=notsofar_dev30_sc_plaza_0_spkattr_wavlm_spashl72k_bpechars4
eval_set=notsofar_eval30_sc_plaza_0_spkattr_wavlm_spashl72k_bpechars4
dev_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
eval_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
suffix=speaker_attributed_training_topo_allframes_exact_hardseg100_bpechars4_spashl72k_20260923

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$dev_set $eval_set" \
  --test-cut-manifest "$dev_set=$dev_manifest" \
  --test-cut-manifest "$eval_set=$eval_manifest" \
  --suffix "$suffix" \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02 --max-num-spks 8 \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method factorized_viterbi \
  --segment-time-window-margin 1.0 \
  --alignment-bpe-max-piece-characters 4 \
  --single-token-duration-per-character 0.08 \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration 80 --max-cuts 2 \
  --num-buckets 1 --num-workers 4 --normalize-loudness false

alignment_path() {
  local set_name=$1
  echo "$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
}

dev_ali=$(alignment_path "$dev_set")
eval_ali=$(alignment_path "$eval_set")
python local/ali_to_word_ali.py "$dev_ali"
python local/ali_to_word_ali.py "$eval_ali"
dev_hyp="$(dirname "$dev_ali")/word_$(basename "$dev_ali")"
eval_hyp="$(dirname "$eval_ali")/word_$(basename "$eval_ali")"
offsets="$exp_dir/align/calibration_notsofar30_spkattr_bpechars4_spashl72k.json"

python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --output "${dev_hyp%.json}_uncalibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$dev_hyp" --ref "$dev_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-output "$offsets" \
  --output "${dev_hyp%.json}_full_dev_calibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --output "${eval_hyp%.json}_uncalibrated_mean_median.json"
python local/score_alignment_mean_median.py --relabel \
  --hyp "$eval_hyp" --ref "$eval_ref" \
  --calibration-offsets "$offsets" \
  --output "${eval_hyp%.json}_dev_calibrated_mean_median.json"

echo "DEV_WORD_ALIGNMENT=$dev_hyp"
echo "EVAL_WORD_ALIGNMENT=$eval_hyp"
echo "DEV_CALIBRATION=$offsets"
