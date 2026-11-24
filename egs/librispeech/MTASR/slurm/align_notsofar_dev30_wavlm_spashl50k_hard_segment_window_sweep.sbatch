#!/usr/bin/env bash
#SBATCH --job-name=wl50-hardseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

variants=(nogate hardseg025 hardseg050 hardseg100 hardseg200)
margins=(none 0.25 0.50 1.00 2.00)
variant=${variants[$SLURM_ARRAY_TASK_ID]}
margin=${margins[$SLURM_ARRAY_TASK_ID]}

exp_dir=exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4
iter=50000
avg=1
collar=32000
set_name=notsofar_dev30_sc_plaza_0
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
suffix=training_topo_allframes_exact_${variant}_full_dev

window_args=()
if [[ $margin != none ]]; then
  window_args+=(--segment-time-window-margin "$margin")
fi

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --ignore-speaker false \
  --max-num-spks 8 --oracle-num-spks 0 \
  --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method factorized_viterbi \
  "${window_args[@]}" \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration 80 --max-cuts 4 \
  --num-buckets 1 --num-workers 4 \
  --normalize-loudness false \
  --use-large true --downsample false --use-layer-norm false \
  --frame-duration 0.02

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"

for calibration in uncalibrated calibrated; do
  stats="${word_ali%.json}_${calibration}_stats.txt"
  extra=()
  [[ $calibration == calibrated ]] && extra+=(--calibrate)
  python local/score_alignments.py --relabel "${extra[@]}" \
    --hyp "$word_ali" --ref "$reference" | tee "$stats"
done

echo "VARIANT=$variant"
echo "SEGMENT_TIME_WINDOW_MARGIN=$margin"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
