#!/usr/bin/env bash
#SBATCH --job-name=par-s3-tok-vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
iter=200000
avg=5
set_name=synth3
collar=32000
max_cuts=${MAX_CUTS:-4}
suffix=parakeet_avg5_token_only_ignore_speaker_exact_batched_viterbi_batch${max_cuts}
reference=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json

python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --suffix "$suffix" \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --ignore-speaker true \
  --max-speaker-score-for-token-graph false \
  --max-num-spks 8 --oracle-num-spks 3 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method factorized_viterbi \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration 400 --max-cuts "$max_cuts" \
  --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false

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

echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
