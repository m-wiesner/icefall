#!/usr/bin/env bash
#SBATCH --job-name=nsf30-plaza-ali
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

collar=32000
suffix=clean_training_topo_allframes_dense

if [[ ${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  model=parakeet
  exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
  iter=200000
  avg=5
  set_name=notsofar_dev30_sc_plaza_0
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
  reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
  aligner=parakeet_native_spashl_ctc/align.py
  model_args=(
    --lang-dir data/lang_bpe_5000_librispeech_yodas_en
    --max-num-spks 8
    --downsample false --use-layer-norm false --frame-duration 0.02
  )
else
  model=wavlm
  exp_dir=exp/shuffle_large_length_paper
  iter=134000
  avg=1
  set_name=notsofar_dev30_sc_plaza_0_max4spk
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_max4spk_normalized_word_aligned.jsonl.gz
  reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_max4spk_normalized_word_alignments.json
  aligner=wavlm_spashl_ctc/align.py
  model_args=(
    --lang-dir data/lang_bpe_5000
    --max-num-spks 4
    --use-large true --downsample false --frame-duration 0.02
  )
fi

python "$aligner" --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --ignore-speaker false \
  --oracle-num-spks 0 \
  --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method dense \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration 80 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 \
  --normalize-loudness false \
  "${model_args[@]}"

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

echo "MODEL=$model"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$word_ali"
