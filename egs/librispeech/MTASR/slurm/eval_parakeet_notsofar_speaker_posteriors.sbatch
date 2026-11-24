#!/usr/bin/env bash
#SBATCH --job-name=pk-nsf-post-der
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
if (( SLURM_ARRAY_TASK_ID == 0 )); then
  set_name=notsofar_sc_plaza_full
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
  split=dev
else
  set_name=notsofar_sc_plaza_test_full
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
  split=eval
fi
exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter 200000 --avg 5 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "row7_graph_free_speaker_posterior_${split}_sweep_smooth500ms" \
  --speaker-posterior-diarization true \
  --diarization-blank-thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
  --diarization-speaker-thresholds 0.35,0.4,0.45,0.5,0.55,0.6,0.7 \
  --diarization-collar 0.25 --diarization-smoothing 0.5 \
  --factorized-token-speaker-output true --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1 \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 --alignment-chunk-overlap 0 \
  --max-duration 10000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false
