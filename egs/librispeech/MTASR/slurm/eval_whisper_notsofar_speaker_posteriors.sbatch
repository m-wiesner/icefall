#!/usr/bin/env bash
#SBATCH --job-name=wh-nsf-post-der
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
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
elif [[ "$split" == test || "$split" == eval ]]; then
  set_name=notsofar_sc_plaza_test_full
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
else
  echo "Unsupported SPLIT=$split" >&2
  exit 2
fi
smoothing=${SMOOTHING:-0}
suffix=${SUFFIX:-graph_free_speaker_posterior_sweep}
blank_thresholds=${BLANK_THRESHOLDS:-0.5,0.9,0.99,0.999}
speaker_thresholds=${SPEAKER_THRESHOLDS:-0.1,0.2,0.3,0.4}
if [[ "${FINE_SWEEP:-0}" == 1 ]]; then
  blank_thresholds=0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7
  speaker_thresholds=0.35,0.4,0.45,0.5,0.55,0.6,0.7
fi

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter 82000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" \
  --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --speaker-posterior-diarization true \
  --diarization-blank-thresholds "$blank_thresholds" \
  --diarization-speaker-thresholds "$speaker_thresholds" \
  --diarization-collar 0.25 \
  --diarization-smoothing "$smoothing" \
  --factorized-token-speaker-output true --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1 \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 \
  --max-duration 10000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false
