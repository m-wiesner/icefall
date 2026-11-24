#!/usr/bin/env bash
#SBATCH --job-name=wl-tsfifo-der
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail

cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

export TORCH_HOME=${TORCH_HOME:-/expscratch/mwiesner/shuffle/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$SLURM_ARRAY_TASK_ID" in
  0)
    set_name=notsofar_sc_plaza_dev_full_teacher_fifo60k
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ;;
  1)
    set_name=notsofar_sc_plaza_eval_full_teacher_fifo60k
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
    ;;
  2)
    set_name=ami_sdm_dev_full_teacher_fifo60k
    manifest=data/manifests/real_meetings/ami/ami_sdm_dev_full_normalized_word_aligned.jsonl.gz
    ;;
  3)
    set_name=ami_sdm_test_full_teacher_fifo60k
    manifest=data/manifests/real_meetings/ami/ami_sdm_test_full_normalized_word_aligned.jsonl.gz
    ;;
  *)
    echo "Unexpected array index: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

exp_dir=exp/fastmss_yodas_en_wavlm_large_teacher_token_fifo_spashl_factorized_v100x4_accum4/pinned
teacher_checkpoint=exp/teacher_snapshots/fastmss_yodas_en_wavlm_large_shuffle_v100x4_accum4/checkpoint-32000.pt

python wavlm_spashl_ctc/align.py --method token -- \
  --iter 60000 --avg 1 --use-averaged-model false \
  --exp-dir "$exp_dir" \
  --frozen-token-teacher-checkpoint "$teacher_checkpoint" \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix posterior_diarization_true_oracle_chunk_hungarian_threshold_smoothing_sweep \
  --speaker-posterior-diarization true \
  --diarization-speaker-mapping oracle_chunk \
  --diarization-blank-thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --diarization-speaker-thresholds 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6 \
  --diarization-smoothing-values 0,0.25,0.5 \
  --diarization-collar 0.25 \
  --use-large true --downsample false --use-layer-norm false \
  --factorized-token-speaker-output true --max-num-spks 8 \
  --sort-strategy start_time --speaker-weight 1 \
  --alignment-chunk-duration 30 --alignment-chunk-context 5 \
  --alignment-chunk-overlap 0 \
  --max-duration 10000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --frame-duration 0.02 --normalize-loudness false
