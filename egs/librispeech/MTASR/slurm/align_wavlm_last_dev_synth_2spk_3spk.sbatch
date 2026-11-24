#!/usr/bin/env bash
#SBATCH --job-name=wav-last-synth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=2-3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

num_speakers=${SLURM_ARRAY_TASK_ID}
set_name=synth${num_speakers}
exp_dir=exp/shuffle_large_length_paper
iter=134000
avg=1
collar=32000
suffix=last_checkpoint_clean_training_topo_allframes_dense

# Clean paper-style alignment: full speaker-attributed graph, training topology,
# all frames/full vocabulary, and standard unpruned k2 dense intersection.
python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" --suffix "$suffix" \
  --use-large true --downsample false --frame-duration 0.02 \
  --factorized-token-speaker-output true \
  --factorized-alignment false \
  --ignore-speaker false \
  --max-num-spks 4 --oracle-num-spks "$num_speakers" \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --original-topo false --single-state-ctc-topo false \
  --intersection-method dense \
  --transcript-specific-dense-remap false \
  --alignment-chunk-duration 0 \
  --token-speaker-mask-window-duration 0 \
  --blockwise-alignment false \
  --beam-size 40 --max-duration 400 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 \
  --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
echo "ALIGNMENT_JSON=$ali"
echo "WORD_ALIGNMENT_JSON=$(dirname "$ali")/word_$(basename "$ali")"
