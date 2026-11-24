#!/usr/bin/env bash
#SBATCH --job-name=w2v-gt-fa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
baseline_dir=/expscratch/mwiesner/shuffle/baselines
python_bin="$baseline_dir/tools/whisperx/bin/python"
cd "$recipe_dir"

case "$SLURM_ARRAY_TASK_ID" in
  0) name=dev_synth_3spk ;;
  1) name=notsofar_dev30 ;;
  2) name=notsofar_eval30 ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export TORCH_HOME="$baseline_dir/cache/torch"
data_dir="$baseline_dir/data/$name"
result="$baseline_dir/results/wav2vec2_base960h_${name}.json"
mkdir -p "$baseline_dir/results"

# This calls CTC forced_align with the ground-truth transcript as its target;
# it never performs unconstrained ASR decoding and never consumes Whisper text.
"$python_bin" local/align_wav2vec2_ctc_reference.py \
  --metadata "$data_dir/metadata.json" \
  --model-dir "$baseline_dir/models/whisperx" \
  --output "$result" \
  --device cuda \
  --max-batch-items 16 \
  --max-batch-audio-seconds 120
