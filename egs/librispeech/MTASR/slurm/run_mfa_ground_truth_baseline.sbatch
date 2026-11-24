#!/usr/bin/env bash
#SBATCH --job-name=mfa-gt-fa
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
baseline_dir=/expscratch/mwiesner/shuffle/baselines
mfa_prefix="$baseline_dir/tools/mfa"
cd "$recipe_dir"

case "$SLURM_ARRAY_TASK_ID" in
  0) name=dev_synth_3spk ;;
  1) name=notsofar_dev30 ;;
  2) name=notsofar_eval30 ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export PATH="$mfa_prefix/bin:$PATH"
# Concurrent MFA processes cannot share the same SQLite/extracted-model work
# directory.  Keep the downloaded model archives shared, but isolate mutable
# state by corpus/array task.
model_root="$baseline_dir/mfa_root"
export MFA_ROOT_DIR="$baseline_dir/mfa_work/$name"
data_dir="$baseline_dir/data/$name"
output_dir="$baseline_dir/mfa_json/$name"
result="$baseline_dir/results/mfa_${name}.json"
mkdir -p "$MFA_ROOT_DIR" "$output_dir" "$baseline_dir/results"

# Every .lab file contains the normalized ground-truth supervision transcript.
mfa align \
  "$data_dir/corpus" \
  "$model_root/pretrained_models/dictionary/english_mfa.dict" \
  "$model_root/pretrained_models/acoustic/english_mfa.zip" \
  "$output_dir" \
  --output_format json \
  --single_speaker \
  --num_jobs "$SLURM_CPUS_PER_TASK" \
  --clean \
  --overwrite \
  --no_use_postgres

source activate.sh
python local/convert_mfa_json_alignments.py \
  --metadata "$data_dir/metadata.json" \
  --mfa-output-dir "$output_dir" \
  --output "$result"
