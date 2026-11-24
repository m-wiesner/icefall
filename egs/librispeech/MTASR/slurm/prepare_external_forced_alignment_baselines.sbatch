#!/usr/bin/env bash
#SBATCH --job-name=ext-fa-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
baseline_dir=/expscratch/mwiesner/shuffle/baselines
cd "$recipe_dir"
source activate.sh

case "$SLURM_ARRAY_TASK_ID" in
  0)
    name=dev_synth_3spk
    manifest=data/manifests/cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz
    reference=data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json
    ;;
  1)
    name=notsofar_dev30
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  2)
    name=notsofar_eval30
    manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz
    reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json
    ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

output_dir="$baseline_dir/data/$name"
mkdir -p "$output_dir"
python local/prepare_external_forced_alignment.py \
  --manifest "$manifest" \
  --reference "$reference" \
  --output-dir "$output_dir" \
  --margin 1.0
