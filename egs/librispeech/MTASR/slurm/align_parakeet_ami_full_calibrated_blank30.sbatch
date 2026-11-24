#!/usr/bin/env bash
#SBATCH --job-name=par-ami-calblank30
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
dev=ami_sdm_dev_full
test=ami_sdm_test_full
suffix=ami_full_viterbi_cf280_overlap10_mask5_calblank_mean30

python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter 200000 --avg 5 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$dev $test" \
  --test-cut-manifest "$dev=data/manifests/real_meetings/ami/ami_sdm_dev_full_normalized_word_aligned.jsonl.gz" \
  --test-cut-manifest "$test=data/manifests/real_meetings/ami/ami_sdm_test_full_normalized_word_aligned.jsonl.gz" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar 32000 --total-max-overlaps 10000000 \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-target-average-fraction 0.30 \
  --blank-frame-filter-calibration-set "$dev" \
  --blank-frame-filter-min-token-ratio 2.0 \
  --transcript-specific-dense-remap true \
  --token-speaker-mask-window-duration 2 \
  --token-speaker-mask-boundary-collar 5 \
  --alignment-chunk-duration 290 --alignment-chunk-overlap 10 \
  --alignment-chunk-context 0 --alignment-diagnostics true \
  --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments true --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false

for set_name in "$dev" "$test"; do
  ali="$exp_dir/align/alignments_chkpt200000_avg5_${set_name}_collar32000_${suffix}.json"
  python local/ali_to_word_ali.py "$ali"
  word_ali="$(dirname "$ali")/word_$(basename "$ali")"
  python local/score_alignments.py --relabel --hyp "$word_ali" \
    --ref "data/alignments/${set_name}_normalized_word_alignments.json"
done
