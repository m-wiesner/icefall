#!/usr/bin/env bash
#SBATCH --job-name=par-nsf-tgtspk-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
set_name=notsofar_sc_plaza_full
suffix=target_speaker_viterbi_smoke_allspkblank
python parakeet_native_spashl_ctc/align.py --method token -- \
  --iter 200000 --avg 5 --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --test-sets "$set_name" \
  --test-cut-manifest "$set_name=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz" \
  --test-cut-id nsf-dev-MTG_30500-sc_plaza_0 --suffix "$suffix" \
  --target-speaker-alignment true \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar 32000 --total-max-overlaps 10000000 \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-keep-fraction 0.30 \
  --blank-frame-filter-min-token-ratio 2.0 \
  --transcript-specific-dense-remap true \
  --token-speaker-mask-window-duration 2 \
  --token-speaker-mask-boundary-collar 5 \
  --alignment-chunk-duration 290 --alignment-chunk-overlap 10 \
  --alignment-chunk-context 0 --alignment-diagnostics true \
  --intersection-method factorized_viterbi \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt200000_avg5_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
python local/score_alignments.py --relabel \
  --hyp "$(dirname "$ali")/word_$(basename "$ali")" \
  --ref data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_normalized_word_alignments.json
