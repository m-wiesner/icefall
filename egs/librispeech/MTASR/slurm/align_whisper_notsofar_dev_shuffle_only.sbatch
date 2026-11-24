#!/usr/bin/env bash
#SBATCH --job-name=align-wh-nsf-shuf
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

exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
iter=${ITER:-76000}
avg=1
collar=${COLLAR:-32000}
suffix=shuffle_only_maxspk_pruned_w0005_active199999_200k
set_name=notsofar_sdm_dev30_word_aligned
manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized_word_aligned.jsonl.gz
ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized_word_alignments.json

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --test-cut-id nsf-dev-MTG_30500-sc_plaza_0-g0000-w0005 \
  --suffix "$suffix" \
  --factorized-token-speaker-output true \
  --ignore-speaker true --max-speaker-score-for-token-graph true --oracle-num-spks 0 \
  --max-num-spks 8 --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --intersection-method dense_pruned --search-beam 100 --beam-size 10 \
  --pruned-min-active-states 199999 --pruned-max-active-states 200000 \
  --max-duration 60 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
stats="$exp_dir/align/$(basename "${word_ali%.json}")_stats.txt"
python local/score_alignments.py --relabel --calibrate \
  --hyp "$word_ali" --ref "$ref" > "$stats"
cat "$stats"
