#!/usr/bin/env bash
#SBATCH --job-name=wh-nsf-full-cf20
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
iter=${ITER:-82000}
avg=1
collar=32000
set_name=${SET_NAME:?SET_NAME is required}
manifest=${MANIFEST:?MANIFEST is required}
reference=${REFERENCE:?REFERENCE is required}
suffix=${SUFFIX:-full_viterbi_cf20_overlap10_mask5}
ignore_speaker=${IGNORE_SPEAKER:-false}
max_speaker_score=${MAX_SPEAKER_SCORE_FOR_TOKEN_GRAPH:-false}

python whisper_spashl_ctc/align.py --method token \
  --whisper-model-id openai/whisper-large-v3-turbo -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest" \
  --suffix "$suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker "$ignore_speaker" \
  --max-speaker-score-for-token-graph "$max_speaker_score" \
  --oracle-num-spks 0 --max-num-spks 8 --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --original-topo false --single-state-ctc-topo true \
  --blank-frame-filter-keep-fraction 0.3 \
  --blank-frame-filter-min-token-ratio 2.0 \
  --transcript-specific-dense-remap true \
  --token-speaker-mask-window-duration 2 \
  --token-speaker-mask-boundary-collar 5 \
  --alignment-chunk-duration 30 \
  --alignment-chunk-overlap 10 \
  --alignment-chunk-context 0 \
  --alignment-diagnostics true \
  --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments true \
  --max-duration 2000 --max-cuts 1 --num-buckets 1 --num-workers 4 \
  --downsample false --use-layer-norm false --frame-duration 0.02 \
  --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_${set_name}_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
stats="$exp_dir/align/$(basename "${word_ali%.json}")_stats.txt"
python local/score_alignments.py --relabel \
  --hyp "$word_ali" --ref "$reference" > "$stats"
cat "$stats"
