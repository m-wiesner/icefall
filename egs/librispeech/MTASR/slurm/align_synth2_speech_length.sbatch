#!/usr/bin/env bash
#SBATCH --job-name=mtasr-align-synth2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exp_dir=${EXP_DIR:-exp/shuffle_large_length_brownian}
iter=${ITER:-72000}
avg=${AVG:-4}
collar=${COLLAR:-32000}
suffix=${SUFFIX:-speech_length_beam40}

python wavlm_spashl_ctc/align.py --method token -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --test-sets synth2 --suffix "$suffix" \
  --use-large true --downsample false --frame-duration 0.02 \
  --factorized-token-speaker-output true \
  --max-num-spks 4 --oracle-num-spks 4 \
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0 \
  --collar "$collar" --total-max-overlaps 10000000 \
  --beam-size 40 --max-duration 400 --num-workers 5 \
  --normalize-loudness false

ali="$exp_dir/align/alignments_chkpt${iter}_avg${avg}_synth2_collar${collar}_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
ref=data/alignments/alignments_dev_synth_2spk_8splices_speech_length.json
stats="$exp_dir/align/$(basename "${word_ali%.json}")_speech_length_stats.txt"
python local/score_alignments.py --relabel --calibrate \
  --hyp "$word_ali" --ref "$ref" > "$stats"
cat "$stats"
