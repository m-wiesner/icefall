#!/usr/bin/env bash
#SBATCH --job-name=mtasr-decode-hlg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
cd /expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
source activate.sh

exp_dir=${EXP_DIR:-exp/shuffle_large_length_brownian}
iter=${ITER:-72000}
avg=${AVG:-4}
suffix=${SUFFIX:-hlg_3gram_1e_7_acwt2_bw2}
hlg=${HLG:-data/lang_bpe_5000/HLG_3_gram_1e_7_modified.pt}

python wavlm_spashl_ctc/decode.py --method hlg -- \
  --iter "$iter" --avg "$avg" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000 \
  --hlg "$hlg" \
  --test-sets synth --suffix "$suffix" \
  --use-large true --downsample false --use-layer-norm false \
  --factorized-token-speaker-output true --max-num-spks 4 \
  --acoustic-weight 2.0 --blank-weight 2.0 \
  --sort-strategy speech_time --max-duration 200 --num-workers 5

hyp="$exp_dir/decode/hyps_chkpt${iter}_avg${avg}_lm_scale_1.0_synth_${suffix}.stm"
ref="$exp_dir/decode/refs_chkpt${iter}_avg${avg}_lm_scale_1.0_synth_${suffix}.stm"
score="$exp_dir/decode/hyps_chkpt${iter}_avg${avg}_lm_scale_1.0_synth_${suffix}_tcpwer.json"
python -m meeteval.wer tcpwer \
  --reference "$ref" --hypothesis "$hyp" --collar 5 --average-out "$score"
