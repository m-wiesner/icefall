#!/usr/bin/env bash
#SBATCH --job-name=libri-yodas-bpe5k
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR/exp/slurm/%x-%j.out
#SBATCH --error=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR/exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
cd "$recipe_dir"
source activate.sh

lang_dir=data/lang_bpe_5000_librispeech_yodas_en
python local/train_bpe_model.py \
  --lang-dir "$lang_dir" \
  --transcript "$lang_dir/transcript_words.txt" \
  --vocab-size 5000
python local/validate_bpe_corpus.py \
  --model "$lang_dir/bpe.model" \
  --corpus "$lang_dir/transcript_words.txt" \
  --expected-vocab-size 5000
