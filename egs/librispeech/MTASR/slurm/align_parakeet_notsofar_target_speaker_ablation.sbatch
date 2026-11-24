#!/usr/bin/env bash
#SBATCH --job-name=par-nsf-tgtspk-abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-27%28
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The array is the Cartesian product of {dev,test} and the same fourteen
# configurations used in the previous full-recording alignment table.
config=$((SLURM_ARRAY_TASK_ID % 14 + 1))
split_index=$((SLURM_ARRAY_TASK_ID / 14))
if (( split_index == 0 )); then
  set_name=notsofar_sc_plaza_full
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
  reference=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
else
  set_name=notsofar_sc_plaza_test_full
  manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
  reference=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json
fi

original_topo=false
single_state=true
keep_fraction=none
compact=false
chunk_duration=0
chunk_overlap=0
chunk_context=0
mask_window=0
mask_collar=5

case "$config" in
  1) original_topo=true; single_state=false; tag=01_training_allframes ;;
  2) tag=02_1state_allframes ;;
  3) keep_fraction=0.30; tag=03_1state_keep030 ;;
  4) keep_fraction=0.30; compact=true; tag=04_keep030_compact ;;
  5) keep_fraction=0.30; compact=true; chunk_duration=25; chunk_overlap=5; tag=05_cf_core20_input25_nomask ;;
  6) keep_fraction=0.30; compact=true; chunk_duration=25; chunk_overlap=5; mask_window=2; tag=06_cf_core20_input25_mask5 ;;
  7) keep_fraction=0.30; compact=true; chunk_duration=30; chunk_context=5; mask_window=2; tag=07_cc_core20_input30_mask5 ;;
  8) keep_fraction=0.30; compact=true; chunk_duration=20; chunk_overlap=10; mask_window=2; tag=08_cf_core10_overlap10_mask5 ;;
  9) keep_fraction=0.30; compact=true; chunk_duration=30; chunk_overlap=10; mask_window=2; tag=09_cf_core20_overlap10_mask5 ;;
  10) keep_fraction=0.30; compact=true; chunk_duration=50; chunk_overlap=10; mask_window=2; tag=10_cf_core40_overlap10_mask5 ;;
  11) keep_fraction=0.30; compact=true; chunk_duration=70; chunk_overlap=10; mask_window=2; tag=11_cf_core60_overlap10_mask5 ;;
  12) keep_fraction=0.30; compact=true; chunk_duration=90; chunk_overlap=10; mask_window=2; tag=12_cf_core80_overlap10_mask5 ;;
  13) keep_fraction=0.30; compact=true; chunk_duration=170; chunk_overlap=10; mask_window=2; tag=13_cf_core160_overlap10_mask5 ;;
  14) keep_fraction=0.30; compact=true; chunk_duration=290; chunk_overlap=10; mask_window=2; tag=14_cf_core280_overlap10_mask5 ;;
esac

exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
run_tag=${RUN_TAG:-}
suffix=target_speaker_allspkblank_${tag}${run_tag}
args=(
  --iter 200000 --avg 5 --use-averaged-model false
  --exp-dir "$exp_dir" --lang-dir data/lang_bpe_5000_librispeech_yodas_en
  --test-sets "$set_name" --test-cut-manifest "$set_name=$manifest"
  --suffix "$suffix" --target-speaker-alignment true
  --factorized-token-speaker-output true --factorized-alignment false
  --ignore-speaker false --oracle-num-spks 0 --max-num-spks 8
  --sort-strategy speech_time --speaker-weight 1.0 --blank-weight 0.0
  --collar 32000 --total-max-overlaps 10000000
  --original-topo "$original_topo" --single-state-ctc-topo "$single_state"
  --blank-frame-filter-min-token-ratio 2.0
  --transcript-specific-dense-remap "$compact"
  --token-speaker-mask-window-duration "$mask_window"
  --token-speaker-mask-boundary-collar "$mask_collar"
  --alignment-chunk-duration "$chunk_duration"
  --alignment-chunk-overlap "$chunk_overlap"
  --alignment-chunk-context "$chunk_context"
  --alignment-diagnostics true --intersection-method factorized_viterbi
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1
  --num-buckets 1 --num-workers 4 --downsample false
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false
)
if [[ "$keep_fraction" != none ]]; then
  args+=(--blank-frame-filter-keep-fraction "$keep_fraction")
fi

python parakeet_native_spashl_ctc/align.py --method token -- "${args[@]}"

ali="$exp_dir/align/alignments_chkpt200000_avg5_${set_name}_collar32000_${suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
stats="${word_ali%.json}_stats.txt"
python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$reference" | tee "$stats"
