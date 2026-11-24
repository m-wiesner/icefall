#!/usr/bin/env bash
#SBATCH --job-name=nsf-eval-vit-abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-8
#SBATCH --output=exp/slurm/%x-%A_%a.out
#SBATCH --error=exp/slurm/%x-%A_%a.err

set -euo pipefail
cd "${RECIPE_DIR:-$SLURM_SUBMIT_DIR}"
source activate.sh
export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

row=${SLURM_ARRAY_TASK_ID}
dev=notsofar_sc_plaza_full
eval_set=notsofar_sc_plaza_test_full
eval_manifest=data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz
dev_ref=data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json
eval_ref=data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json

single_state=false
blank_threshold=
remap=false
chunk_duration=0
chunk_context=0
mask_window=0
mask_collar=0
blockwise=false

case "$row" in
  1)
    label=training_topo_allframes
    dev_suffix=full_viterbi_stack_00_baseline_training_topo_allframes
    ;;
  2)
    label=single_allframes
    dev_suffix=full_viterbi_stack_01_single_state_allframes
    single_state=true
    ;;
  3)
    label=single_corpus30
    dev_suffix=ablation_r3_single_corpus30
    single_state=true
    blank_threshold=0.9998374581336975
    ;;
  4)
    label=single_corpus30_remap
    dev_suffix=ablation_r4_single_corpus30_remap
    single_state=true
    blank_threshold=0.9998374581336975
    remap=true
    ;;
  5)
    label=single_corpus30_remap_center20
    dev_suffix=ablation_r5_single_corpus30_remap_center20
    single_state=true
    blank_threshold=0.9998317360877991
    remap=true
    chunk_duration=30
    chunk_context=5
    ;;
  6)
    label=single_corpus30_remap_center20_mask
    dev_suffix=ablation_r6_single_corpus30_remap_center20_mask
    single_state=true
    blank_threshold=0.9998317360877991
    remap=true
    chunk_duration=30
    chunk_context=5
    mask_window=2
    mask_collar=5
    ;;
  7)
    label=single_corpus30_remap_center20_mask_graphchunks
    dev_suffix=ablation_r7_single_corpus30_remap_center20_mask_graphchunks
    single_state=true
    blank_threshold=0.9998317360877991
    remap=true
    chunk_duration=30
    chunk_context=5
    mask_window=2
    mask_collar=5
    blockwise=true
    ;;
  8)
    label=whisper_single_corpus30_remap_center20_mask_graphchunks
    dev_suffix=ablation_r8_whisper_single_corpus30_remap_center20_mask_graphchunks
    single_state=true
    blank_threshold=0.9968393445014954
    remap=true
    chunk_duration=30
    chunk_context=5
    mask_window=2
    mask_collar=5
    blockwise=true
    ;;
esac

eval_suffix=eval_calibrated_ablation_r${row}_${label}
if [[ "$row" == 8 ]]; then
  exp_dir=exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6
  entry=(python whisper_spashl_ctc/align.py --method token --whisper-model-id openai/whisper-large-v3-turbo --)
  checkpoint=(--iter 82000 --avg 1)
  checkpoint_tag=chkpt82000_avg1
  lang_dir=data/lang_bpe_5000
else
  exp_dir=exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
  entry=(python parakeet_native_spashl_ctc/align.py --method token --)
  checkpoint=(--iter 200000 --avg 5)
  checkpoint_tag=chkpt200000_avg5
  lang_dir=data/lang_bpe_5000_librispeech_yodas_en
fi

blank_args=()
if [[ -n "$blank_threshold" ]]; then
  blank_args=(--blank-frame-filter-threshold "$blank_threshold" --blank-frame-filter-min-token-ratio 2)
fi

"${entry[@]}" \
  "${checkpoint[@]}" --use-averaged-model false \
  --exp-dir "$exp_dir" --lang-dir "$lang_dir" \
  --test-sets "$eval_set" --test-cut-manifest "$eval_set=$eval_manifest" \
  --suffix "$eval_suffix" \
  --factorized-token-speaker-output true --factorized-alignment false \
  --ignore-speaker false --oracle-num-spks 0 --max-num-spks 8 \
  --sort-strategy speech_time --speaker-weight 1 --blank-weight 0 \
  --collar 32000 --total-max-overlaps 10000000 --allow-self-overlap false \
  --original-topo false --single-state-ctc-topo "$single_state" \
  "${blank_args[@]}" \
  --transcript-specific-dense-remap "$remap" \
  --alignment-chunk-duration "$chunk_duration" --alignment-chunk-context "$chunk_context" \
  --alignment-chunk-overlap 0 \
  --token-speaker-mask-window-duration "$mask_window" \
  --token-speaker-mask-boundary-collar "$mask_collar" \
  --blockwise-alignment "$blockwise" --blockwise-token-time-mask false \
  --alignment-diagnostics true --intersection-method factorized_viterbi \
  --search-beam 100 --beam-size 10 --max-states 25000000 \
  --skip-empty-alignments false --max-duration 10000 --max-cuts 1 \
  --num-buckets 1 --num-workers 4 --downsample false \
  --use-layer-norm false --frame-duration 0.02 --normalize-loudness false

ali="$exp_dir/align/alignments_${checkpoint_tag}_${eval_set}_collar32000_${eval_suffix}.json"
python local/ali_to_word_ali.py "$ali"
word_ali="$(dirname "$ali")/word_$(basename "$ali")"
dev_hyp="$exp_dir/align/word_alignments_${checkpoint_tag}_${dev}_collar32000_${dev_suffix}.json"
stats="${word_ali%.json}_calibrated_stats.txt"
python local/score_alignments.py --relabel --hyp "$word_ali" --ref "$eval_ref" \
  --calibration-hyp "$dev_hyp" --calibration-ref "$dev_ref" \
  --calibration-statistic median | tee "$stats"

