# Reproducing the long-form multi-talker alignment paper

This is the supported entry point for reproducing the experiments in
*Long-form Speaker Attributed Forced Alignment with Shuffles*.  It separates
the paper configuration from the many exploratory Parakeet, Whisper,
diarization, and pruning experiments in the archive branch.

## What is fixed here

- WavLM Large, eight speaker slots, checkpoint 72,000.
- FastMSS and filtered English YODAS training data, sampled equally.
- A 5,000-unit unigram BPE model trained on normalized LibriSpeech-960 and
  YODAS transcripts.
- Four V100 GPUs, gradient accumulation 4, and an effective 2,880 seconds per
  optimizer update.
- The exact short-form and full-recording manifests, calibration protocol,
  long-form ablations, and paper table values.

The speaker-attributed checkpoint used for the paper is intentionally not
stored in Git.  A compact 1,282,524,827-byte inference checkpoint was exported
from the exact 5,096,412,766-byte training checkpoint and archived at the
location recorded in `artifacts.json`; both immutable SHA-256 values are
recorded. Put a downloaded copy at the recorded destination, or pass
`--checkpoint`. The language model files are small and versioned with this
branch. Raw AMI, NOTSOFAR, FastMSS, and YODAS audio must be obtained under
their respective licenses.

## Quick start

From `egs/librispeech/MTASR`:

```bash
./install.sh
source activate.sh

# Shows every required model, BPE, manifest, and reference file.
python repro/longform_alignment/reproduce.py check --hash-checkpoint

# Show the exact paper experiment matrix and expected tables.
python repro/longform_alignment/reproduce.py list

# Submit the short-form ablation and the full-recording jobs.
python repro/longform_alignment/reproduce.py submit shortform
python repro/longform_alignment/reproduce.py submit longform
```

Use `--dry-run` to print commands without submitting them.  On another Slurm
cluster, override the requested accelerator, for example
`--gres=gpu:a100:1`.  The scripts use paths relative to this recipe and accept
`RECIPE_DIR` and `EXP_DIR`; no username-specific path is required.

Build the manuscript after the experiments:

```bash
python repro/longform_alignment/reproduce.py paper
```

## Data preparation

The paper uses:

1. FastMSS: four LibriSpeech shards, grouped at pauses of at least 1 s and
   excluding groups longer than 80 s.
2. YODAS: English `doverlap` hypotheses filtered with judge score at least
   8.0 and DER at most 0.069, grouped across pauses up to 1 s into groups up to
   60 s with 2--8 speakers.
3. NOTSOFAR: the `sc_plaza_0` single distant, pre-mixed channel.
4. AMI: the SDM channel.

Relevant portable preparation programs are in `local/`:

- `build_librispeech_yodas_bpe_corpus.py`
- `rechunk_supervision_groups.py`
- `prepare_ami_notsofar.py`
- `prepare_word_alignment_eval.py`
- `prepare_synthetic_word_alignment_reference.py`

The YODAS filter itself is maintained in the companion `yodas_pseudo`
repository on branch `feature/mtasr-english-filtered-cutset`.

Set `YODAS_SHAR_DIR` before training.  The paper filter invocation was:

```bash
python build_filtered_cutset.py \
  --shar-dir /path/to/yodas_shar_doverlap_v2 \
  --hypothesis doverlap --vs precision \
  --out-dir /path/to/yodas_filtered_en \
  --min-judge-score 8.0 --max-der 0.069 \
  --lang en --min-lang-count 50
```

## Paper jobs

The orchestrator submits these versioned scripts:

- `slurm/train_fastmss_yodas_wavlm_large_spashl_factorized_v100x4_accum4.sbatch`
- `slurm/train_fastmss_yodas_wavlm_large_shuffle_v100x4_accum4.sbatch`
- `slurm/align_wavlm_spashl72k_shortform_time_mask_ablation.sbatch`
- `slurm/align_wavlm_spashl72k_notsofar_full_longform_ablation.sbatch`
- `slurm/align_wavlm_spashl72k_full_ami_notsofar_oracle.sbatch`
- `slurm/align_wavlm_spashl72k_longform_bpechars4.sbatch`

`expected/` contains the exact rounded values typeset in the paper.  These
are immutable reference values, not synthetic replacements for run outputs.
Run outputs remain under the ignored `exp/` directory.  The scoring scripts
produce per-recording means, calibrate NSF/AMI evaluation using development
offsets, and preserve uncalibrated outputs separately.

## Important provenance notes

The pinned speaker-attributed 72k checkpoint is available locally and its
hash is recorded.  The rolling token-only experiment no longer contains its
historical 72k training checkpoint; later checkpoints remain in the private
experiment directory.  Therefore the paper's token-only alignment row is
reproducible from the pinned speaker-attributed model by ignoring its speaker
branch, which is also how the alignment ablation is configured here.  This
avoids silently substituting a different token-only checkpoint.

Speaker attribution in the long-form table uses an oracle mapping from each
chunk-local speaker slot to the recording-global speaker identity, exactly as
stated in the manuscript.  It is not a deployable non-oracle stitching score.
