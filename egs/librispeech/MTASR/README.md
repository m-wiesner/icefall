# Shuffle CTC for multi-talker ASR

This directory reconstructs the paper's multi-talker ASR systems, beginning
with the representative WavLM Large, length-ordered shuffle-loss experiment.
Only source and small assets needed to understand and run the systems are
included; historical experiment directories and checkpoints are not copied.

## Reproduce the paper: five commands

This is the shortest supported path for someone who wants the original
WavLM-Large, speaker-attributed shuffle model. Run the commands from this
directory (`icefall-shuffle/egs/librispeech/MTASR`). Slurm commands are shown
for convenience, with direct Python equivalents where applicable.

### 1. Install

```bash
./install.sh
source activate.sh
```

`install.sh` installs uv locally when necessary, creates `.venv` from the
locked environment, installs this project, and runs an import smoke test. Run
`source activate.sh` again in each new shell.

### 2. Prepare the training data

```bash
python local/prepare_data.py \
  --librispeech-dir /path/to/SLR12 \
  --lm-dir /path/to/SLR11 \
  --num-jobs 16 \
  --bpe-mode preserved
```

This prepares the standard Lhotse manifests, speaker-specific shuffled
training manifests, and the paper's BPE-5000 language assets. It is resumable.
On another system, replace the two corpus paths. To download the corpora and
MFA alignments instead, see [Data preparation](#data-preparation).

For Synth decoding and forced-alignment evaluation, also install the supplied
synthetic manifests while relocating their audio paths:

```bash
python local/prepare_data.py \
  --stages synth validate \
  --librispeech-dir /path/to/SLR12 \
  --synth-manifests \
    /path/to/cuts_librispeech_dev_synth.jsonl.gz \
    /path/to/cuts_librispeech_dev_synth_2spk_8splices.jsonl.gz \
    /path/to/cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz \
  --bpe-mode preserved
```

### 3. Train the multi-talker model

```bash
sbatch slurm/train_shuffle_large_length_paper.sbatch
```

Without Slurm:

```bash
python local/run_experiment.py paper --max-duration 280
```

This trains the original fixed-collar, speaker-attributed WavLM-Large system
in `exp/shuffle_large_length_paper`. The default is one V100,
`max_duration=280`, checkpoints every 2,000 updates, and 200,000 updates. To
resume, for example, from update 88,000:

```bash
START_BATCH=88000 sbatch slurm/train_shuffle_large_length_paper.sbatch
```

### 4. Decode the multi-talker model

The simplest decoder is framewise argmax followed by MeetEval tcpWER scoring:

```bash
MODEL_KIND=wavlm EXP_DIR=exp/shuffle_large_length_paper \
ITER=88000 AVG=4 SUFFIX=paper_argmax \
  sbatch slurm/decode_synth_argmax.sbatch
```

Without Slurm:

```bash
python wavlm_spashl_ctc/decode.py --method greedy -- \
  --iter 88000 --avg 4 --use-averaged-model false \
  --exp-dir exp/shuffle_large_length_paper \
  --lang-dir data/lang_bpe_5000 --test-sets synth \
  --suffix paper_argmax --use-large true --downsample false \
  --factorized-token-speaker-output true --max-num-spks 4 \
  --speaker-weight 1.0 --sort-strategy speech_time
```

`AVG=4` averages checkpoints 88k, 86k, 84k, and 82k. Results are written to
`exp/shuffle_large_length_paper/decode`; the Slurm log prints tcpWER. See
[LibriSpeech 3-gram HLG decoding](#librispeech-3-gram-hlg-decoding) for
language-model decoding.

### 5. Align the multi-talker model

```bash
EXP_DIR=exp/shuffle_large_length_paper ITER=88000 AVG=4 \
  sbatch slurm/align_synth2_speech_length.sbatch
```

Without Slurm, run alignment and its two scoring stages directly:

```bash
python wavlm_spashl_ctc/align.py --method token -- \
  --iter 88000 --avg 4 --use-averaged-model false \
  --exp-dir exp/shuffle_large_length_paper \
  --lang-dir data/lang_bpe_5000 --test-sets synth2 \
  --suffix speech_length_beam40 --use-large true --downsample false \
  --frame-duration 0.02 --factorized-token-speaker-output true \
  --max-num-spks 4 --oracle-num-spks 4 --sort-strategy speech_time \
  --speaker-weight 1.0 --blank-weight 0.0 --collar 32000 \
  --total-max-overlaps 10000000 --beam-size 40 \
  --max-duration 400 --num-workers 5 --normalize-loudness false

ALI=exp/shuffle_large_length_paper/align/alignments_chkpt88000_avg4_synth2_collar32000_speech_length_beam40.json
python local/ali_to_word_ali.py "$ALI"
python local/score_alignments.py --relabel --calibrate \
  --hyp "$(dirname "$ALI")/word_$(basename "$ALI")" \
  --ref data/alignments/alignments_dev_synth_2spk_8splices_speech_length.json
```

This aligns the 2-speaker, 8-splice Synth set with the fixed 2-second graph,
converts BPE alignments to words, and reports the paper-style timing and order
metrics. Outputs are under `exp/shuffle_large_length_paper/align`; the final
file ends in `_speech_length_stats.txt`.

That is the complete basic workflow. The rest of this README describes the
token-only loss, Brownian constraints, alternative encoders and decoders,
data provenance, resume behavior, and the alignment metrics in more detail.

The three principal implementations are:

- `wavlm_spashl_ctc`: speaker-attributed shuffle CTC. This is the canonical
  implementation used by the reconstructed `shuffle_large_length` experiment.
- `wavlm_shuffle_ctc`: token-shuffle CTC without the factored speaker-
  attribution output. It also includes `decode_nospk.py`.
- `wavlm_sdctc`: speaker-distinguished CTC comparison system.

## Reusable shuffle loss

The first standalone layer is available as `shuffle_ctc`. It has no dependency
on Lhotse, WavLM, SentencePiece, or the recipe data structures; it requires
PyTorch and k2. `ShuffleCTCLoss` follows the usual PyTorch criterion pattern:

```python
from shuffle_ctc import ShuffleCTCLoss

criterion = ShuffleCTCLoss(
    reduction="mean",
    output_beam=20,
    zero_infinity=True,
)
loss = criterion(
    log_probs,       # (batch, time, classes)
    shuffle_graphs,  # one k2 target FSA per batch item
    input_lengths,   # (batch,)
)
```

A linear CTC target graph gives ordinary graph-based CTC; a target graph that
accepts several legal token orders gives shuffle CTC. This keeps the numerical
loss independent of how an application obtains transcripts, token timing,
speaker labels, or partial-order constraints. The WavLM speaker-attributed and
token-only trainers both use this public criterion.

The next extraction step is a tokenizer-independent shuffle-graph builder that
accepts integer token sequences and optional timing intervals. Until that step
is complete, the recipe's graph compilers remain the adapter between its
SentencePiece/Lhotse batches and the standalone criterion.

Each implementation has its own flat research-code module namespace. Run an
entry point by filename from this directory so its sibling modules take
precedence, e.g. `python wavlm_sdctc/train.py --help`.

### Optional factorized CUDA intersection

`shuffle_ctc.factorized_graph_ctc_loss` avoids materializing the full
token-by-speaker score tensor. It looks up token and speaker columns and adds
their scores inside k2's dense-intersection CUDA kernel. The ordinary
`graph_ctc_loss` remains available without compiling this extension.

The extension is intentionally a narrow, local Ninja build: it links against
the k2 wheel installed by `install.sh` and compiles only the binding and the
modified intersection source. Obtain the exact matching source headers and
the header-only moderngpu dependency, then apply the checked-in patch:

```bash
mkdir -p /path/to/third_party
git clone https://github.com/k2-fsa/k2 /path/to/third_party/k2
git -C /path/to/third_party/k2 checkout 0a09f67e1ae9c4f90ea61a161633baacdd5b5af8
git -C /path/to/third_party/k2 apply \
  "$PWD/shuffle_ctc/csrc/k2-factorized-intersect.patch"

git clone https://github.com/moderngpu/moderngpu \
  /path/to/third_party/moderngpu
git -C /path/to/third_party/moderngpu checkout \
  8ec9ac0de8672de7217d014917eedec5317f75f3

export K2_SOURCE_DIR=/path/to/third_party/k2
export MODERNGPU_SOURCE_DIR=/path/to/third_party/moderngpu
export CUDA_HOME=/path/to/cuda-11.8-toolkit
export TORCH_CUDA_ARCH_LIST='7.0;8.0'
python tools/build_factorized_k2_extension.py
```

This requires the CUDA 11.8 compiler and Ninja, but does not rebuild or replace
the installed k2 wheel. Architectures 7.0 (V100) and 8.0 (A100) are the
default. Enable it in the shared WavLM training loop with
`--factorized-ctc-loss true`; the model then returns separate token and speaker
log probabilities instead of constructing joint logits.

Start with:

```bash
cd egs/librispeech/MTASR
./install.sh
source activate.sh
python smoke_test.py
```

The checked-in `pyproject.toml` and `uv.lock` define a Linux x86-64,
Python-3.11 environment using PyTorch 2.6.0 and the matching k2 CUDA 11.8
wheel. `uv sync --locked` recreates the exact resolved environment after the
initial lock file has been generated. CUDA wheels can be imported on a login
node without a GPU; actual training requires a CUDA-capable compute node and a
driver compatible with CUDA 11.8.

`install.sh` uses an existing uv or installs uv under `.tools/`, runs the
locked sync, and executes the import smoke test. It does not modify global
shell configuration. `activate.sh` adds the recipe, its legacy flat module
directory, and this Icefall checkout to `PYTHONPATH`.

On a cluster where the repository is on a filesystem that does not implement
file locks or atomic directory replacement reliably, put the environment and
cache on node-local storage:

```bash
export UV_PROJECT_ENVIRONMENT="${TMPDIR}/shuffle-ctc-venv"
export UV_CACHE_DIR="${TMPDIR}/shuffle-ctc-uv-cache"
./install.sh
```

In that case, use `uv run --locked ...` or set `PATH` to the selected
`UV_PROJECT_ENVIRONMENT`; the default `activate.sh` expects `.venv`.

The shared exact BPE-5000 assets are under `data-lang-bpe-5000`; the installed
working copy is under `data/lang_bpe_5000`. Corpus download is intentionally
optional. Prepared Lhotse manifests belong under `data/manifests`.

## Parakeet encoder with SPaSHL heads

`parakeet_native_spashl_ctc/` treats Parakeet like the WavLM encoder: padded
16-kHz waveforms pass through Parakeet's feature extractor and pretrained
FastConformer encoder, and independent token and speaker output heads consume
the same final encoder states. The factorized SPaSHL distribution is identical
to the WavLM recipe's token/speaker configuration.

This path deliberately does not use Parakeet's TDT decoder, tokenizer, encoder
projector, intermediate-layer taps, temporal upsampling, recovery adapters, or
custom convolution normalization. The encoder remains at its native 80-ms
frame shift. Shuffle collars are still specified in 16-kHz waveform samples.

Run the native SPaSHL configuration with:

```bash
sbatch slurm/train_parakeet_native_spashl.sbatch
```

or invoke it directly using the same options as `wavlm_spashl_ctc/train.py`:

```bash
python parakeet_native_spashl_ctc/train.py \
  --parakeet-model-id nvidia/parakeet-tdt-0.6b-v3 \
  --factorized-token-speaker-output true \
  --speaker-branch true ...
```

Native-resolution decode and alignment wrappers set the frame duration to
80 ms automatically:

```bash
python parakeet_native_spashl_ctc/decode.py --method greedy -- --help
python parakeet_native_spashl_ctc/align.py --method token -- --help
```

The older `parakeet_spashl_ctc/` and `parakeet_shuffle_ctc/` experiments are
retained only as historical work and are not part of this recommended path.

## Data preparation

`local/prepare_data.py` provides staged preparation from either an existing
LibriSpeech installation or reproducible public downloads. WavLM consumes raw
waveforms, so this workflow does not precompute filterbank features.

With existing OpenSLR installations, provide their local paths:

```bash
source activate.sh
python local/prepare_data.py \
  --librispeech-dir /path/to/SLR12 \
  --lm-dir /path/to/SLR11 \
  --num-jobs 16 \
  --bpe-mode preserved
```

`--librispeech-dir` accepts either the SLR12 directory or its nested
`LibriSpeech` directory. The default stages create standard recording and
supervision manifests, waveform cuts for every partition, one trimmed training
manifest per speaker (deterministically shuffled with `--shuffle-seed`, default
`42`), install the exact paper BPE-5000 language assets, and
write `data/manifests/mtasr_preparation.json` with the resolved input paths and
preparation metadata, including the shuffle seed. Use the same seed to recreate
the exact per-speaker cut ordering.

For a fully downloaded copy, include the optional first stage:

```bash
python local/prepare_data.py \
  --stages download manifests speakers bpe validate \
  --download-dir /path/with/at-least-70GB-free \
  --num-jobs 16 \
  --bpe-mode preserved
```

This downloads all seven LibriSpeech partitions from OpenSLR SLR12, the
LibriSpeech lexicon and selected decoding LMs from SLR11, and the MFA word and
phone alignment archive from Zenodo record 2619474. The alignment archive is
verified against its published MD5 (`2bab567d0ace651a4ba254e813629f46`) and
extracted under the download directory. These MFA TextGrids are retained as a
prepared external resource; the current shuffle training graphs still use
utterance boundaries rather than importing MFA token times.

To use an existing alignment copy without downloading, add:

```bash
  --alignments-dir /path/to/extracted/librispeech_alignments
```

Existing synthetic overlap manifests can be installed while relocating their
embedded audio paths to the selected LibriSpeech copy:

```bash
python local/prepare_data.py \
  --stages synth validate \
  --librispeech-dir /path/to/SLR12 \
  --synth-manifests \
    /path/to/cuts_librispeech_dev_synth.jsonl.gz \
    /path/to/cuts_librispeech_dev_synth_2spk_8splices.jsonl.gz \
    /path/to/cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz \
  --bpe-mode preserved
```

The installed copies are written under `data/manifests`; the source manifests
are left unchanged. Validation opens every synthetic mixture and loads its
audio, catching missing or stale recording paths before training or alignment.

The paper used the preserved BPE model. To deliberately train a new unigram
model from the three LibriSpeech training partitions instead, use:

```bash
python local/prepare_data.py \
  --stages bpe validate \
  --librispeech-dir /path/to/LibriSpeech \
  --manifest-dir data/manifests \
  --lang-dir data/lang_bpe_5000_retrained \
  --bpe-mode train --vocab-size 5000
```

`--bpe-mode train` creates `transcript_words.txt`, the SentencePiece model and
`bpe.model`, which is sufficient for shuffle-CTC training and greedy decoding.
Use `--bpe-mode preserved` for exact paper reproduction and HLG
decoding. Individual stages are resumable and existing outputs are reused;
pass `--force` to regenerate them.

## Validated end-to-end workflow

This section provides more detail about the WavLM experiments validated in
this repository. Run every command from `egs/librispeech/MTASR` after
`source activate.sh`. Slurm launchers and scheduler-independent Python
commands are both provided.

### 1. Check the installation and data

```bash
./install.sh
source activate.sh
python smoke_test.py

# Use existing LibriSpeech and LM copies. This is resumable.
python local/prepare_data.py \
  --librispeech-dir /path/to/SLR12 \
  --lm-dir /path/to/SLR11 \
  --num-jobs 16 \
  --bpe-mode preserved
```

Alignment evaluation additionally requires these relocated synthetic
manifests under `data/manifests`:

```text
cuts_librispeech_dev_synth.jsonl.gz
cuts_librispeech_dev_synth_2spk_8splices.jsonl.gz
cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz
```

Install and validate external copies with the `--stages synth validate`
command in the preceding data-preparation section. The 2-speaker alignment
reference is
`data/alignments/alignments_dev_synth_2spk_8splices_speech_length.json`.
Despite the historical Python option name `speech_time`, this evaluation
orders speakers by their total speech duration (speech length).

### 2. Train a model

The four maintained WavLM configurations are:

| Preset | Objective | Constraint model | Default output directory |
|---|---|---|---|
| `paper` | speaker-attributed shuffle | fixed 4 s collar | `exp/shuffle_large_length_paper` |
| `brownian` | speaker-attributed shuffle | Brownian bridge, 0 s edges/4 s midpoint | `exp/shuffle_large_length_brownian` |
| `token-only` | token shuffle, no speaker output | fixed 4 s collar | `exp/shuffle_only_large_length` |
| `four-gpu-400k` | speaker-attributed shuffle | fixed 4 s collar | `exp/shuffle_large_length_4gpu_400k` |

Inspect or run the portable command:

```bash
python local/run_experiment.py paper --print-only
python local/run_experiment.py paper --max-duration 280
```

Equivalent Slurm launchers are:

```bash
sbatch slurm/train_shuffle_large_length_paper.sbatch
sbatch slurm/train_shuffle_large_length_brownian.sbatch
sbatch slurm/train_shuffle_only_large_length.sbatch
sbatch slurm/train_shuffle_large_length_4gpu_400k.sbatch
```

The direct Python equivalents are:

```bash
python local/run_experiment.py paper
python local/run_experiment.py brownian
python local/run_experiment.py token-only
python local/run_experiment.py four-gpu-400k
```

Override launcher defaults through environment variables, for example:

```bash
MAX_DURATION=260 sbatch slurm/train_shuffle_large_length_paper.sbatch
```

To resume at a saved update, set `START_BATCH` to the checkpoint number:

```bash
START_BATCH=72000 MAX_DURATION=260 \
  sbatch slurm/train_shuffle_large_length_paper.sbatch
```

Checkpoints are saved every 2,000 updates. TensorBoard event files and training
logs live below each experiment directory; Slurm stdout/stderr live in
`exp/slurm`:

```bash
tensorboard --logdir exp --port 6006
```

### 3. Select and average checkpoints

All decode/alignment entry points use Icefall's iteration-checkpoint selection.
For example,

```text
--iter 72000 --avg 4 --use-averaged-model false
```

loads and arithmetically averages `checkpoint-72000.pt`,
`checkpoint-70000.pt`, `checkpoint-68000.pt`, and
`checkpoint-66000.pt`. `--use-averaged-model false` is important here: these
are ordinary periodic checkpoints, not Icefall's separately stored averaged
model endpoints.

### 4. Greedy/argmax decoding and tcpWER

Decode the Synth set and score it with MeetEval's 5-second-collar tcpWER:

```bash
MODEL_KIND=wavlm \
EXP_DIR=exp/shuffle_large_length_paper \
ITER=72000 AVG=4 SUFFIX=argmax_avg4 \
  sbatch slurm/decode_synth_argmax.sbatch
```

Change `EXP_DIR` to decode the Brownian speaker-attributed model. The launcher
writes STM hypotheses, STM references, aggregate tcpWER JSON, and per-recording
tcpWER JSON below `$EXP_DIR/decode`. Its Slurm log prints the aggregate score.
The speaker-attributed launcher is not appropriate for `shuffle_only`; use the
token-only dispatcher and inspect its arguments with:

```bash
python wavlm_shuffle_ctc/decode.py --method greedy --method-help
```

### 5. HLG decoding with the LibriSpeech 3-gram LM

Build the graph once (this requires the preserved BPE language assets):

```bash
python local/prepare_hlg.py \
  --lm-dir /path/to/SLR11 \
  --lang-dir data/lang_bpe_5000 \
  --modified-topology \
  --output data/lang_bpe_5000/HLG_3_gram_1e_7_modified.pt
```

Then decode and score:

```bash
EXP_DIR=exp/shuffle_large_length_paper \
ITER=72000 AVG=4 SUFFIX=hlg_3gram_avg4 \
  sbatch slurm/decode_synth_hlg.sbatch
```

This uses the paper settings: the pruned LibriSpeech 3-gram LM and acoustic
and blank weights of 2.0. Outputs are written below `$EXP_DIR/decode`.

### 6. Forced alignment and paper-style scoring

The validated evaluation uses the 2-speaker, 8-splice synthetic set, a fixed
32,000-sample (2 s) alignment collar, beam 40, and the speech-length reference.
The Brownian model still uses a **fixed** graph at evaluation time; Brownian
only describes the constraints used while training that model.

Run the three directly comparable systems with checkpoint-72k/average-4:

```bash
# Original paper-like fixed-collar, speaker-attributed model
EXP_DIR=exp/shuffle_large_length_paper ITER=72000 AVG=4 \
  sbatch slurm/align_synth2_speech_length.sbatch

# Brownian-trained, speaker-attributed model
EXP_DIR=exp/shuffle_large_length_brownian ITER=72000 AVG=4 \
  sbatch slurm/align_synth2_speech_length.sbatch

# Fixed-collar token-only model
EXP_DIR=exp/shuffle_only_large_length ITER=72000 AVG=4 \
  sbatch slurm/align_synth2_shuffle_only_speech_length.sbatch
```

Each launcher performs all three stages: GPU token alignment,
SentencePiece-to-word conversion, and `local/score_alignments.py --relabel
--calibrate`. The token alignment JSON format is consistently
`[[cut_id, rows], ...]`, sorted by cut ID. Files are written under
`$EXP_DIR/align`; the final `*_speech_length_stats.txt` contains mean and
aggregate IoU, boundary error, failure rate, WDER, partition error, and
Kendall ordering statistics.

The validated checkpoint-72k/average-4 results are useful installation checks:

| Model | Mean IoU | IoU | Boundary error | Kendall tau |
|---|---:|---:|---:|---:|
| fixed speaker-attributed | 62.0 | 63.9 | 65 ms | 0.998403 |
| fixed token-only | 61.5 | 63.3 | 66 ms | 0.998362 |
| Brownian speaker-attributed | 60.7 | 62.6 | 67 ms | 0.998341 |

All three aligned all 28,240 reference words with a 0% cut failure rate.
These numbers validate the pipeline; they are not the paper's final selected
checkpoint results.

### Output map

```text
exp/<experiment>/
├── checkpoint-<update>.pt       training checkpoints
├── log/                         training logs
├── tensorboard/                 TensorBoard events (when enabled)
├── decode/                      STM hypotheses/references and tcpWER JSON
└── align/                       token/word alignments and statistics

exp/slurm/                       scheduler stdout and stderr
data/lang_bpe_5000/              BPE assets and prepared HLG
data/manifests/                  Lhotse manifests
data/alignments/                 alignment evaluation references
```

## Training presets

After preparation, the maintained configurations can run without Slurm or
checkout-specific absolute paths:

```bash
source activate.sh
python local/run_experiment.py paper --print-only
python local/run_experiment.py brownian
python local/run_experiment.py four-gpu-400k
python local/run_experiment.py token-only
```

Use `--print-only` to inspect the resolved command. Override portable settings
directly and pass trainer-specific options after `--`:

```bash
python local/run_experiment.py paper \
  --world-size 2 --max-duration 250 --total-steps 400000 \
  -- --save-every-n 1000
```

A scheduler may wrap this command with its normal GPU allocation mechanism;
the files under `slurm/` are examples rather than recipe requirements.

### Training from pre-built multi-talker cuts

The trainer can bypass on-the-fly `CutSpliceIterable` simulation and repeatedly
consume one or more existing CutSet manifests:

```bash
python wavlm_spashl_ctc/train.py \
  --train-cut-manifests shard1.jsonl.gz shard2.jsonl.gz \
  --interval-model brownian_bridge --edge-collar 0 --collar 64000 \
  --max-num-spks 8 --max-overlap 5 \
  ...
```

Every supplied finite shard is consumed once per pass before the next pass
begins. On resume, model/optimizer/scheduler state is restored but the fixed-cut
sampler starts a new pass; it does not invoke the `CutSpliceIterable`
curriculum-reconstruction policy.

For the FastMSS LibriSpeech transfer, two-minute source cuts are first grouped
at supervision gaps. A 0.5-second maximum pause produces 908,965 groups across
the four shards; 98.9% are at most 80 seconds long. An 80-second configurable
per-cut threshold removes the remaining long tail:

```bash
lhotse cut trim-to-supervision-groups --max-pause 0.5 \
  source-cuts.jsonl.gz groups-pause0.5.jsonl.gz
```

The ready-to-run four-GPU Brownian configuration is:

```bash
MAX_DURATION=180 MAX_TRAIN_CUT_DURATION=80 \
  sbatch slurm/train_fastmss_brownian.sbatch
```

`MAX_TRAIN_CUT_DURATION` rejects individual outliers; `MAX_DURATION` is the
dynamic sampler's pooled nonlinear batch-cost budget. The FastMSS preset uses
`--max-overlap 4` and retains `--total-max-overlaps 6000`.

The experiment uses WavLM Large, four speakers, up to ten utterance segments,
speech-duration (`speech_time`) ordering, and a 4 s shuffle collar represented
as 64,000 input waveform samples at 16 kHz. The collar is not expressed in
WavLM encoder frames. The experiment uses the custom Lhotse cut-splice sampler
preserved under `vendor/`.
The sampler still needs to be integrated cleanly with upstream Lhotse before
this recipe is ready for an Icefall pull request.

Training checkpoints use a distributional resume policy for the infinite
on-the-fly cut stream. They restore the model, optimizer, scheduler, scaler,
and the cut generator's compact curriculum counter, but deliberately discard
the old dynamic-bucketing buffer. The generator restarts at
`saved_generated_examples - sampler_buffer_size`; filling the new empty buffer
advances it back to approximately the saved curriculum position. This avoids
Lhotse's full-history sampler replay and avoids serializing thousands of mixed
cuts. Resumed batches are statistically comparable but are not expected to be
identical to the interrupted batches.

The paper configuration remains the default fixed-collar constraint model.
Training also supports an optional Brownian-bridge uncertainty envelope over
the uniformly estimated **token onset** times:

```bash
python wavlm_spashl_ctc/train.py \
  --interval-model brownian_bridge \
  --edge-collar 16000 \
  --collar 64000 \
  ...
```

Here `--edge-collar` is the pairwise ambiguity threshold near utterance
boundaries and `--collar` is its maximum at the utterance midpoint, both in
input waveform samples (the example is 1 s at the edges and 4 s at the
midpoint). Setting them equal reproduces fixed-collar constraints. Token pairs
whose uncertainty intervals overlap remain unordered; disjoint intervals
receive chronological precedence. The same options are available for
token-only shuffle training.

The locked runtime includes PyTorch/torchaudio, k2, Lhotse, SentencePiece,
pyloudnorm, MeetEval, SciPy, TensorBoard, and Icefall's import dependencies.
It has been tested by importing all three implementations and by running the
training and greedy-decoding `--help` interfaces for each. Speaker-attributed
shuffle additionally has HLG decoding and alignment smoke coverage.

Each model directory has the same public entry-point shape:

```bash
python wavlm_spashl_ctc/train.py --help
python wavlm_spashl_ctc/decode.py --help
python wavlm_spashl_ctc/decode.py --method hlg --method-help
python wavlm_spashl_ctc/align.py --method token --method-help
```

Arguments not consumed by the dispatcher are passed unchanged to the selected
algorithm. Use `--method-help` to see that algorithm's full CLI. The available
methods are:

- Speaker-attributed shuffle decoding: `greedy`, `hlg`.
- Speaker-attributed shuffle alignment: `graph`, `token`, `oracle-mono`.
- Token-only shuffle decoding: `greedy`, `hlg`, `no-speaker`.
- Token-only shuffle alignment: `graph`.
- SD-CTC decoding: `greedy`, `hlg`.
- SD-CTC alignment: `graph`.

For example, an actual greedy invocation is written as:

```bash
python wavlm_spashl_ctc/decode.py --method greedy \
  --iter 88000 --avg 4 --exp-dir /path/to/exp \
  --lang-dir data/lang_bpe_5000 --test-sets "synth ls_test_clean"
```

Preserved checkpoint-88k/average-4 tcpWER JSON files remain under
`provenance/greedy_tcpwer`; they reproduce the paper's 17.2 macro average.

The paper evaluation uses greedy timestamped STM output and MeetEval tcpWER
with a 5-second collar over Synth, LibriMix 2-clean/2-both/3-clean, and
LibriSpeech test-clean/test-other. The maintained training presets above expose
the reconstructed systems; the preserved 88k results record the historical
paper checkpoint selection.

### LibriSpeech 3-gram HLG decoding

The paper's target-speaker decoder used the OpenSLR SLR11
`3-gram.pruned.1e-7.arpa.gz` LM, the preserved 5000-piece BPE language
directory, k2's modified CTC topology, and acoustic and blank weights of 2.0.

Topology names are easy to confuse here. Training with
`--original-topo false` uses the recipe's `build_ctc_topo(...,
original=False)`: after emitting a token, a path must return through blank
before emitting another token. This is **not** standard CTC, but it is also
not identical to `k2.ctc_topo(modified=True)`, which contains additional token
loops/transitions. The exported historical decoder loads `HLG_modified.pt`,
so `--modified-topology` is the paper-reproduction setting. The new
`--training-topology` option constructs an HLG with the exact training
topology for a controlled topology-matched experiment.

Prepare the graph once with:

```bash
source activate.sh
python local/prepare_hlg.py \
  --lm-dir /path/to/SLR11 \
  --lang-dir data/lang_bpe_5000 \
  --modified-topology \
  --output data/lang_bpe_5000/HLG_3_gram_1e_7_modified.pt
```

This performs ARPA -> OpenFST text conversion with `kaldilm`, composes
`H o L o G` with k2, and writes:

- `data/lm/G_3_gram_1e_7.fst.txt` and `.pt`
- `data/lang_bpe_5000/HLG_3_gram_1e_7_modified.pt`
- adjacent JSON provenance describing the inputs and conversion command

Existing non-empty intermediates are reused; pass `--force` to rebuild them.
To build the topology-matched alternative instead of the historical paper
HLG, use a distinct output name:

```bash
python local/prepare_hlg.py \
  --lm-dir /path/to/SLR11 \
  --lang-dir data/lang_bpe_5000 \
  --training-topology \
  --output data/lang_bpe_5000/HLG_3_gram_1e_7_training_topology.pt
```

Decode the Synth set using the paper settings with:

```bash
python wavlm_spashl_ctc/decode.py --method hlg -- \
  --iter 88000 --avg 4 --use-averaged-model false \
  --exp-dir /path/to/exp --lang-dir data/lang_bpe_5000 \
  --hlg data/lang_bpe_5000/HLG_3_gram_1e_7_modified.pt \
  --test-sets synth --suffix hlg_3gram_1e_7_acwt2_bw2 \
  --use-large true --downsample false \
  --factorized-token-speaker-output true --max-num-spks 4 \
  --acoustic-weight 2.0 --blank-weight 2.0 \
  --sort-strategy speech_time
```

HLG is a decoding graph and is not the constrained alignment graph. The
alignment commands build utterance-specific transcript FSAs using the BPE
model and the chosen fixed/Brownian token-time constraints; they do not use
the 3-gram LM. For example, inspect those separate options with
`python wavlm_spashl_ctc/align.py --method graph --method-help` and
`--method token --method-help`.

## Provenance

- Historical Icefall work: `3f331fe77d056ac5b4ac29d9bbdedcaeaf77519b`
- Custom Lhotse splice branch: `0620d18a79fd3c2d34aab957ac5d19ca2d5ab2f5`
- Historical configuration: `wavlm_spashl_ctc/conf/shuffle_large_length.json`
- Paper decode selection: checkpoint 88k, average 4

The copied implementation is still research code. The next cleanup should
extract the shuffle loss into a tested module, remove recipe-local duplication,
and finish shared data preparation and tcpWER scoring interfaces.
