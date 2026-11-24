# MTASR / Shuffle project handoff

Last updated: 2026-11-24

This is the durable context for continuing the multi-talker ASR, forced
alignment, diarization, and Shuffle/SpAShl work in a new Codex chat. Read this
file before changing code or launching jobs. The project accumulated many
similarly named experiments; use the supported paper release and its checked-in
expected tables as the source of truth whenever possible.

## 1. Current state

- Workspace: `/expscratch/mwiesner/shuffle`
- Main repository: `/expscratch/mwiesner/shuffle/icefall-shuffle`
- Recipe: `/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR`
- There are no active Slurm jobs as of this update.
- All relevant repositories are clean and their current branches have been
  pushed.
- Do not delete old experiment directories or checkpoints without explicit
  permission. Many historical comparisons refer to exact filenames.

### Repositories and branches

1. `icefall-shuffle`
   - Supported release: `paper/longform-multitalker-alignment`
   - HEAD: `deaa30e9` (`docs(mtasr): publish paper model artifact`)
   - Exploratory archive: `archive/mtasr-experiments-2026`
   - Archive HEAD: `a390b80d`
   - GitHub: `https://github.com/m-wiesner/icefall`
2. `third_party/k2`
   - Branch: `feature/shuffle-factorized-intersection`
   - HEAD: `354283f`
   - GitHub: `https://github.com/m-wiesner/k2`
3. `yodas_pseudo`
   - Branch: `feature/mtasr-english-filtered-cutset`
   - HEAD: `13579b0`
   - Upstream remote: `https://github.com/BUTSpeechFIT/yodas_pseudo`
4. `third_party/moderngpu`
   - Pinned at `8ec9ac0`; no local changes.

Branch purposes are documented in `BRANCHES.md`. The paper branch contains
only the supported long-form alignment release. Parakeet, Whisper, Zipformer,
diarization stitching, target-speaker, alternative-loss, and diagnostic work
is intentionally kept on `archive/mtasr-experiments-2026`.

## 2. Starting the environment

```bash
cd /expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
source activate.sh
```

For a fresh checkout:

```bash
./install.sh
source activate.sh
python smoke_test.py
```

The installation is locked by `pyproject.toml` and `uv.lock`. The local uv
binary is `/expscratch/mwiesner/shuffle/.tools/uv`. The recipe uses PyTorch
2.6, CUDA 11.8, and a matching k2 wheel.

## 3. Supported paper reproduction release

Start here:

```text
egs/librispeech/MTASR/repro/longform_alignment/README.md
```

Public entry point:

```text
https://github.com/m-wiesner/icefall/blob/paper/longform-multitalker-alignment/egs/librispeech/MTASR/repro/longform_alignment/README.md
```

Main commands, from the MTASR recipe directory:

```bash
python repro/longform_alignment/reproduce.py fetch-model
python repro/longform_alignment/reproduce.py check --hash-checkpoint
python repro/longform_alignment/reproduce.py list
python repro/longform_alignment/reproduce.py submit shortform --dry-run
python repro/longform_alignment/reproduce.py submit longform --dry-run
python repro/longform_alignment/reproduce.py paper
```

Important release files:

- `repro/longform_alignment/artifacts.json`: immutable model URL, sizes, and
  hashes.
- `repro/longform_alignment/experiments.json`: exact paper experiment matrix.
- `repro/longform_alignment/expected/*.csv`: authoritative paper numbers.
- `repro/longform_alignment/MODEL_CARD.md`: versioned Hugging Face model card.
- `paper/longform_alignment/source/`: complete recovered Overleaf source.
- `paper/longform_alignment/Longform_Multi_Talker_Alignment.pdf`: final PDF.

The cluster TeX installation lacks `multirow`, `shuffle`, `algorithm2e`, and
`stmaryrd`; the checked-in build program reports this cleanly. Use complete
TeX Live to rebuild the PDF.

## 4. Released WavLM paper model

Model repository:

```text
https://huggingface.co/mshwiesner/longform-multitalker-alignment
```

Artifact:

```text
wavlm8spk-checkpoint-72000.pt
size:   1,282,524,827 bytes
sha256: 9cb800a1c6584bd5bbfc14359dcf4f767e87f3ea9b45ca4369588221dbb5db21
```

The immutable weight revision is
`db306e6a127becd3a91aeec77b3aef560d005ec4`. The full model card is Hugging
Face commit `88605d1bbeb347156cc6b1c6d61c29b313f77f53`.

The compact artifact contains the model state and provenance needed for
checkpoint-72k, `avg=1` alignment/decoding. The original 5.096 GB training
checkpoint additionally contains optimizer, scheduler, scaler, and sampler
state. Its SHA-256 is
`78afc798a41b8133aee5df6bab6e4ad9c1eafe1c09b707bc191771ebd6bfde37`.

Model/training summary:

- WavLM Large, 16 kHz, 50 Hz output, eight speaker slots.
- Separate token and speaker heads with factorized CUDA Shuffle CTC loss.
- FastMSS plus filtered English YODAS, equal mixture weight.
- Combined 5k unigram BPE trained on normalized LibriSpeech-960 and YODAS.
- Four V100s, `max_duration=180` per GPU, accumulation 4.
- Base LR `5e-5`, frozen encoder 2,000 steps, head LR `5e-3`, configured for
  200k steps.
- Brownian-bridge interval model, 2 s collar, beam 10, duration/speech-time
  speaker ordering, maximum eight slots and four concurrent synthetic
  speakers.

The license used for the fine-tuned checkpoint is CC BY-SA 3.0. The official
Hugging Face WavLM card links CC BY-SA 3.0 while Microsoft's UniLM source tree
uses MIT; an upstream issue requesting clarification remains open. Preserve
both attributions and do not silently relicense the weights.

## 5. Custom factorized CUDA loss

The purpose is to avoid materializing joint token-by-speaker logits. For a
nonblank graph arc, the kernel adds the selected token and speaker log scores
on demand and returns separate arc maps so autograd reaches both factors.

Paper branch files:

- `shuffle_ctc/factorized.py`: Python/autograd wrapper.
- `shuffle_ctc/csrc/factorized_intersect_bindings.cu`: narrow PyTorch binding.
- `shuffle_ctc/csrc/k2-factorized-intersect.patch`: exact k2 patch.
- `tools/build_factorized_k2_extension.py`: local Ninja build.

The integrated k2 implementation is also browsable on
`m-wiesner/k2:feature/shuffle-factorized-intersection`. The paper recipe does
not require replacing the installed k2 wheel: it compiles a narrow local
extension against k2 commit `0a09f67` and moderngpu `8ec9ac0`.

The ordinary non-factorized loss remains available. Tests cover the reference
and factorized Viterbi paths; CUDA-only tests require a GPU allocation.

## 6. Data and manifests

### FastMSS

- Four LibriSpeech-derived training shards.
- Groups split at pauses of at least 1 s.
- Groups longer than 80 s excluded in the supported WavLM setup.
- Manifests are under `data/manifests/fastmss_librispeech/`.

### YODAS

- Raw Shar: `/expscratch/mwiesner/shuffle/yodas_shar_doverlap_v2`
- Helper repo: `/expscratch/mwiesner/shuffle/yodas_pseudo`
- Filtered English scheme used by training:
  `/expscratch/mwiesner/shuffle/yodas_shar_filtered/en_judge8_der0069_doverlap`
- Filter: English `doverlap`, judge score >= 8.0, DER <= 0.069.
- Grouping: gap <= 1 s, duration <= 60 s, 2--8 speakers.
- Audio is referenced through Shar; filtering/re-grouping should not copy or
  clip audio unnecessarily.

### NOTSOFAR

- Corpus root: `/export/common/data/corpora/ASR/NOTSOFAR`
- Evaluation uses only the single pre-mixed distant microphone `sc_plaza_0`.
- Short form: <=30 s manifests with word alignments.
- Long form: full recordings, also with word-alignment references.
- Main manifests are under
  `data/manifests/real_meetings/notsofar/`.
- `dev` calibrates timing offsets; apply those offsets to `eval/test`. Do not
  calibrate test independently when reporting comparable results.

### AMI

- Corpus root: `/export/common/data/corpora/amicorpus`
- Use SDM (single distant microphone), not arbitrary headset channels.
- Full dev/test manifests and references are under
  `data/manifests/real_meetings/ami/` and `data/alignments/`.

### SBCSAE

- Recording-style labels: `data/manifests/sbcsae/sbcsae_recording_styles.csv`
- Task/conversation <=8-speaker subsets:
  - `sbcsae_cuts_asr_aligned_tasks_max8spk.jsonl.gz`
  - `sbcsae_cuts_asr_aligned_conversations_max8spk.jsonl.gz`
- Normalized 30 s and 60 s conversation evaluation manifests also exist in
  `data/manifests/sbcsae/`.

### Synthetic evaluation

- Primary 3-speaker set:
  `data/manifests/cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz`
- Reference:
  `data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json`
- Calibration for this set uses first-half calibration and reports the second
  half. Do not compare it with metrics produced from a different split.

## 7. Alignment implementation and lessons

### Exact custom Viterbi

The current supported alignment uses the custom exact Viterbi implementation
in `shuffle_ctc/viterbi.py`, not k2 `intersect_dense`. A batched implementation
and blockwise/lazy graph implementation are also present.

Why this was necessary:

- k2 dense intersection could return an empty lattice or OOM on difficult
  long partial-order graphs even when a valid path existed.
- The experimental forward-pruned Viterbi used a purely local
  `max(candidate)-search_beam` cutoff. High-scoring states that could not reach
  the final state could prune all lower-scoring complete paths.
- `min_active` in that implementation was advisory rather than a hard floor.
- The exact Viterbi succeeded on cases where k2 and locally pruned variants
  failed. Do not reintroduce forward-only pruning without a coaccessibility or
  backward-beta/lookahead criterion.

### Time mask

The current time mask is a hard, segment-specific feasibility mask applied
during Viterbi: at each frame, only token/speaker labels belonging to reference
segments whose coarse time window (plus margin) includes the frame are
allowed. The paper uses a 1 s margin on each side.

This is related to, but more precise than, the older “local mask,” which built
a vocabulary/speaker set over fixed local windows. Keep these names distinct.

### One-state topology

The long-form alignment topology is a single state with symbol self-loops. It
removes the large expansion induced by the training CTC topology. The original
training topology is retained for short-form ablations but does not complete
on full recordings.

### Blank-frame removal

A first pass scores blank probability. A corpus-level threshold is selected to
retain a target average fraction of frames; it is not forced to retain exactly
the same fraction in every recording. Original frame indices are retained so
timestamps are restored correctly.

- Standard long-form ablation: retain 30% of frames.
- Final BPE-trim rows: retain 60% (remove 40%).
- Do not confuse “30% retained” with “30% removed.” This caused repeated
  experimental confusion.

### Transcript remapping and local masks

After blank removal, token and speaker output dimensions may be remapped to
the labels that can occur in the transcript/recording. A still more local mask
limits labels to those plausible around each time window. Token and speaker
normalization denominators must be preserved when comparing their relative
scores.

### Lazy graph chunks

The graph is compiled only for the time-local frontier rather than building a
full recording FSA. At a chunk boundary, reachable frontier states become the
next chunk's initial states with their accumulated Viterbi scores. New
candidate terminal/frontier states are selected using coarse segment timing
and the collar. Backpointers are retained across chunks so one global best path
is recovered; chunks are not aligned independently.

Audio chunking and graph chunking are separate:

- Long-form audio inference generally uses overlapping 30 s model inputs and
  retains the center 20 s output.
- Graph chunks can be smaller than or aligned differently from audio chunks.
- Speaker slot labels are local to an audio chunk. Speaker-attributed
  long-form results use a chunk-local-to-global oracle map. Token-only graph
  alignment avoids speaker labels during search but post-facto attribution
  still must solve the slot permutation across audio chunks.

### BPE trimming/decomposition

Reference BPE pieces are deterministically decomposed into shorter existing
BPE units, with maximum piece length four characters in the best configuration.
This improved short-form and final long-form timing. Single-token fallback
duration is character-proportional, 0.08 s per character. It is not ordinary
word-to-BPE composition: the key experiment further decomposes the tokenizer's
original reference pieces.

## 8. Metrics and reporting conventions

The checked-in CSV files are authoritative. Important conventions:

- Mean IoU and boundary error are averaged per recording/cut, not globally
  pooled over tokens, unless explicitly labeled otherwise.
- Boundary error uses both start and end boundaries.
- Normalized Kendall distance is lower-is-better and often reported as a
  percentage. Do not substitute the other Kendall implementation that was
  accidentally reported in several earlier comparisons.
- “Pooled IoU” aggregates intersections/unions over all tokens; “mean IoU” is
  the mean of per-token/per-cut contributions. The paper tables use the
  documented per-recording aggregation.
- RTFx is speed-up over real time (`audio duration / runtime`), so larger is
  faster. It includes neural inference, graph construction, and Viterbi, but
  excludes model/data-loader startup and corpus blank-threshold calibration.
- Peak CUDA allocation includes the neural network as well as graph/Viterbi
  tensors. It is not the total observed `nvidia-smi` process footprint.
- For NOTSOFAR and AMI eval/test, apply dev timing calibration.
- For `dev_synth_3spk`, calibrate on the first half and report the second half.

## 9. Authoritative paper results

Full tables are in `repro/longform_alignment/expected/`. Headline values:

### Short form, calibrated

| Method | dev3spk IoU / BE / Kendall% | NSF dev | NSF eval |
|---|---|---|---|
| WavLM 8spk | 59.9 / 72 ms / 13.0 | 58.3 / 115 / 9.3 | 58.6 / 104 / 17.4 |
| Token only | 59.7 / 73 / 13.0 | 57.8 / 123 / 10.6 | 58.0 / 116 / 19.1 |
| One-state | 60.1 / 71 / 12.7 | 58.4 / 113 / 9.1 | 58.6 / 102 / 17.2 |
| BPE trim | 64.1 / 63 / 12.8 | 60.9 / 105 / 9.0 | 61.3 / 96 / 17.5 |

### NOTSOFAR full-recording ablation

| Method | Dev IoU / BE / Kendall% | Eval | RTFx | Peak GPU |
|---|---|---|---:|---:|
| One-state, all frames | 58.4 / 117 ms / 12.6 | 58.7 / 104 / 19.0 | 9.7 | 14.3 GiB |
| Retain 30% frames | 56.7 / 128 / 12.7 | 57.4 / 111 / 19.2 | 19.3 | 8.5 GiB |
| Lazy graph chunks | 57.4 / 123 / 12.6 | 57.8 / 108 / 19.2 | 29.8 | 2.0 GiB |
| BPE trim | 61.0 / 112 / 12.4 | 61.4 / 97 / 19.4 | 17.0 | 2.0 GiB |

### Full recordings

- Speaker-attributed: NSF dev/eval IoU 57.4/57.8; AMI dev/eval 61.5/61.1.
- Token-only: NSF 57.2/57.8; AMI 61.6/61.2.
- Speaker-attributed + BPE trim: NSF 60.0/61.0; AMI 64.7/64.1.
- Segment-level WDER: NSF dev/eval 22.0/32.8; AMI dev/eval 21.1/17.6.
- BPE trim changes timing but not materially the segment WDER.

## 10. Models and exploratory experiments

All code below is on `archive/mtasr-experiments-2026`, not the supported paper
branch.

### Parakeet high-resolution parallel branch

Code:

```text
parakeet_native_spashl_ctc/
slurm/train_fastmss_yodas_parakeet_factorized_a100_accum4.sbatch
tests/test_parakeet_adapter.py
```

Final experiment:

```text
exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe
```

Architecture/configuration:

- Base: `nvidia/parakeet-tdt-0.6b-v3`.
- Tap high-resolution convolutional features at 50 Hz.
- Lightweight parallel adapter: dimension 256, four heads, FFN 1024.
- Three adapter/cross-attention stages corresponding to Parakeet backbone
  layers 7, 15, and 23.
- Local self-attention window 257, cross-attention window 65, block size 256.
- The 50 Hz adapter queries low-rate Parakeet states. Final adapter output is
  fused with learned-blank-upsampled Parakeet output, normalized, then passed
  to factorized token/speaker heads.
- FastMSS + filtered English YODAS, combined 5k BPE, factorized loss.
- Two A100s, `max_duration=2700`, at most two cuts/GPU, accumulation 4.
- LR `1e-4`, freeze 5k, freeze/head LR `4e-5`, 200k steps.

The final five checkpoints (192k--200k, about 9.97 GB each) are preserved
locally and were commonly averaged for evaluation. They are not published,
because the user only required the code. The public archive branch contains
the complete model implementation and launcher.

### Whisper

Main exploratory families:

- `fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6`
- `fastmss_yodas_en_whisper_large_v3_turbo_shuffle_a100x2_accum4_maxlr5e-6`
- `fastmss_yodas_en_whisper_large_v3_turbo_shuffle_segment_factor_softmask_a100x2_accum4_maxlr5e-6`

Whisper used 30 s inputs and center-crop chunking for long recordings. The
lower post-unfreeze maximum LR (`5e-6`) was adopted after unstable/poor runs.
The segment-factor soft-mask objective was introduced to distinguish duplicate
token paths by segment identity while assigning a time-local soft distribution
over plausible segments.

### Token-only graph non-determinism

Token-only shuffle graphs can contain multiple graph paths emitting identical
token sequences when utterances share tokens. Forward summation then counts
duplicate paths and can produce negative graph CTC losses after subtracting
the denominator; clamping hid rather than fixed this. Determinization was
tested but potentially expensive. The segment-factor graph preserves segment
labels and supplies a time-local soft segment prior. This issue affects summed
training loss much more directly than Viterbi alignment, where tied duplicate
paths mostly make the recovered segment ID arbitrary.

### Teacher-token FIFO speaker model and diarization baselines

The experiment
`fastmss_yodas_en_wavlm_large_teacher_token_fifo_spashl_factorized_v100x4_accum4`
uses a frozen token-only WavLM as an alignment teacher with local transcript
masks, while a second WavLM speaker branch learns FIFO speaker posteriors.
Alternative frame-level FIFO BCE and powerset CE diarization models were also
implemented. Posterior-only diarization, ECAPA/AHC stitching, VBx/PLDA,
token-derived embedding selection, oracle-K, and estimated-K sweeps are all
archived. Their results were exploratory and are not part of the authoritative
paper tables.

### Other archived work

- Encoder-only and random-initialized Parakeet runs.
- Zipformer-from-scratch experiments (did not train successfully).
- Target-speaker alignment with speaker-specific blank probabilities.
- Shuffle-only alignment followed by segment/word speaker allocation.
- MFA and Wav2Vec2 CTC forced-alignment baselines.
- Browser alignment viewer with zoom, seek, and playback-rate control.
- Exact, batched, blockwise, and forward-pruned Viterbi diagnostics.
- AHC/VBx threshold and embedding-selection sweeps.
- Whisper/Parakeet argmax decoding and tcpWER experiments.

## 11. Speaker attribution and diarization caveats

- Short-form speaker-attributed graphs do not require cross-chunk stitching.
- Long-form slot labels are local to model input chunks. A single global
  permutation is meaningless when slots are ordered independently by local
  speaker duration. Use a per-chunk oracle mapping for oracle experiments or a
  real stitching system (AHC/VBx) for deployable results.
- In token-only post-facto attribution, sum speaker log posterior scores over
  all emitted tokens/frames belonging to one reference segment, then assign
  the segment to the best speaker. Word-level assignment was also tested and
  was generally worse in the final paper table.
- DER should normally be scored against supervision/utterance activity
  regions, not reference word boxes. Word-derived reference segments omit
  inter-word speech and create artificial missed speech.
- The 0.2 s historical fallback token duration was replaced by the
  character-proportional 0.08 s/character rule in final BPE-trim experiments.
- WDER and DER answer different questions. Do not infer frame-level DER from
  WDER without explicitly constructing speech activity intervals and applying
  the desired collar.

## 12. Checkpoint preservation

Every checkpoint present on 2026-11-23 was hard-linked into:

```text
/expscratch/mwiesner/shuffle/release/experimental_models_2026-11-23/
```

There are 151 archived checkpoints with 1.17 TB logical size. Hardlinks do not
duplicate data blocks, but protect the checkpoint when a rolling experiment
removes its original filename. The archive branch contains:

```text
archive/checkpoint_inventory_2026-11-23.json
archive/pin_experiment_checkpoints.py
```

Do not move these using a copy/delete operation without understanding the
hardlink behavior. The WavLM paper checkpoint has a separate compact public
artifact as described above.

## 13. Tests already run for the release

- Python compilation passed for the reproduction and export programs.
- Shell syntax passed for the paper build and main Slurm scripts.
- Factorized Viterbi CPU test passed; CUDA portion skipped without an
  allocated GPU.
- Blockwise Viterbi: 6 tests passed.
- Dynamic collar guard: 2 tests passed.
- WavLM joint-label tests: 2 tests passed.
- Total reported: 12 passed, one CUDA-only test skipped.
- `reproduce.py check --hash-checkpoint` passed all model, BPE, manifest, and
  reference checks.
- Short- and long-form Slurm dry runs passed.
- Hugging Face reported the exact checkpoint byte count and content SHA-256.

## 14. Presentation and visualization assets

These generated presentation files live outside the Git repository:

- Conference deck source: `/expscratch/mwiesner/shuffle/slides/main_slides.tex`
- Rendered deck: `/expscratch/mwiesner/shuffle/slides/main_slides.pdf`
- Editable PowerPoint conversion:
  `/expscratch/mwiesner/shuffle/slides/main_slides_editable.pptx`
- PowerPoint builder: `/expscratch/mwiesner/shuffle/slides/build_pptx.py`
- Rendered slide checks: `/expscratch/mwiesner/shuffle/slides/render/`

Reusable result visualizations also live outside the repository:

- NOTSOFAR WavLM-8spk alignment visualization and generator:
  `/expscratch/mwiesner/shuffle/visualizations/notsofar_wavlm8spk_alignment/`
- Synthetic-overlap posterior/interleaving visualization and generator:
  `/expscratch/mwiesner/shuffle/visualizations/dev_synth_wavlm8spk_overlap/`

The latter directories contain PDF, SVG, PNG, and HTML renderings plus the
source script and provenance/data needed to regenerate them. A requested
Parakeet architecture diagram was discussed, but no corresponding saved asset
was found under `visualizations/`; reconstruct it from the implementation
details in Section 10 if it is needed again.

## 15. Practical rules for the next chat

1. Establish which branch a requested change belongs on. Paper reproduction
   changes go on `paper/longform-multitalker-alignment`; exploratory systems
   go on `archive/mtasr-experiments-2026` or a new focused branch.
2. Before reporting an old result, cite the exact metrics JSON/CSV and model
   checkpoint. Similar filenames often differ by topology, frame retention,
   calibration, microphone, or Kendall implementation.
3. Always state whether audio is short-form 30 s or a full recording, whether
   audio inference is chunked, and whether graph construction is chunked.
4. Always state topology, time mask, blank retention/removal, BPE trimming,
   token-only versus speaker-attributed graph, checkpoint averaging, and
   calibration protocol.
5. For NOTSOFAR use only `sc_plaza_0`; for AMI use SDM.
6. Preserve duration/speech-time speaker ordering unless an experiment
   explicitly requests FIFO or another supplied ordering.
7. Do not call a per-chunk oracle map a deployable stitching result.
8. Avoid `k2.intersect_dense` for the supported long-form result; use the
   exact custom Viterbi unless deliberately running a baseline.
9. Do not install/replace k2 globally. Use the narrow local factorized CUDA
   extension.
10. Check `squeue -u mwiesner` before assuming a job is pending or finished.

## 16. Good first commands in a new session

```bash
cd /expscratch/mwiesner/shuffle/icefall-shuffle
git status --short
git branch --show-current

cd egs/librispeech/MTASR
source activate.sh
python repro/longform_alignment/reproduce.py check --hash-checkpoint
python repro/longform_alignment/reproduce.py list
squeue -u mwiesner
```

If the new task concerns Parakeet, Whisper, diarization, segment-factor losses,
or other non-paper work, inspect the archive branch without losing local state:

```bash
git show archive/mtasr-experiments-2026:egs/librispeech/MTASR/archive/README.md
git ls-tree -r --name-only archive/mtasr-experiments-2026 | less
```
