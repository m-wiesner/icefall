# MTASR branch guide

The MTASR work is split by purpose so a paper reader does not have to navigate
all of the exploratory experiments.

## `paper/longform-multitalker-alignment`

The reproducible release for *Long-form Speaker Attributed Forced Alignment
with Shuffles*. It contains:

- the complete LaTeX source, final reference PDF, bibliography, and plot data;
- the WavLM speaker-attributed and token-only training implementation;
- exact Viterbi alignment, time masks, one-state topology, blank-frame
  removal, lazy graph chunks, transcript remapping, and BPE trimming;
- portable FastMSS/YODAS and AMI/NOTSOFAR preparation tools;
- exact Slurm configurations, artifact hashes, calibration/scoring programs,
  and the values typeset in the paper.

This is the branch colleagues should clone to reproduce the paper.

## `archive/mtasr-experiments-2026`

An archival superset created from the paper branch. It preserves work that is
not claimed in the long-form paper, including Parakeet, Whisper, Zipformer,
target-speaker, diarization-stitching, alternative-loss, memory-diagnostic,
and failed/partial ablation code, plus compact result artifacts and historical
Slurm launchers. It is retained for provenance and future work, not presented
as a supported paper recipe.

Large `exp/` directories and model binaries are not committed to ordinary Git.
The archive instead contains a machine-readable inventory with hashes and
local artifact locations. Published models should live in a release/artifact
store and be verified using the checked-in manifest.

## Companion repositories

- `third_party/k2`, branch `feature/shuffle-factorized-intersection`: the
  modified factorized CUDA dense-intersection implementation used by training.
- `yodas_pseudo`, branch `feature/mtasr-english-filtered-cutset`: the English
  YODAS filtering change used to prepare the paper training subset.
