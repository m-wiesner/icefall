# MTASR experimental archive

This directory belongs to branch `archive/mtasr-experiments-2026`. It is a
provenance archive for work that is not part of the long-form alignment paper:
Parakeet/Whisper/Zipformer encoders, speaker stitching and VBx/AHC work,
target-speaker alignment, segment-factor losses, standalone diarization
losses, memory diagnostics, smoke tests, historical Slurm launchers, and
compact plot/result artifacts.

These files are intentionally separated from the supported paper release in
`repro/longform_alignment/`. A result in this branch should not be interpreted
as a claim in the paper unless it is also present in the paper source and its
`expected/` tables.

## Large checkpoints

Git does not contain multi-gigabyte model checkpoints. On 2026-11-23, every
checkpoint still present under the local `exp/` tree was hard-linked into:

```text
/expscratch/mwiesner/shuffle/release/experimental_models_2026-11-23/
```

Hard links consume no second copy of the model data but protect the files when
rolling experiment directories delete old names. The committed
`checkpoint_inventory_2026-11-23.json` records source path, archive path,
size, modification time, and inode. This is a local preservation archive, not
a public download service. Use `pin_experiment_checkpoints.py` to make a new
snapshot; it refuses to overwrite a different file.

The paper's 72k WavLM checkpoint is additionally exported as a compact,
hashed release artifact and documented in
`../repro/longform_alignment/artifacts.json`.

