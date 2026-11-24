# Prompt for the next chat

I am continuing a long-running multi-talker ASR, forced-alignment, and
diarization project in this workspace:

```text
/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
```

Before doing anything, read this entire handoff:

```text
/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR/HANDOFF.md
```

Then inspect the live repository status, current branch, Slurm queue, and the
paper reproduction README. Do not assume an experiment is running, finished,
or comparable based only on a similar filename.

The supported paper release is on
`paper/longform-multitalker-alignment`. Exploratory Parakeet, Whisper,
Zipformer, diarization, target-speaker, segment-factor, and diagnostic work is
on `archive/mtasr-experiments-2026`. Keep new work on the appropriate branch
and preserve existing checkpoints/results.

The authoritative reproduction entry point is:

```text
repro/longform_alignment/README.md
```

The released WavLM 8-speaker checkpoint is at:

```text
https://huggingface.co/mshwiesner/longform-multitalker-alignment
```

Use the checked-in expected CSVs and exact metrics JSON files when reporting
results. Keep calibration, microphone, short/full recording status, topology,
time mask, blank-frame retention, graph chunking, BPE trimming, graph type,
and checkpoint averaging explicit. In particular, do not substitute the wrong
Kendall metric or pooled/token-weighted aggregation for the per-recording
paper metrics.

Start the environment with:

```bash
cd /expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
source activate.sh
```

My next request follows after you have read and summarized the handoff.
