# Long-form Speaker Attributed Forced Alignment with Shuffles

This directory contains the manuscript source and the final PDF for the paper.
The source was recovered from the Overleaf export
`Longform_Multi_Talker_Alignment.zip`; it is kept verbatim under `source/`.
The reported experiment configuration and result tables are versioned in
`../../egs/librispeech/MTASR/repro/longform_alignment/`.

## Build the paper

A complete TeX Live installation is required.  In particular, `multirow`,
`shuffle`, `algorithm2e`, `stmaryrd`, `subcaption`, `pgfplots`, and TikZ must
be installed.  Build without modifying the recovered source tree:

```bash
./build.sh
```

The output is `build/Longform_Multi_Talker_Alignment.pdf`.  The checked-in
`Longform_Multi_Talker_Alignment.pdf` is the submitted/rendered reference.
PDF byte identity is not expected because TeX embeds build metadata; the
source, tables, bibliography, and plot data are versioned.

To rebuild the experimental tables first, follow the recipe README:

```bash
cd ../../egs/librispeech/MTASR
python repro/longform_alignment/reproduce.py check
python repro/longform_alignment/reproduce.py list
```
