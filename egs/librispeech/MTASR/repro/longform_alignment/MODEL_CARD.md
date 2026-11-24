---
license: cc-by-sa-3.0
base_model: microsoft/wavlm-large
tags:
  - wavlm
  - automatic-speech-recognition
  - speaker-attributed-asr
  - forced-alignment
  - multi-speaker
  - ctc
---

# WavLM 8-speaker Shuffle/SpAShl model

This repository contains the checkpoint used for the experiments in
*Long-form Speaker Attributed Forced Alignment with Shuffles*. It is a WavLM
Large model with factorized token and eight-slot speaker output branches,
trained with the speaker-attributed Shuffle/SpAShl CTC objective.

This is a research checkpoint rather than a standard Transformers pipeline.
Inference and alignment require the matching icefall MTASR recipe and custom
factorized Shuffle CTC extension from the
`paper/longform-multitalker-alignment` branch of
[`m-wiesner/icefall`](https://github.com/m-wiesner/icefall).

## Artifact

| File | Description | Size | SHA-256 |
|---|---|---:|---|
| `wavlm8spk-checkpoint-72000.pt` | Compact model state and provenance for checkpoint 72,000 | 1,282,524,827 bytes | `9cb800a1c6584bd5bbfc14359dcf4f767e87f3ea9b45ca4369588221dbb5db21` |

The compact archive is sufficient for the paper's checkpoint-72,000,
`avg=1` alignment and decoding runs. It intentionally excludes optimizer,
scheduler, gradient-scaler, sampler, and other training-resume state.

## Training configuration

- Base encoder: WavLM Large, 16 kHz input, without output downsampling.
- Outputs: separate token and eight-speaker-slot branches; their scores are
  combined inside the factorized CUDA loss without materializing the full
  token-by-speaker tensor.
- Tokenization: 5,000-unit unigram BPE trained on normalized LibriSpeech-960
  and filtered English YODAS transcripts.
- Data: FastMSS LibriSpeech supervision groups and filtered English YODAS
  `doverlap` groups, sampled with equal mixture weight.
- YODAS filter: judge score at least 8.0 and DER at most 0.069; groups contain
  2--8 speakers, span at most 60 seconds, and join pauses up to 1 second.
- FastMSS: four shards grouped at 1-second pauses; groups longer than 80
  seconds are excluded.
- Optimization: four V100 GPUs, 180 seconds maximum duration per GPU,
  gradient accumulation 4, base learning rate `5e-5`, 2,000 frozen-encoder
  iterations with head learning rate `5e-3`, and 200,000 configured steps.
- Supervision: Brownian-bridge interval model, 2-second collar, beam 10,
  speech-duration speaker ordering, and up to four concurrent speakers in
  synthetic mixtures.

The exact command line, data preparation, BPE files, expected result tables,
and alignment scripts are versioned with the icefall paper branch.

## Intended use

The model is intended for research on multi-talker ASR, speaker-attributed
forced alignment, token-only alignment, and the long-form alignment
ablations described in the accompanying manuscript. It was developed for
English speech. It is not packaged as a production transcription service and
has not been evaluated for safety-critical use.

## Limitations

- The output has eight speaker slots and assumes the recipe's speaker-ordering
  convention; slot identities are not persistent identities across arbitrary
  independently processed chunks.
- Several reported long-form speaker-attribution experiments use an oracle
  mapping from chunk-local slots to recording-global speaker identities.
- Performance may not transfer to other languages, acoustic domains,
  microphones, or substantially larger speaker populations.
- Reproducing corpus results requires separately licensed AMI, NOTSOFAR,
  LibriSpeech/FastMSS, and YODAS audio; no training or evaluation audio is
  included here.

## License and attribution

This checkpoint is a fine-tuned derivative of
[`microsoft/wavlm-large`](https://huggingface.co/microsoft/wavlm-large) and is
distributed under the Creative Commons Attribution-ShareAlike 3.0 Unported
license (CC BY-SA 3.0). Users must provide appropriate attribution and
distribute adaptations under the same or a compatible license.

The original WavLM implementation is Copyright Microsoft Corporation. The
Microsoft UniLM source repository is distributed under the MIT License. This
model is intended primarily for research and evaluation; that statement does
not add restrictions beyond CC BY-SA 3.0.

## Citation

```bibtex
@misc{wiesner2026longform,
  title  = {Long-form Speaker Attributed Forced Alignment with Shuffles},
  author = {Matthew Wiesner and Alexander Polok and Samuele Cornell and
            Lucas Ondel Yang and Sanjeev Khudanpur and Luk\'{a}\v{s} Burget},
  year   = {2026}
}

@article{chen2021wavlm,
  title   = {WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
             Speech Processing},
  author  = {Sanyuan Chen and Chengyi Wang and Zhengyang Chen and Yu Wu and
             Shujie Liu and Zhuo Chen and Jinyu Li and Naoyuki Kanda and
             Takuya Yoshioka and Xiong Xiao and Jian Wu and Long Zhou and
             Shuo Ren and Yanmin Qian and Yao Qian and Jian Wu and
             Michael Zeng and Furu Wei},
  journal = {IEEE Journal of Selected Topics in Signal Processing},
  year    = {2022}
}
```
