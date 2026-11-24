# From-scratch Zipformer SPaSHL

This variant replaces the pretrained WavLM/Parakeet encoder with the waveform
frontend and `Zipformer2` encoder used by `egs/librispeech/SSL/zipformer`.
The seven-layer convolutional frontend has a total stride of 320 samples, so
16-kHz input remains at 50 frames per second. The Zipformer output
downsampling factor is fixed to one. Separate token and speaker heads feed the
same factorized shuffle-CTC loss used by the other MTASR models.

Activate the MTASR environment and inspect the available training arguments:

```bash
source egs/librispeech/MTASR/activate.sh
python egs/librispeech/MTASR/zipformer_spashl_ctc/train.py --help
```

The default `--zipformer-*` settings match the SSL recipe's full Zipformer
configuration. They are exposed individually so a smaller model can be
selected without modifying code. Because this encoder is initialized from
scratch, `--freeze-iters` defaults to zero. At startup the training log reports
the encoder, output-head, and total parameter counts separately.
