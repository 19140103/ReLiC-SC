# Tests and validation

## Automated tests

Run the test suite from the repository root after installing the dependencies
and pytest:

```bash
python -B -m pytest -q -p no:cacheprovider
```

All 51 tests pass with Python 3.11.8 and PyTorch 2.5.1 on CPU. The tests use
synthetic data and cover:

- G bottleneck global visual adapter.
- H learned stage/layer mixture of raw CLIP features, shared stage projection,
  and normalized stage-weighted auxiliary loss (weight 0.2).
- Alljoined G/H configuration loading, final-epoch training, raw train/test
  embedding export, and full scoring with branch fusion.
- CLIP depth-cache identity, repeat-averaged EEG preparation, split separation,
  and output-conflict protection.
- Global-only encoder behavior and scoring/normalization regressions.

These checks validate software behavior and numerical consistency; they are
separate from full-dataset accuracy, multi-seed, and GPU training benchmarks.

## Numerical and integration checks

The following checks were completed on 5 November 2026:

- G/H visual forward outputs compared with the archived implementation under
  the same weights: exact equality, including H stage targets and weights.
- Each tiny branch trained for one epoch, with 8 training and 4 test images;
  gradients, parameter updates, checkpoint/export values and indices checked.
  Both exports then passed through the full scorer and branch fusion.
- Actual CLIP cache CLI checked on two synthetic images with the official local
  ViT-B/32 checkpoint, using Python 3.12.7, OpenCLIP 2.31.0, PyTorch 2.6.0 and CPU.
  The checkpoint SHA-256 matched the OpenAI metadata:
  `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
  Output: `[2,8,512]`, finite float16 values, no L2 normalization.

The [encoder documentation](../encoder/README.md) describes input preparation
and random-seed behavior. Dataset access and licensing are described in the
[main README](../README.md#data-sources-and-access).
