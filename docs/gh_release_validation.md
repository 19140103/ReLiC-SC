# G/H release validation

Checked on 2026-11-05. This is an implementation-completeness check for the
Alljoined G/H training and scoring path, not a rerun of the manuscript results.

## Included paths

- G bottleneck global visual adapter.
- H learned stage/layer mixture of raw CLIP features, shared stage projection,
  and normalized stage-weighted auxiliary loss (weight 0.2).
- Portable Alljoined G/H configuration files, final-epoch training and raw
  train/test embedding export for both scoring programs.
- Image-ID-indexed CLIP depth cache extraction and retained-repeat mean EEG
  preparation, with split/identity checks and conflict protection.
- Legacy global-only encoder behavior and existing scoring/normalization fixes.

## Checks performed

- `python -B -m pytest -q -p no:cacheprovider`: 51 tests passed in the local
  Python 3.11.8 / PyTorch 2.5.1 CPU test environment. Tests use synthetic data.
- G/H visual forward outputs compared with the archived implementation under
  the same weights: exact equality, including H stage targets and weights.
- Each tiny branch trained for one epoch, with 8 training and 4 test images;
  gradients, parameter updates, checkpoint/export values and indices checked.
  Both exports then passed through the full scorer and branch fusion.
- Actual CLIP cache CLI checked on two synthetic images with the official local
  ViT-B/32 checkpoint, using Python 3.12.7, OpenCLIP 2.31.0, PyTorch 2.6.0 and CPU.
  Its SHA-256 matched the OpenAI checkpoint metadata:
  `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
  Output: `[2,8,512]`, finite float16 values, no L2 normalization.
- No raw EEG, participant data, model weights, generated caches or run outputs
  are included in the release. Existing source attribution is retained.

## Not claimed

No full-dataset accuracy reproduction, multi-seed rerun or GPU training benchmark
was performed for this release. The mean-only data route does not implement the
single-trial/repetition-budget supplementary analyses. Removing unused historic
training branches can change random-number consumption; matching a seed is not
a guarantee of bitwise-identical historical training. The data providers' access
and license conditions continue to apply independently of this code's license.
