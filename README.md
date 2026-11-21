# ReLiC-SC

This repository provides the implementation accompanying **ReLiC-SC: Calibrated
Response Modeling for Reliable Visual EEG Retrieval**.

The versioned code release is
[v0.1.0](https://github.com/19140103/ReLiC-SC/releases/tag/v0.1.0).
The repository is organized into four parts:

- `encoder/`: the cumulative-window, multi-layer EEG/image representation
  training code used to obtain the branch embeddings;
- `relic_sc/score.py`: the compact ridge-prediction and mean-calibration
  reference scorer;
- `relic_sc/full_score.py`: the fixed ReLiC-SC scorer used for the paper,
  including mean/variance calibration, candidate-local residual weighting,
  visual-layer scores, residual-MSE layer weights, and nested score fusion;
- `src/rapid_eeg/` and `scripts/data/`: leakage-safe Alljoined metadata and
  data-preparation utilities used by the experiments.

The repository does not contain raw EEG recordings, participant-level data,
model checkpoints, generated result logs, or credentials.

## Environment

The reference environment uses Python 3.12, PyTorch 2.6, CUDA 12.6, NumPy,
Pandas, and the packages listed in `requirements.txt`. Install the dependencies
with:

```bash
pip install -r requirements.txt
```

For the `src/rapid_eeg` helper imports, install the repository in editable mode:

```bash
pip install -e .
```

Encoder entrypoints under `encoder/scripts/` resolve their sibling package
directly and can be run from the repository root.

## Data sources and access

The experiments use two publicly released datasets. Download them from the
official providers and follow their current access and use terms:

- **THINGS-EEG2**: [official OSF release](https://osf.io/3jk45/), described by
  Gifford et al. (2022).
- **Alljoined-1.6M**: [official Hugging Face dataset](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M),
  with the associated [official code/data page](https://github.com/Alljoined/Alljoined-1.6M).

The Alljoined data card currently specifies CC BY-NC-SA 4.0 together with
additional non-commercial access terms. Users must comply with the provider's
license, terms of access, privacy policy, and any later dataset revision. The
download helper in `scripts/data/download_alljoined.py` defaults to metadata-only
operations and requires an explicit selection before downloading large files.

## Reproducing the scoring analysis

The scoring script consumes saved branch embeddings. Each run specification is
given as `label=run_name`, and each run directory contains
`train_test_embeddings.npz` with the training/query and image embeddings:

```bash
python relic_sc/score.py \
  --embedding-root /path/to/embeddings \
  --subjects sub-01 sub-02 \
  --seeds 20260829 20260830 20260831 \
  --run-spec G=global_run \
  --run-spec H=stage_run \
  --out results/relic_sc.csv
```

The full scorer additionally requires an image-bank directory containing one
`level-*` subdirectory per visual layer, with `image_train.npy` and
`image_test.npy` files aligned to the index fields in each embedding archive:

```bash
python relic_sc/full_score.py \
  --embedding-root /path/to/embeddings \
  --image-bank-root /path/to/ViT-B32-depth-bank-8 \
  --subjects sub-01 sub-02 \
  --seeds 20260829 20260830 20260831 \
  --run-spec G=global_run \
  --run-spec H=stage_run \
  --out results/relic_sc_full.csv
```

The compact script reports direct cosine matching, prediction-based ReLiC
scoring, mean-calibrated ReLiC-SC, shuffled-prediction controls, Top-1/5/10,
MRR, mean rank, and paired deltas. It also writes summary and alpha-selection
CSV files. For the full paper configuration, provide the selected visual-layer
feature banks and run `relic_sc/full_score.py`; its fixed settings are recorded
in the module docstring and match the equations in the manuscript.

## Representation learning

The scripts under `encoder/` implement the cumulative EEG-window and selected
CLIP ViT-B/32 visual-layer representation training used in the manuscript.
The Alljoined G/H presets include G's bottleneck visual adapter and H's learned
stage-specific visual targets and auxiliary contrastive loss. The existing
THINGS-EEG2 presets retain their original global-alignment behavior. See
[`encoder/README.md`](encoder/README.md) for preparation, training, export, and
scoring commands. Training does not download data implicitly, and outputs stay
outside version control.

## Tests

After installing the dependencies, install pytest and run:

```bash
python -m pip install pytest
python -m pytest -q
```

The synthetic-data tests cover the G/H adapters and weighted stage objective,
CPU training/export/scoring, data identity, and scoring regressions. See
[Tests and validation](docs/gh_release_validation.md) for coverage and the test
environment.

## Ethics and data handling

The paper reports a secondary analysis of de-identified public EEG data. No new
participants were recruited and no new recordings were collected for this
release. Ethical approval and informed consent for the original recordings are
the responsibility of the dataset providers.

## License

The code is released under the [MIT License](LICENSE). The `encoder/cevr` and
`src/rapid_eeg` paths contain pre-existing encoder components retained for
compatibility; source attribution is preserved in the source files and LICENSE.
The licenses and access terms of THINGS-EEG2, Alljoined-1.6M, CLIP, and other
external resources apply separately.
