# Encoder release

This directory contains the representation-learning component used before
ReLiC-SC scoring. The default configuration uses cumulative EEG windows and
selected layers of a frozen CLIP ViT-B/32 visual encoder.

Keep downloaded EEG arrays, visual features, checkpoints, and results outside
Git; the root `.gitignore` excludes these file types. All commands below run
from the repository root. Install the root requirements first; preparing
Alljoined metadata also requires PyArrow (listed there).

## THINGS-EEG2

Example commands (after preparing the THINGS-EEG2 data and visual features):

```bash
python encoder/scripts/extract_multilevel_visual_features.py \
  --image-set-dir data/things_eeg/image_set \
  --output-dir data/things_eeg/image_feature/ViT-B32-multilevel-8 \
  --layers 1 2 4 5 7 8 10 12

python encoder/scripts/train_intra_subject.py \
  --config encoder/configs/main.json \
  --subject sub-01 \
  --eeg-data-dir data/things_eeg/preprocessed_eeg_npy \
  --visual-feature-root data/things_eeg/image_feature/ViT-B32-multilevel-8 \
  --output-dir results/encoder
```

Each completed run writes `train_test_embeddings.npz`, containing the train
and test EEG predictions, mapped visual targets, and object/image indices
consumed by the scoring scripts. A compact test-only `final_embeddings.npz`
is also written for backward compatibility.

## Alljoined-1.6M: G and H

The portable presets `encoder/configs/alljoined_G.json` and
`encoder/configs/alljoined_H.json` specify the paper's Alljoined branches:

| Setting | G | H |
|---|---|---|
| Global visual adapter | 512 → 128 → GELU → dropout 0.1 → 512 | Linear 512 → 512 |
| EEG stage pooling | Mean across eight cumulative windows | Same |
| Stage-specific visual supervision | Disabled | Learned softmax mixture of raw visual layers, then one shared projection |
| Training objective | Global symmetric contrastive loss | Global loss + 0.2 × weighted stage loss |

Stage losses are averaged with weights `[1,2,3,4,5,6,7,8]/36`.
H stage/layer logits start at 2 on the diagonal and 0 elsewhere; their softmax
temperature is 1, distinct from contrastive temperature 0.07. Both use all 32
Alljoined channels, windows `[25,50,75,100,125,150,200,250]`, train-derived
per-channel EEG standardization, AdamW, batch size 1024, and 20 epochs with a
cosine learning-rate schedule. G and H are trained independently.

### Prepare the encoder inputs

Start with a processed subject from `scripts/data/prepare_pilot.py` and the
official stimulus images. Create the depth cache using a locally downloaded
official OpenAI CLIP ViT-B/32 checkpoint:

```bash
python scripts/data/extract_encoder_visual_cache.py \
  --processed-subject /path/to/processed/sub-01 \
  --image-root /path/to/Alljoined-1.6M/stimuli/images \
  --output-dir /path/to/clip_depth_cache \
  --pretrained /path/to/local/ViT-B-32.pt --device cuda:0
```

Add `--trusted-torchscript-checkpoint` only when the trusted official file is
a TorchScript archive rather than a weights-only checkpoint. Use `--device cpu`
if needed. No image or checkpoint downloads occur in this command. It fixes the
historical Alljoined backbone to `ViT-B-32-quickgelu`, uses patch-token means
after `ln_post` and the visual projection, and stores unnormalized float16
features from float32 inference. It records checkpoint SHA-256 and sorted image
identities. The THINGS example above remains a separate preset.

The subsequent bridge requires:

- Processed subject: `train_eeg.npy`, `test_eeg.npy`, corresponding
  `train_metadata.parquet`/`test_metadata.parquet`, `channels.json`, and `times.npy`.
- Visual cache: `embeddings.npy` shaped `[images,8,512]`, `image_index.parquet`
  with `image_id` and `embedding_row`, and `cache_report.json` identifying
  `visual_layers: [1,2,4,5,7,8,10,12]`.

```bash
python scripts/data/prepare_encoder_data.py \
  --processed-subject /path/to/processed/sub-01 \
  --image-cache /path/to/clip_depth_cache \
  --output-root /path/to/alljoined_encoder \
  --subject sub-01
```

The bridge checks image identity, split separation, row order, channel/time
metadata and shared visual-bank consistency. It averages the actually retained
repetitions for each image; it does not pad or invent missing trials. The saved
`[objects,images,1,channels,time]` arrays have one **averaged** slot, not one
original trial. This route does not reproduce single-trial or repetition-budget
analyses. Repeat counts and image-key manifests accompany the prepared inputs.
Conflicting existing outputs are rejected.

### Train, export, score

```bash
python encoder/scripts/train_intra_subject.py \
  --config encoder/configs/alljoined_G.json --subject sub-01 --seed 20260829 \
  --eeg-data-dir /path/to/alljoined_encoder/preprocessed_eeg_npy \
  --visual-feature-root /path/to/alljoined_encoder/image_feature/ViT-B32-depth-bank-8 \
  --output-dir /path/to/embeddings

python encoder/scripts/train_intra_subject.py \
  --config encoder/configs/alljoined_H.json --subject sub-01 --seed 20260829 \
  --eeg-data-dir /path/to/alljoined_encoder/preprocessed_eeg_npy \
  --visual-feature-root /path/to/alljoined_encoder/image_feature/ViT-B32-depth-bank-8 \
  --output-dir /path/to/embeddings

python relic_sc/full_score.py \
  --embedding-root /path/to/embeddings \
  --image-bank-root /path/to/alljoined_encoder/image_feature/ViT-B32-depth-bank-8 \
  --subjects sub-01 --seeds 20260829 \
  --run-spec G=G_all8_mean_bottleneck \
  --run-spec H=H_all8_mean_stage_scalar_hier02 \
  --out /path/to/results/relic_sc_full.csv
```

Repeat for the required subjects and seeds (the paper uses 20260829, 20260830,
20260831). `--dry-run --cpu` validates inputs/model construction without training;
it still loads the prepared data. Choose a new output directory to rerun a
completed run: the trainer refuses to overwrite a final checkpoint.

Exports use the **final epoch**, never the best test-diagnostic epoch. H's
intermediate stages supervise training only; retrieval exports its final EEG
embedding. `train_test_embeddings.npz` contains raw float32 `train_pred`,
`train_image`, `test_pred`, `test_image`, plus `train_object_idx`,
`train_image_idx`, `test_object_idx`, `test_image_idx`. Scorers normalize these
vectors at consumption. Test data are used for diagnostics/export, not the
training objective or EEG standardization.

## Provenance and verification boundary

The G/H adapter and stage-target equations are ported from the existing
CEVR-derived training implementation; source attribution is retained in the
root LICENSE. Presets preserve the effective model/objective and recorded
training settings. Unused auxiliary branches and historical machine paths are
not carried over. Consequently a seed alone does not promise bit-for-bit
identity with historical training runs (initialization/dropout RNG consumption
may differ).

Lightweight tests use synthetic data, small dimensions and one epoch. They
verify forward/gradient behavior, normalized stage weights, train/export/score
connectivity and data identity; they do not establish the paper's reported
accuracy. Full-dataset training and benchmark reproduction remain separate
operations requiring the original licensed data and computational resources.
