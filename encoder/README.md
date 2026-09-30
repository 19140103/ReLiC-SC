# Encoder release

This directory contains the representation-learning component used before
ReLiC-SC scoring. The default configuration uses cumulative EEG windows and
selected layers of a frozen CLIP ViT-B/32 visual encoder.

The data layout and training commands are documented in the repository-level
README. Keep downloaded EEG arrays, visual features, checkpoints, and results
outside Git; the root `.gitignore` excludes these file types.

Example commands (after preparing the THINGS-EEG2 data and visual features):

```bash
python scripts/extract_multilevel_visual_features.py \
  --image-set-dir data/things_eeg/image_set \
  --output-dir data/things_eeg/image_feature/ViT-B32-multilevel-8 \
  --layers 1 2 4 5 7 8 10 12

python scripts/train_intra_subject.py \
  --config configs/main.json \
  --subject sub-01 \
  --eeg-data-dir data/things_eeg/preprocessed_eeg_npy \
  --visual-feature-root data/things_eeg/image_feature/ViT-B32-multilevel-8 \
  --output-dir results/encoder
```

Each completed run writes `train_test_embeddings.npz`, containing the train
and test EEG predictions, mapped visual targets, and object/image indices
consumed by the scoring scripts. A compact test-only `final_embeddings.npz`
is also written for backward compatibility.
