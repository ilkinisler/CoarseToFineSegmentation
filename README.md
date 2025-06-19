# Coarse-to-Fine Tumor Segmentation with Uncertainty and Anatomical Post-Processing

This repository implements a two-stage segmentation framework for lung tumors in CT scans, combining uncertainty modeling and anatomically informed post-processing. The pipeline includes coarse full-volume segmentation followed by refined ROI-based segmentation, and is designed for robust and interpretable results.

## Features

- **Two-stage cascade**: Coarse-to-fine segmentation for improved accuracy
- **Uncertainty-aware loss**: Adaptive training for boundary calibration
- **Anatomy-aware filtering**: Lung overlap and surface-based post-processing
- **Support for multiple models**: Swin UNETR, UNETR, UNet
- **Evaluation metrics**: Dice, HD95, boundary Dice

## Post-Processing Strategy

We provide a reproducible implementation of the component filtering pipeline used in our post-processing step. The algorithm filters predicted tumor components based on:

- **Size**: Components with fewer than 50 voxels are discarded.
- **Lung overlap**: Components with lung overlap below 80% are rejected unless located peripherally near the lung surface.
- **Distance to lung boundary**: Peripheral components within 5 voxels of the lung are allowed if sufficiently large.
- **Mediastinal exclusion zone**: For components located in the central XY zone, a stricter overlap threshold is enforced to suppress false positives.
- **Component merging**: Binary dilation with a (2,2,2) kernel is applied prior to connected component analysis to merge fragmented regions.
- **Top-K filtering (optional)**: The top-K components are retained by voxel volume, with K typically set to 1.

## Citation

If you use this codebase, please cite:

@article{isler2025uncertainty,
  title={Uncertainty-Guided Coarse-to-Fine Tumor Segmentation with Anatomy-Aware Post-Processing},
  author={Isler, Ilkin Sevgi and Mohaisen, David and Lisle, Curtis and Turgut, Damla and Bagci, Ulas},
  journal={arXiv preprint arXiv:2504.12215},
  year={2025}
}
