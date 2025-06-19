# Coarse-to-Fine Tumor Segmentation with Uncertainty and Anatomical Post-Processing

This repository implements a two-stage segmentation framework for lung tumors in CT scans, combining uncertainty modeling and anatomically informed post-processing. The pipeline includes coarse full-volume segmentation followed by refined ROI-based segmentation, and is designed for robust and interpretable results.

---

## Features

- **Two-stage cascade**: Coarse-to-fine segmentation for improved accuracy
- **Uncertainty-aware loss**: Adaptive training for boundary calibration
- **Anatomy-aware filtering**: Lung overlap and surface-based post-processing
- **Support for multiple models**: Swin UNETR, UNETR, UNet
- **Evaluation metrics**: Dice, HD95, boundary Dice

---

## Citation

If you use this codebase, please cite:

@inproceedings{isler2025coarse2fine,
  title={Uncertainty-Guided Coarse-to-Fine Tumor Segmentation with Anatomy-Aware Post-Processing},
  author={Isler, Ilkin and Mohaisen, David and Lisle, Curtis and Turgut, Damla and Bagci, Ulas},
  booktitle={ACDSA},
  year={2025}
}
