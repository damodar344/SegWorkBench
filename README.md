# SegWorkbench

A unified desktop application for deep-learning-based and interactive image segmentation.

---

## Features

- **UNet Training** — Train a U-Net model with live **train/validation loss** and **Dice/IoU metric** plots, pause/resume/stop controls, automatic best-model checkpointing via `CheckpointManager`, **data augmentation** (flip/rotation), **ReduceLROnPlateau** scheduling, and **early stopping**.
- **Batch Inference** — Run segmentation on a folder of images using a saved checkpoint. Outputs are restored to the **original image resolution**. Supports configurable probability threshold, variable model input size, and **16-bit/grayscale preservation** for scientific images.
- **Statistical Thresholding** — Gaussian Mixture Model (GMM), Otsu, Triangle, and IsoData thresholding with real-time histogram overlay.
- **Interactive Mask Editor** — Brush / eraser tools, zoom & pan, undo/redo, and transparent PNG export.

---

## Installation

### 1. Clone or download the repository

```bash
cd segworkbench
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core requirements include **PySide6**, **PyTorch**, **torchvision**, **NumPy**, **Pillow**, **scikit-learn**, **scikit-image**, **matplotlib**, and **pandas**.

---

## Running the Application

### Recommended: using the `main.py` entry point

```bash
python main.py
```

### Alternative: as a module

```bash
python -m segworkbench
```

The application launches a dark-themed Qt window with four tabs:

1. **Training (UNet)** — Dataset selection, hyperparameters, data-loader workers, and live **dual plots** (loss + Dice/IoU). Training runs in a background `QThread` with thread-safe signal updates.
2. **Inference** — Batch model inference with configurable input size, probability threshold, and 16-bit depth preservation. Outputs are resized back to original image resolution.
3. **GMM** — Image thresholding and histogram analysis with GMM component curves.
4. **Mask Editor** — Interactive binary mask drawing and export.

---

## Directory Structure

```
segworkbench/
├── configs/          # Model configuration & checkpoint utilities
├── editor/           # Mask editing geometry & undo buffer
├── engine/           # Training & inference back-ends (no GUI)
├── gui/              # PySide6 tabs, widgets, stylesheets, and QThread workers
│   ├── tabs/
│   ├── widgets/
│   ├── styles/
│   └── workers.py    # TrainingWorker & InferenceWorker (QThread)
├── models/           # UNet architecture
├── analysis/         # GMM & threshold algorithms
├── utils/            # General image processing helpers
└── __main__.py

legacy/               # Archived original tkinter scripts
image/                # Sample images
masks/                # Sample masks
main.py               # Primary entry point (thin wrapper)
requirements.txt
```

---

## Dataset Structure

For training and inference, place images and masks in two parallel folders:

```
images/
  ├── 0001.tif
  ├── 0002.tif
  └── ...
masks/
  ├── 0001.tif
  ├── 0002.tif
  └── ...
```

Each image must have a corresponding mask with the **same base filename**.

Supported formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`

---

## Training Tab

### Hyperparameters

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| Epochs | 1 – 10000 | 5 | Training epochs |
| Batch Size | 1 – 256 | 2 | Images per batch |
| Learning Rate | 1e-8 – 1.0 | 1e-4 | AdamW learning rate |
| Workers | 0 – 16 | 0 | DataLoader workers (0 = main-thread loading) |

### Features

- **Live dual plots** — Left: Train/Val loss curves. Right: Dice/IoU score curves.
- **Data augmentation** — Random horizontal flip, vertical flip, and ±15° rotation applied to training data only.
- **Checkpointing** — Every time validation loss improves, a checkpoint is saved via `CheckpointManager` (keeps last 3). The best model is also saved at the end.
- **LR Scheduling** — `ReduceLROnPlateau` halves the learning rate when validation loss stalls.
- **Early Stopping** — Training halts if validation loss does not improve for `patience` epochs (`patience` and `min_delta` are read from `ModelConfig`).
- **Pause / Resume / Stop** — Cooperative cancellation and pause via `QThread` events.
- **Device auto-detection** — Prefers CUDA, then Apple Silicon MPS, then CPU. On CPU, PyTorch threads are capped to 4 to keep the GUI responsive.

---

## Inference Tab

### Options

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| Save as | `.png` / `.tif` | `.png` | Output format |
| Image size | 64 – 2048 | 256 | Square size fed to the model |
| Threshold | 0.0 – 1.0 | 0.5 | Probability threshold for binarisation |
| Preserve 16-bit | On / Off | Off | Avoid forcing 8-bit RGB conversion for grayscale/16-bit images |

### Features

- **Original resolution restoration** — The model runs at the configured square size, then the probability map is bilinearly interpolated back to the original image dimensions before thresholding.
- **16-bit / grayscale support** — When enabled, `L`, `I`, `I;16`, and `F` mode images are normalized directly without clamping to 8-bit RGB.
- **Progress tracking** — Status label shows `Processed N/M: filename`.
- **Cooperative cancellation** — Stop button aborts between images.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F11` | Toggle fullscreen |
| `Escape` | Exit fullscreen |
| `Ctrl+Q` | Exit application |
| `Ctrl+O` | Open image (Mask Editor) |
| `Ctrl+S` | Save mask PNG (Mask Editor) |
| `Ctrl+Shift+S` | Save transparent PNG (Mask Editor) |
| `Ctrl+Z` | Undo last stroke (Mask Editor) |

---

## Logging

Structured Python `logging` is used throughout the engine modules. To see detailed output when running from a terminal, set the level before launching:

```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG)" -m segworkbench
```

Or in your own script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Citation

If you use SegWorkbench in your research, please cite our paper:

```bibtex
@inproceedings{dhital2026segworkbench,
  title={SegWorkbench: A Unified Framework for Deep Learning-Based and Interactive Image Segmentation},
  author={Dhital, Damodar and Pokhrel, Abishek and Ashu, Favour and Darm, Vir Chuy and Qiao, Guanda and Zhang, Lei},
  booktitle={2026 IEEE/ACIS 24th International Conference on Software Engineering Research, Management and Applications (SERA)},
  pages={231--236},
  year={2026},
  publisher={IEEE},
  doi={10.1109/SERA69989.2026.11618632}
}
```

---

## Authors

- Damodar Dhital — Towson University  
- Advisor: Dr. Lei Zhang
