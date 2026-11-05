# SegWorkbench: Tkinter → PySide6 Migration Plan

## 1. Executive Summary

Migrate the single-file `Main app.py` (~1,200 lines of tkinter) into a modular **PySide6** desktop application. PySide6 is chosen over Streamlit because the Mask Editor requires a high-performance interactive canvas (brush, eraser, zoom, undo) that Qt’s `QGraphicsView`/`QGraphicsScene` handles natively, whereas Streamlit would require fragile custom HTML/JS components.

---

## 2. Recommended Target Stack

| Layer | Current | Target |
|-------|---------|--------|
| GUI Framework | tkinter / ttk | **PySide6** (Qt for Python) |
| Canvas / Image Editor | tkinter `Canvas` + `ImageTk` | `QGraphicsView` + `QGraphicsPixmapItem` |
| Matplotlib Embedding | `FigureCanvasTkAgg` | `FigureCanvasQTAgg` |
| Threading | Python `threading` | Keep `threading` or upgrade to `QThread` |
| Layout | tkinter grid/pack | Qt `QVBoxLayout`, `QHBoxLayout`, `QGridLayout` |
| File Dialogs | `tkinter.filedialog` | `QFileDialog` |
| Messages | `tkinter.messagebox` | `QMessageBox` |
| Packaging | none | `PyInstaller` or `cx_Freeze` |

---

## 3. Phase-by-Phase Roadmap

### Phase 0: Preparation & Branching (1 day)
1. **Create a feature branch**: `git checkout -b pyside6-migration`.
2. **Pin environment**: copy `requirements.txt` to `requirements-tkinter.txt` for legacy reference.
3. **Add PySide6 dependency**: append `pyside6>=6.6.0` and `matplotlib` (already present) to a new `requirements.txt`.
4. **Smoke-test install**: create a minimal `hello_qt.py` to verify PySide6 loads and the platform plugin works on the target OS.

### Phase 1: Structural Refactoring (2–3 days)
Goal: decouple business logic from the UI before rewriting widgets.

1. **Extract the model pipeline** (`segworkbench/` already exists; expand it):
   - Move `SegDataset`, `_make_dataset`, `_train_loop`, and `_inference_loop` from `Main app.py` into a new module:  
     `segworkbench/engine/trainer.py`  
     `segworkbench/engine/inferencer.py`
   - Keep them as plain Python classes/functions with **no tkinter imports**.
   - Replace tkinter `StringVar`/`IntVar` state with plain Python attributes or `dataclasses`.
   - Use callback functions (e.g., `on_epoch_end`, `on_batch_end`) so the GUI can subscribe to updates without the engine knowing about Qt.

2. **Extract the GMM logic**:
   - Move `refit_gmm`, `compute_threshold`, `apply_segmentation` into  
     `segworkbench/analysis/gmm_segmenter.py`.

3. **Extract mask editor geometry**:
   - Move `_draw_circle`, `_draw_line_circles`, `_crop_to_content`, undo stack management into  
     `segworkbench/editor/mask_ops.py`.

4. **Introduce a settings/config module**:
   - `segworkbench/configs/app_config.py` — persist last-used folders, window geometry, and theme.

### Phase 2: Core Application Shell (1 day)
Build the Qt application scaffold.

1. Create `segworkbench/gui/app.py`:
   - Subclass `QMainWindow`.
   - Add `QTabWidget` as the central widget (replaces `ttk.Notebook`).
   - Replicate window-title, size, and fullscreen keybindings (`F11`, `Escape`).
   - Add a `QStatusBar` for global messages.

2. Create a base tab class `segworkbench/gui/base_tab.py`:
   - Inherit from `QWidget`.
   - Provide common helpers: `show_info()`, `show_error()`, `browse_folder()`, `browse_file()`, `run_in_thread()`.

3. **Theme / stylesheet** (optional but recommended):
   - Create `segworkbench/gui/styles/dark.qss` to match the current `#1e1e1e` dark canvas look.

### Phase 3: Port Tab 1 — Training (2 days)
1. **Layout**:
   - `QGroupBox` for dataset controls (Image / Mask / Save folders).
   - `QSpinBox` (Epochs, Batch Size), `QDoubleSpinBox` (Learning Rate).
   - `QPushButton` for Start / Pause / Resume / Stop.
   - `QLabel` for status text.
2. **Matplotlib**:
   - Embed via `FigureCanvasQTAgg` inside a `QVBoxLayout`.
   - Keep the same `ax1` line-plot logic; only the canvas container changes.
3. **Threading**:
   - ~~Keep the existing `threading.Thread` approach initially (low risk).~~
   - ~~Communicate back to the GUI using `QMetaObject.invokeMethod(..., Qt.QueuedConnection)` or a simple `pyqtSignal`-based wrapper.~~
   - ✅ *Completed*: moved to `QThread` + `pyqtSignal` for cleaner cancellation and progress reporting.
4. **State management**:
   - Replace `tk.StringVar`/`tk.IntVar` with direct widget accessors (`spinbox.value()`, `line_edit.text()`) or a small `TrainingConfig` dataclass.

### Phase 4: Port Tab 2 — Inference (1 day)
1. **Layout**:
   - Similar group-box layout for Model / Input / Output folders.
   - `QComboBox` for output format (`.png`, `.tif`).
2. **Threading**:
   - Same pattern as Training: run `_inference_loop` in a thread, emit progress strings back to a `QLabel`.
3. **Callbacks**:
   - `on_file_processed(fname, i, total)` updates the status label.
   - `on_complete(dur, out_dir)` shows a `QMessageBox.information`.

### Phase 5: Port Tab 3 — GMM (1–2 days)
1. **Layout**:
   - Horizontal splitter (`QSplitter`) with image on the left, histogram on the right.
   - `QSpinBox` for Gaussians.
   - `QRadioButton` group for GMM / Otsu / Triangle / IsoData.
   - `QSlider` + `QLineEdit` for threshold value.
2. **Matplotlib**:
   - Two `FigureCanvasQTAgg` widgets inside the splitter.
3. **Interactions**:
   - Connect `valueChanged` signals to the same `compute_threshold` / `refit_gmm` logic extracted in Phase 1.
   - Draw the green threshold line the same way (`axvline`), just on the Qt canvas.

### Phase 6: Port Tab 4 — Mask Editor (3–4 days) *(Highest Complexity)*
This tab benefits the most from Qt.

1. **Replace tkinter Canvas with `QGraphicsView`**:
   - `QGraphicsScene` holds the background image (`QPixmap`) and the mask overlay (`QGraphicsPixmapItem` with alpha).
   - Implement the red fill + green edge overlay as a second `QGraphicsPixmapItem` on top.
2. **Brush / Eraser**:
   - Override `mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent` on the `QGraphicsView` (or a custom `EditorView`).
   - Map mouse coordinates to image coordinates via the view transform.
   - Paint into a `QImage` (or a `numpy` buffer) using the same `_draw_circle` / `_draw_line_circles` algorithms.
   - After each stroke, regenerate the overlay pixmap and call `scene.update()`.
3. **Zoom**:
   - Override `wheelEvent` and call `scale(factor, factor)` on the `QGraphicsView`.
   - Add “Fit to Window” via `fitInView(..., Qt.KeepAspectRatio)`.
4. **Undo**:
   - Keep the same `undo_stack` of `numpy` arrays; no change needed except perhaps a `QAction` with `Ctrl+Z` shortcut.
5. **Cursor**:
   - Draw the brush cursor as a `QGraphicsEllipseItem` that follows the mouse; much simpler than manually redrawing on a tkinter canvas.
6. **Saving**:
   - `QFileDialog.getExistingDirectory` / `getSaveFileName`.
   - Reuse the same `save_mask_png` / `save_transparent_png` logic.

### Phase 7: Cleanup & Polish (1–2 days)
1. **Delete tkinter imports and dead code** from the old `Main app.py`.
2. **Rename / remove** `Main app.py`; create a new entry point:
   - `python -m segworkbench` or `python main.py` that imports and runs `segworkbench.gui.app`.
3. **Centralize icons / assets** (if any) into `segworkbench/gui/assets/`.
4. **Keyboard shortcuts**:
   - `Ctrl+O` open image, `Ctrl+S` save mask, `Ctrl+Z` undo, `F11` fullscreen.
5. **Error handling**:
   - Wrap long-running operations in try/except and show `QMessageBox.critical`.

### Phase 8: Testing & Packaging (2 days)
1. **Manual test matrix**:
   - Train a small model (2 epochs) on sample data.
   - Run inference on the resulting checkpoint.
   - Open a `.tif` in GMM, switch methods, adjust threshold slider.
   - Open an image in Mask Editor, draw strokes, zoom with mouse wheel, undo, save PNG + transparent PNG.
2. **Automated smoke tests** (optional but recommended):
   - Use `pytest-qt` to instantiate each tab widget and verify default states.
3. **Packaging**:
   - Add a `PyInstaller` spec file (`build.spec`) or `setup.py` console-script entry point.
   - Ensure the `image/` sample folder is bundled correctly if distributed.

---

## 4. New Directory Structure (Post-Migration)

```
segworkbench/
├── __init__.py
├── __main__.py                     # new entry point
├── configs/
│   ├── __init__.py
│   ├── model_config.py             # existing
│   └── app_config.py               # NEW: window state, recent folders
├── engine/
│   ├── __init__.py
│   ├── trainer.py                  # NEW: extracted TrainingTab logic
│   └── inferencer.py               # NEW: extracted InferenceTab logic
├── analysis/
│   ├── __init__.py
│   └── gmm_segmenter.py            # NEW: extracted GMM logic
├── editor/
│   ├── __init__.py
│   └── mask_ops.py                 # NEW: brush geometry, undo, overlays
├── models/
│   ├── __init__.py
│   └── unet.py                     # existing
└── gui/
    ├── __init__.py
    ├── app.py                      # NEW: QMainWindow + QTabWidget
    ├── base_tab.py                 # NEW: shared dialogs, threading helpers
    ├── styles/
    │   └── dark.qss                # NEW: optional Qt stylesheet
    ├── tabs/
    │   ├── __init__.py
    │   ├── training_tab.py         # NEW: Tab 1
    │   ├── inference_tab.py        # NEW: Tab 2
    │   ├── gmm_tab.py              # NEW: Tab 3
    │   └── mask_editor_tab.py      # NEW: Tab 4 (QGraphicsView)
    └── widgets/
        ├── __init__.py
        └── mpl_canvas.py           # NEW: thin wrapper around FigureCanvasQTAgg

image/                              # existing sample images
requirements.txt                      # updated (adds pyside6)
requirements-tkinter.txt            # archived old deps
build.spec                          # NEW: PyInstaller spec (optional)
```

---

## 5. Risk Register & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mask Editor canvas performance regressions | High | Use `QImage`/`QPixmap` caching; redraw only the overlay, not the base image, on every stroke. |
| Threading deadlocks during training | Medium | Keep the simple `threading.Thread` + `QMetaObject.invokeMethod` bridge in Phase 3; upgrade to `QThread` only after stability is proven. |
| PySide6 not available on target platform | Low | PySide6 wheels are available for Windows, macOS, and Linux (x86_64 & ARM64). |
| matplotlib backend conflicts | Medium | Explicitly set `matplotlib.use("QtAgg")` before any figure creation. |
| Scope creep (rewriting model code) | Medium | Strict rule: **do not touch** `segworkbench.models.unet.py` or `model_config.py` during this migration. |

---

## 6. Decision Log

- **Why PySide6 and not PyQt6?**  
  PySide6 is the official Qt for Python project, uses the LGPL license, and avoids the GPL viral clause that can complicate distribution of the bundled application.

- **Why not Streamlit?**  
  Streamlit is excellent for dashboard-style ML apps, but the Mask Editor requires low-latency mouse drawing, pixel-perfect overlays, and custom zoom/pan. Replicating this in a browser-based framework would require significant custom HTML5 Canvas / Streamlit Components code, negating Streamlit’s simplicity advantage.

- **Why not Immediate Mode (Dear PyGui / ImGui)?**  
  While fast, immediate-mode libraries lack mature image-canvas widgets with built-in zoom/pan. The Mask Editor would still need a custom texture renderer, and the learning curve is steeper for the team.

---

## 7. Definition of Done

- [ ] `Main app.py` is deleted or archived.
- [ ] All 4 tabs are functional in PySide6 with equivalent or better UX.
- [ ] Training can be started, paused, resumed, and stopped; loss plot updates live.
- [ ] Inference runs in a background thread without freezing the UI.
- [ ] GMM tab displays histograms, fits models, and updates threshold line in real time.
- [ ] Mask Editor supports brush/eraser, mouse-wheel zoom, fit-to-window, undo, and saving PNG / transparent PNG.
- [ ] App runs from a single command: `python -m segworkbench`.
- [ ] `requirements.txt` installs cleanly into a fresh venv and the app launches.

---

*Plan version: 1.0*  
*Estimated total effort: 12–16 days for one developer*
