"""Tab 4 — Mask Editor."""

import os

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
)

from segworkbench.editor import MaskBuffer, draw_circle, draw_line_circles
from segworkbench.gui.base_tab import BaseTab


class _EditorView(QGraphicsView):
    """Interactive image viewer with brush/eraser editing on a full-res mask."""

    status_changed = Signal(str)
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)

        # Items
        self._bg_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._cursor_item = QGraphicsEllipseItem()
        self._cursor_item.setPen(QPen(QColor("#00ff00"), 2))
        self._cursor_item.setBrush(Qt.NoBrush)
        self._cursor_item.setVisible(False)

        self._scene.addItem(self._bg_item)
        self._scene.addItem(self._overlay_item)
        self._scene.addItem(self._cursor_item)

        # State
        self._full_img: Image.Image | None = None
        self._mask_buffer: MaskBuffer | None = None
        self._tool = "brush"
        self._brush_size = 18
        self._opacity = 0.45
        self._is_drawing = False
        self._last_scene_pos = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_image(self, pil_img: Image.Image):
        """Set a new background image (resets mask on size mismatch)."""
        if self._full_img is not None and self._mask_buffer is not None:
            old_h, old_w = self._mask_buffer.array.shape
            if (old_h, old_w) != (pil_img.height, pil_img.width):
                self._mask_buffer = None

        self._full_img = pil_img
        self._bg_item.setPixmap(self._pil_to_pixmap(pil_img))
        self._scene.setSceneRect(0, 0, pil_img.width, pil_img.height)

        if self._mask_buffer is None:
            self._new_blank_mask()

        self.fit_to_window()
        self._update_status()

    def load_mask(self, mask_pil: Image.Image):
        """Replace the current mask with an external one (resized to fit)."""
        if self._full_img is None:
            return
        if (mask_pil.width, mask_pil.height) != (
            self._full_img.width,
            self._full_img.height,
        ):
            mask_pil = mask_pil.resize(
                (self._full_img.width, self._full_img.height), Image.NEAREST
            )
        self._push_undo()
        self._mask_buffer.array = (
            np.array(mask_pil, dtype=np.uint8) >= 128
        ).astype(np.uint8) * 255
        self._update_overlay()
        self._update_status()

    def new_blank_mask(self):
        """Clear the mask (with undo)."""
        if self._full_img is None:
            return
        self._push_undo()
        self._new_blank_mask()
        self._update_overlay()
        self._update_status()

    def clear_mask(self):
        """Zero the mask without undo push (undo already saved beforehand)."""
        if self._mask_buffer is None:
            return
        self._mask_buffer.clear()
        self._update_overlay()
        self._update_status()

    def undo(self):
        """Restore the previous mask state."""
        if self._mask_buffer is None:
            return
        if self._mask_buffer.undo():
            self._update_overlay()
            self._update_status()

    @property
    def mask_array(self):
        """Current full-res mask or ``None``."""
        return self._mask_buffer.array if self._mask_buffer is not None else None

    @property
    def image(self):
        return self._full_img

    # Tool / display properties
    def set_tool(self, tool: str):
        self._tool = tool

    def set_brush_size(self, size: int):
        self._brush_size = max(1, size)
        self._update_cursor()

    def set_opacity(self, opacity: float):
        self._opacity = max(0.0, min(1.0, opacity))
        self._update_overlay()

    def fit_to_window(self):
        if self._full_img is None:
            return
        # Fit the background item into the viewport
        self.fitInView(self._bg_item, Qt.KeepAspectRatio)
        self.zoom_changed.emit(self.transform().m11())

    def set_explicit_zoom(self, zoom: float):
        self.resetTransform()
        self.scale(zoom, zoom)
        self._update_cursor()

    # ------------------------------------------------------------------ #
    # Event overrides
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._full_img is not None:
            self._is_drawing = True
            self._push_undo()
            pos = self.mapToScene(event.pos())
            self._apply_brush(pos.x(), pos.y())
            self._last_scene_pos = pos
            self._update_status()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        if self._is_drawing and self._last_scene_pos is not None:
            self._apply_brush(
                pos.x(), pos.y(), from_xy=(self._last_scene_pos.x(), self._last_scene_pos.y())
            )
            self._last_scene_pos = pos
            self._update_status()
        self._update_cursor(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_drawing = False
            self._last_scene_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self._full_img is None:
            return
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else (1 / 1.1)
        current = self.transform().m11()
        new_scale = current * factor
        if new_scale < 0.25:
            factor = 0.25 / current
        elif new_scale > 4.0:
            factor = 4.0 / current
        self.scale(factor, factor)
        self.zoom_changed.emit(self.transform().m11())
        self._update_cursor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep image centered; fit-to-window is explicit via button

    def leaveEvent(self, event):
        self._cursor_item.setVisible(False)
        super().leaveEvent(event)

    def enterEvent(self, event):
        self._cursor_item.setVisible(True)
        super().enterEvent(event)

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _apply_brush(self, x, y, from_xy=None):
        if self._mask_buffer is None or self._full_img is None:
            return
        w, h = self._full_img.width, self._full_img.height
        ix = int(x)
        iy = int(y)
        if not (0 <= ix < w and 0 <= iy < h):
            return

        view_scale = self.transform().m11()
        r = max(1, int(self._brush_size / view_scale))
        val = 255 if self._tool == "brush" else 0

        if from_xy is not None:
            fx, fy = int(from_xy[0]), int(from_xy[1])
            draw_line_circles(self._mask_buffer.array, fx, fy, ix, iy, r, val)
        else:
            draw_circle(self._mask_buffer.array, ix, iy, r, val)

        self._update_overlay()

    def _new_blank_mask(self):
        if self._full_img is None:
            return
        h, w = self._full_img.height, self._full_img.width
        self._mask_buffer = MaskBuffer(h, w)

    def _push_undo(self):
        if self._mask_buffer is not None:
            self._mask_buffer.push_undo()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _update_overlay(self):
        if self._mask_buffer is None:
            return
        pix = self._render_overlay_pixmap(self._mask_buffer.array, self._opacity)
        self._overlay_item.setPixmap(pix)

    def _render_overlay_pixmap(self, mask: np.ndarray, opacity: float) -> QPixmap:
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        fill_alpha = int(opacity * 255)
        overlay[:, :, 0] = 255
        overlay[:, :, 3] = ((mask > 0).astype(np.uint8) * fill_alpha)

        mm = (mask > 0).astype(np.uint8)
        up = np.pad(mm[1:, :], ((0, 1), (0, 0)), mode="constant")
        dn = np.pad(mm[:-1, :], ((1, 0), (0, 0)), mode="constant")
        lf = np.pad(mm[:, 1:], ((0, 0), (0, 1)), mode="constant")
        rt = np.pad(mm[:, :-1], ((0, 0), (1, 0)), mode="constant")
        edge = (mm == 1) & ((up + dn + lf + rt) < 4)
        overlay[edge, 0] = 0
        overlay[edge, 1] = 255
        overlay[edge, 2] = 0
        overlay[edge, 3] = 255

        qimg = QImage(
            overlay.tobytes(), w, h, w * 4, QImage.Format_RGBA8888
        ).copy()
        return QPixmap.fromImage(qimg)

    def _pil_to_pixmap(self, pil_img: Image.Image) -> QPixmap:
        arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        h, w, _ = arr.shape
        qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _update_cursor(self, x=None, y=None):
        if x is None or y is None:
            # Called when zoom changes — keep cursor at last known mouse position
            # Fallback: hide it; it will reappear on next move event
            return

        view_scale = self.transform().m11()
        r = self._brush_size / view_scale
        self._cursor_item.setRect(x - r, y - r, r * 2, r * 2)
        self._cursor_item.setVisible(True)

    def _update_status(self):
        if self._full_img is None or self._mask_buffer is None:
            self.status_changed.emit("Idle")
            return
        total = int(self._mask_buffer.array.size)
        on = int(np.count_nonzero(self._mask_buffer.array))
        pct = (100.0 * on / total) if total else 0.0
        self.status_changed.emit(f"Mask pixels: {on} ({pct:.2f}%)")


class MaskEditorTab(BaseTab):
    """Interactive binary mask editor tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Paths -------------------------------------------------------
        paths_group = QGroupBox("Files")
        v_paths = QVBoxLayout(paths_group)

        # Image row
        h_img = QHBoxLayout()
        h_img.addWidget(QLabel("Image:"))
        self._edit_img = QLineEdit()
        h_img.addWidget(self._edit_img, stretch=1)
        btn_browse_img = QPushButton("Browse")
        btn_browse_img.clicked.connect(self._pick_image)
        h_img.addWidget(btn_browse_img)
        btn_load_img = QPushButton("Load")
        btn_load_img.clicked.connect(self._load_image)
        h_img.addWidget(btn_load_img)
        v_paths.addLayout(h_img)

        # Mask row
        h_msk = QHBoxLayout()
        h_msk.addWidget(QLabel("Mask:"))
        self._edit_msk = QLineEdit()
        h_msk.addWidget(self._edit_msk, stretch=1)
        btn_browse_msk = QPushButton("Browse")
        btn_browse_msk.clicked.connect(self._pick_mask)
        h_msk.addWidget(btn_browse_msk)
        btn_load_msk = QPushButton("Load")
        btn_load_msk.clicked.connect(self._load_mask)
        h_msk.addWidget(btn_load_msk)
        btn_new_msk = QPushButton("New Blank Mask")
        btn_new_msk.clicked.connect(self._new_blank_mask)
        h_msk.addWidget(btn_new_msk)
        v_paths.addLayout(h_msk)

        # Output row
        h_out = QHBoxLayout()
        h_out.addWidget(QLabel("Output Folder:"))
        self._edit_out = QLineEdit()
        h_out.addWidget(self._edit_out, stretch=1)
        btn_browse_out = QPushButton("Browse")
        btn_browse_out.clicked.connect(self._pick_out)
        h_out.addWidget(btn_browse_out)
        btn_save_png = QPushButton("Save Mask PNG")
        btn_save_png.clicked.connect(self._save_mask_png)
        h_out.addWidget(btn_save_png)
        btn_save_trans = QPushButton("Save Transparent PNG")
        btn_save_trans.clicked.connect(self._save_transparent_png)
        h_out.addWidget(btn_save_trans)
        v_paths.addLayout(h_out)

        layout.addWidget(paths_group)

        # --- Tools -------------------------------------------------------
        tools_layout = QHBoxLayout()

        tools_layout.addWidget(QLabel("Tool:"))
        self._rb_brush = QRadioButton("Brush")
        self._rb_brush.setChecked(True)
        self._rb_eraser = QRadioButton("Eraser")
        tools_layout.addWidget(self._rb_brush)
        tools_layout.addWidget(self._rb_eraser)
        self._rb_brush.toggled.connect(self._on_tool_changed)

        tools_layout.addSpacing(20)
        tools_layout.addWidget(QLabel("Brush size:"))
        self._slider_brush = QSlider(Qt.Horizontal)
        self._slider_brush.setRange(1, 120)
        self._slider_brush.setValue(18)
        self._slider_brush.valueChanged.connect(self._on_brush_changed)
        tools_layout.addWidget(self._slider_brush, stretch=1)

        tools_layout.addSpacing(20)
        tools_layout.addWidget(QLabel("Opacity:"))
        self._slider_opacity = QSlider(Qt.Horizontal)
        self._slider_opacity.setRange(5, 95)
        self._slider_opacity.setValue(45)
        self._slider_opacity.valueChanged.connect(self._on_opacity_changed)
        tools_layout.addWidget(self._slider_opacity, stretch=1)

        tools_layout.addSpacing(20)
        tools_layout.addWidget(QLabel("Zoom:"))
        self._slider_zoom = QSlider(Qt.Horizontal)
        self._slider_zoom.setRange(25, 400)
        self._slider_zoom.setValue(100)
        self._slider_zoom.valueChanged.connect(self._on_zoom_changed)
        tools_layout.addWidget(self._slider_zoom, stretch=1)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(self._fit_to_window)
        tools_layout.addWidget(btn_fit)

        btn_undo = QPushButton("Undo")
        btn_undo.setShortcut("Ctrl+Z")
        btn_undo.clicked.connect(self._undo)
        tools_layout.addWidget(btn_undo)

        btn_clear = QPushButton("Clear Mask")
        btn_clear.clicked.connect(self._clear_mask)
        tools_layout.addWidget(btn_clear)

        layout.addLayout(tools_layout)

        # --- Status ------------------------------------------------------
        self._status = QLabel("Idle")
        layout.addWidget(self._status)

        # --- Canvas ------------------------------------------------------
        self._view = _EditorView()
        self._view.status_changed.connect(self._status.setText)
        self._view.zoom_changed.connect(self._sync_zoom_slider)
        layout.addWidget(self._view, stretch=1)

        # --- Shortcuts ---------------------------------------------------
        QShortcut(
            QKeySequence("Ctrl+O"), self, activated=self._load_image
        )
        QShortcut(
            QKeySequence("Ctrl+S"), self, activated=self._save_mask_png
        )
        QShortcut(
            QKeySequence("Ctrl+Shift+S"), self, activated=self._save_transparent_png
        )
        QShortcut(
            QKeySequence("Ctrl+Z"), self, activated=self._undo
        )

    # ------------------------------------------------------------------ #
    # Slot helpers
    # ------------------------------------------------------------------ #
    def _on_tool_changed(self, checked):
        if self._rb_brush.isChecked():
            self._view.set_tool("brush")
        else:
            self._view.set_tool("eraser")

    def _on_brush_changed(self, val: int):
        self._view.set_brush_size(val)

    def _on_opacity_changed(self, val: int):
        self._view.set_opacity(val / 100.0)

    def _on_zoom_changed(self, val: int):
        zoom = val / 100.0
        self._view.set_explicit_zoom(zoom)

    def _sync_zoom_slider(self, zoom: float):
        self._slider_zoom.blockSignals(True)
        self._slider_zoom.setValue(int(round(zoom * 100)))
        self._slider_zoom.blockSignals(False)

    def _fit_to_window(self):
        self._view.fit_to_window()

    def _undo(self):
        self._view.undo()

    def _clear_mask(self):
        self._view._push_undo()  # save before clear
        self._view.clear_mask()

    # ------------------------------------------------------------------ #
    # File operations
    # ------------------------------------------------------------------ #
    def _pick_image(self):
        p = self.ask_open_file(
            "Pick image",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif);;All files (*.*)",
        )
        if p:
            self._edit_img.setText(p)

    def _pick_mask(self):
        p = self.ask_open_file(
            "Pick mask",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif);;All files (*.*)",
        )
        if p:
            self._edit_msk.setText(p)

    def _pick_out(self):
        d = self.ask_directory("Pick output folder")
        if d:
            self._edit_out.setText(d)

    def _load_image(self):
        p = self._edit_img.text().strip()
        if not p or not os.path.isfile(p):
            self.show_error("Mask Editor", "Please select a valid image file.")
            return
        try:
            img = Image.open(p).convert("RGB")
            self._view.set_image(img)
        except Exception as exc:
            self.show_error("Mask Editor", f"Failed to load image: {exc}")
            return

    def _load_mask(self):
        if self._view.image is None:
            self.show_error("Mask Editor", "Load an image first.")
            return
        p = self._edit_msk.text().strip()
        if not p or not os.path.isfile(p):
            self.show_error("Mask Editor", "Please select a valid mask file.")
            return
        try:
            msk = Image.open(p).convert("L")
            self._view.load_mask(msk)
        except Exception as exc:
            self.show_error("Mask Editor", f"Failed to load mask: {exc}")

    def _new_blank_mask(self):
        if self._view.image is not None:
            self._view.new_blank_mask()
        else:
            self.show_error("Mask Editor", "Load an image first.")

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def _get_out_dir(self) -> str:
        outd = self._edit_out.text().strip()
        if not outd:
            p = self._edit_img.text().strip()
            if p:
                outd = os.path.dirname(p)
            else:
                outd = os.getcwd()
        os.makedirs(outd, exist_ok=True)
        return outd

    def _save_mask_png(self):
        arr = self._view.mask_array
        if arr is None:
            self.show_error("Mask Editor", "No mask to save.")
            return
        outd = self._get_out_dir()
        base = "edited_mask"
        p = self._edit_img.text().strip()
        if p:
            base = os.path.splitext(os.path.basename(p))[0] + "_mask_edited"
        out_path = os.path.join(outd, f"{base}.png")
        Image.fromarray(arr, mode="L").save(out_path)
        self.show_info("Mask Editor", f"Saved mask PNG:\n{out_path}")

    def _save_transparent_png(self):
        img = self._view.image
        arr = self._view.mask_array
        if img is None or arr is None:
            self.show_error("Mask Editor", "Load an image and mask first.")
            return
        outd = self._get_out_dir()
        base = "transparent"
        p = self._edit_img.text().strip()
        if p:
            base = os.path.splitext(os.path.basename(p))[0] + "_transparent"
        out_path = os.path.join(outd, f"{base}.png")
        rgba = img.convert("RGBA")
        np_img = np.array(rgba, dtype=np.uint8)
        np_img[:, :, 3] = arr
        Image.fromarray(np_img, mode="RGBA").save(out_path)
        self.show_info("Mask Editor", f"Saved transparent PNG:\n{out_path}")
