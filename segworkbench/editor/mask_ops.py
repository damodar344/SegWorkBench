"""Mask editing geometry and state management."""

import numpy as np


class MaskBuffer:
    """Hold a binary mask array with undo capability."""

    def __init__(self, height: int, width: int, max_undo: int = 20):
        self._height = height
        self._width = width
        self._max_undo = max_undo
        self.array = np.zeros((height, width), dtype=np.uint8)
        self._undo_stack: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Undo / Redo
    # ------------------------------------------------------------------ #
    def push_undo(self) -> None:
        """Save the current mask state for undo."""
        self._undo_stack.append(self.array.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore the previous mask state. Returns ``True`` if an undo was performed."""
        if not self._undo_stack:
            return False
        self.array = self._undo_stack.pop()
        return True

    def clear(self) -> None:
        """Zero out the mask."""
        self.array[:, :] = 0

    # ------------------------------------------------------------------ #
    # Brush primitives (instance methods that operate on internal array)
    # ------------------------------------------------------------------ #
    def draw_circle(self, cx: int, cy: int, r: int, value: int) -> None:
        """Paint a filled circle on the mask."""
        draw_circle(self.array, cx, cy, r, value)

    def draw_line_circles(self, x0: int, y0: int, x1: int, y1: int, r: int, value: int) -> None:
        """Paint a line stroke as a sequence of circles on the mask."""
        draw_line_circles(self.array, x0, y0, x1, y1, r, value)


def draw_circle(arr: np.ndarray, cx: int, cy: int, r: int, val: int) -> None:
    """Draw a filled circle into *arr*."""
    h, w = arr.shape
    x0 = max(0, cx - r)
    x1 = min(w - 1, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h - 1, cy + r)
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    arr[y0 : y1 + 1, x0 : x1 + 1][m] = val


def draw_line_circles(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, val: int) -> None:
    """Interpolate a straight line and stamp circles along it."""
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy), 1))
    for i in range(steps + 1):
        t = i / steps
        cx = int(x0 + dx * t)
        cy = int(y0 + dy * t)
        draw_circle(arr, cx, cy, r, val)
