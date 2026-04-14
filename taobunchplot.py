#!/usr/bin/env python3
"""
Tao Bunch Distribution Viewer
==============================
PySide6 + Matplotlib GUI for visualising particle phase-space data from live
Tao/Bmad accelerator simulations via pytao.

Install dependencies:
    pip install PySide6 matplotlib numpy scipy pytao

Run:
    python taobunchplot.py

Tao itself must be installed separately; pytao is the Python binding.

Author: Randika Gamage  (randika@jlab.org)
Support: Absolutely not. Figure it out.
"""

import os
os.environ['QT_API'] = 'pyside6'

import sys
import json
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QLabel, QPushButton, QSlider, QComboBox,
    QCheckBox, QScrollArea, QFrame, QFileDialog, QMessageBox,
    QDialog, QDoubleSpinBox, QSpinBox, QTabWidget, QSizePolicy,
    QToolBar, QStatusBar, QGroupBox, QLineEdit, QRadioButton,
    QProgressBar, QTextEdit,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, QSize, QDateTime
from PySide6.QtGui import QColor, QPalette, QFont, QAction

# ── Constants ──────────────────────────────────────────────────────────────────

# Bmad coordinate names as they appear in pytao pipe bunch1 calls
BMAD_COORDS = ["x", "px", "y", "py", "z", "pz", "t"]

# Display names for the axis selectors (adds "z→t" as a derived option)
DISPLAY_COLS  = ["x", "px", "y", "py", "z", "pz", "t_derived"]
COL_LABELS    = {
    "x":         "x",
    "px":        "px",
    "y":         "y",
    "py":        "py",
    "z":         "z",
    "pz":        "pz",
    "t_derived": "t  [from z]",
}
COL_UNITS = {
    "x":         "m",
    "px":        "",
    "y":         "m",
    "py":        "",
    "z":         "m",
    "pz":        "",
    "t_derived": "s",
}

# Conjugate pairs for Twiss calculation
CONJUGATE_PAIRS = {("x", "px"), ("px", "x"), ("y", "py"), ("py", "y")}

# Default panel axis pairs
DEFAULT_PAIRS = [("x", "px"), ("y", "py"), ("z", "pz"), ("x", "y")]

# Speed of light
C_LIGHT = 299_792_458.0  # m/s

FILE_COLORS = [
    "#4f9ef0",  # blue
    "#f0904f",  # orange
    "#4ff0a0",  # green
    "#f04f90",  # pink
    "#c0a0f0",  # purple
    "#f0e04f",  # yellow
]

TRACK_COLORS = [
    "#ffffff", "#ffdd44", "#ff6644", "#44ff88",
    "#ff44ff", "#44ffff", "#ffaa22", "#aaffaa",
]

# Dark theme colours (matching the SDDS viewers)
BG      = "#1a1a2e"
AX_BG   = "#12122a"
GRID_C  = "#2a2a50"
TEXT_C  = "#c8cde4"
SPINE_C = "#333366"

_RMS95 = 2.4477  # sqrt(-2·ln(0.05)) for 95% containment ellipse


# ── Physics helpers ────────────────────────────────────────────────────────────

def z_to_t(z: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """
    Convert Bmad z (path-length deviation, m) to time deviation (s).

    Bmad definition:  z = -beta*c*(t - t_ref)
    => t_dev = -z / (beta * c)

    beta is estimated per-particle from pz (Δp/p):
        p/p0 = 1 + pz  =>  gamma*beta ≈ (1+pz) * gamma0*beta0
    For a relativistic approximation we use beta ≈ 1 when pz << 1,
    which is accurate enough for the conversion.  For a fully rigorous
    conversion the caller would need p0 (reference momentum), which we
    don't always have from pytao.  The approximation beta ≈ 1 introduces
    an error of order (1-beta) which is < 0.03 % for electrons above 1 MeV.
    """
    beta = np.ones_like(z)  # relativistic approximation
    return -z / (beta * C_LIGHT)


# ── Analysis helpers ───────────────────────────────────────────────────────────

def _compute_twiss(xd: np.ndarray, yd: np.ndarray):
    if len(xd) < 3:
        return None
    xc, yc = xd - xd.mean(), yd - yd.mean()
    s11 = float(np.mean(xc ** 2))
    s12 = float(np.mean(xc * yc))
    s22 = float(np.mean(yc ** 2))
    det = s11 * s22 - s12 ** 2
    if det <= 0:
        return None
    emit = float(np.sqrt(det))
    return {"emit": emit, "beta": s11 / emit,
            "alpha": -s12 / emit, "gamma": s22 / emit}


def _style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.grid(True, color=GRID_C, linewidth=0.4, zorder=0)


def _draw_overlay(ax, xd, yd, color, xn="x", yn="y"):
    xc = float(xd.mean())
    yc = float(yd.mean())
    xs = float(xd.std()) or 1e-10
    ys = float(yd.std()) or 1e-10

    # Crosshairs
    ax.axvline(xc, color=color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=5)
    ax.axhline(yc, color=color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=5)

    # Covariance ellipses
    xdev = xd - xd.mean()
    ydev = yd - yd.mean()
    cov  = np.array([[np.mean(xdev ** 2), np.mean(xdev * ydev)],
                     [np.mean(xdev * ydev), np.mean(ydev ** 2)]])
    vals, vecs = np.linalg.eigh(cov)
    vals  = np.maximum(vals, 0)
    theta = np.linspace(0, 2 * np.pi, 200)
    unit  = np.array([np.cos(theta), np.sin(theta)])
    basis = vecs @ (np.sqrt(vals)[:, None] * unit)

    ax.plot(xc + basis[0], yc + basis[1],
            color=color, linewidth=1.2, alpha=0.9, zorder=6)
    e95 = basis * _RMS95
    ax.plot(xc + e95[0], yc + e95[1],
            color=color, linewidth=1.0, alpha=0.45, linestyle="--", zorder=6)

    # Twiss box (conjugate pairs only)
    if (xn, yn) in CONJUGATE_PAIRS:
        tw = _compute_twiss(xd, yd)
        if tw:
            lines = (f"emit = {tw['emit']:.3g} m\n"
                     f"beta = {tw['beta']:.3g} m\n"
                     f"alph = {tw['alpha']:.3g}\n"
                     f"gamm = {tw['gamma']:.3g} /m")
            ax.text(0.02, 0.98, lines, transform=ax.transAxes,
                    va="top", ha="left", fontsize=10.5, family="monospace",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a18",
                              edgecolor=color, alpha=0.75), zorder=10)

    # σ text box top-right
    sigma_txt = f"σ_{xn} = {xs:.4g}\nσ_{yn} = {ys:.4g}\n[RMS]"
    ax.text(0.98, 0.98, sigma_txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=10.5, family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a18",
                      edgecolor="#555577", alpha=0.75), zorder=10)


# ── Qt Dark Palette ────────────────────────────────────────────────────────────

def apply_dark_palette(app: QApplication):
    palette = QPalette()
    dark    = QColor("#1a1a2e")
    mid     = QColor("#252540")
    light   = QColor("#2e2e50")
    text    = QColor("#c8cde4")
    hi      = QColor("#4f9ef0")
    hi_text = QColor("#ffffff")
    dis     = QColor("#555570")

    palette.setColor(QPalette.Window,          dark)
    palette.setColor(QPalette.WindowText,      text)
    palette.setColor(QPalette.Base,            mid)
    palette.setColor(QPalette.AlternateBase,   light)
    palette.setColor(QPalette.ToolTipBase,     dark)
    palette.setColor(QPalette.ToolTipText,     text)
    palette.setColor(QPalette.Text,            text)
    palette.setColor(QPalette.Button,          light)
    palette.setColor(QPalette.ButtonText,      text)
    palette.setColor(QPalette.BrightText,      hi_text)
    palette.setColor(QPalette.Highlight,       hi)
    palette.setColor(QPalette.HighlightedText, hi_text)
    palette.setColor(QPalette.Disabled, QPalette.Text,       dis)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, dis)
    app.setPalette(palette)

    app.setStyleSheet("""
        QWidget { font-size: 11px; }
        QMainWindow { background: #1a1a2e; }
        QPushButton {
            background: #252540; border: 1px solid #333366;
            border-radius: 4px; padding: 4px 10px; color: #c8cde4;
        }
        QPushButton:hover  { background: #2e2e60; }
        QPushButton:pressed { background: #1a1a50; }
        QPushButton:checked { background: #2a4a8a; border-color: #4f9ef0; }
        QPushButton:disabled { background: #1a1a2e; color: #444460; border-color: #252540; }
        QComboBox {
            background: #252540; border: 1px solid #333366;
            border-radius: 4px; padding: 2px 6px; color: #c8cde4;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background: #252540; color: #c8cde4;
            selection-background-color: #2a4a8a;
        }
        QSlider::groove:horizontal {
            height: 4px; background: #333366; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            width: 12px; height: 12px; margin: -4px 0;
            background: #4f9ef0; border-radius: 6px;
        }
        QSlider::sub-page:horizontal { background: #4f9ef0; border-radius: 2px; }
        QLabel { color: #c8cde4; }
        QGroupBox {
            border: 1px solid #333366; border-radius: 6px;
            margin-top: 8px; padding-top: 6px; color: #8ab0d8;
            font-weight: bold;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        QCheckBox { color: #c8cde4; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #333366; border-radius: 3px;
            background: #252540;
        }
        QCheckBox::indicator:checked { background: #4f9ef0; }
        QScrollArea { border: none; }
        QScrollBar:vertical {
            background: #1a1a2e; width: 8px; border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #333366; border-radius: 4px; min-height: 20px;
        }
        QLineEdit {
            background: #252540; border: 1px solid #333366;
            border-radius: 4px; padding: 2px 6px; color: #c8cde4;
        }
        QSplitter::handle { background: #333366; }
        QToolBar { background: #0f0f1e; border: none; spacing: 4px; }
        QStatusBar { background: #0f0f1e; color: #8ab0d8; }
        QFrame[frameShape="4"] { color: #333366; }
        QProgressBar {
            border: 1px solid #333366; border-radius: 3px;
            background: #252540; color: #c8cde4; text-align: center;
        }
        QProgressBar::chunk { background: #4f9ef0; border-radius: 2px; }
        QRadioButton { color: #c8cde4; }
    """)


# ── Slider helper ──────────────────────────────────────────────────────────────

# ── Scientific notation spinbox ───────────────────────────────────────────────

class SciSpinBox(QDoubleSpinBox):
    """
    A QDoubleSpinBox that displays values in scientific notation (e.g. 1.00e-06)
    and accepts user input in the same format.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(10)      # internal precision — not shown directly
        self.setRange(0.0, 1e10)
        self.setValue(1e-6)
        self.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)

    def textFromValue(self, value: float) -> str:
        if value == 0.0:
            return "0.00e+00"
        return f"{value:.2e}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.strip())
        except ValueError:
            return 0.0

    def validate(self, text: str, pos: int):
        # Accept anything that could be part of a valid float/sci-notation number
        import re
        from PySide6.QtGui import QValidator
        t = text.strip()
        # Allow partial input while typing
        if re.match(r'^-?[\d]*\.?[\d]*(e[+-]?[\d]*)?$', t, re.IGNORECASE) or t in ('', '-', '.', 'e', 'E'):
            return (QValidator.Acceptable if self._is_complete(t)
                    else QValidator.Intermediate, text, pos)
        return (QValidator.Invalid, text, pos)

    def _is_complete(self, t: str) -> bool:
        try:
            float(t)
            return True
        except ValueError:
            return False


def make_slider(mn, mx, val, decimals=0):
    s = QSlider(Qt.Horizontal)
    s.setMinimum(0)
    s.setMaximum(1000)
    s._mn       = mn
    s._mx       = mx
    s._decimals = decimals
    s.setValue(int((val - mn) / (mx - mn) * 1000))

    def real_value():
        v = s.value() / 1000.0 * (s._mx - s._mn) + s._mn
        return round(v, decimals) if decimals > 0 else int(round(v))

    s.real_value = real_value
    return s


# ── Collapsible sidebar section ────────────────────────────────────────────────

class SidebarSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._toggle = QPushButton(f"▾  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setStyleSheet(
            "QPushButton { background: #0f0f1e; border: none; "
            "text-align: left; padding: 4px 8px; color: #8ab0d8; "
            "font-weight: bold; font-size: 11px; }"
            "QPushButton:hover { background: #1a1a3a; }"
        )
        self._toggle.toggled.connect(self._on_toggle)
        layout.addWidget(self._toggle)

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 2, 8, 6)
        body_layout.setSpacing(4)
        layout.addWidget(self._body)
        self.body_layout = body_layout

    def _on_toggle(self, checked):
        self._toggle.setText(
            ("▾  " if checked else "▸  ") + self._toggle.text()[3:]
        )
        self._body.setVisible(checked)

    def add(self, widget):
        self.body_layout.addWidget(widget)

    def add_row(self, label_text, widget):
        row = QWidget()
        rl  = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #8ab0d8; font-size: 10px;")
        rl.addWidget(lbl)
        rl.addWidget(widget, 1)
        self.body_layout.addWidget(row)


# ── Tao data fetcher (runs in a worker thread) ─────────────────────────────────

class FetchWorker(QObject):
    """
    Worker that pulls bunch coordinates from a running Tao instance via pytao.

    Signals:
      progress(str)               -- log message for each step
      finished(data, meta, error) -- final result
    """
    progress = Signal(str)
    finished = Signal(dict, dict, str)

    def __init__(self, tao, element: str):
        super().__init__()
        self.tao     = tao
        self.element = element

    def _cmd(self, cmd: str):
        """Run a single tao command, check for Tao-level errors, emit progress."""
        self.progress.emit(f"  tao.cmd({cmd!r})")
        t0  = time.monotonic()
        raw = self.tao.cmd(cmd)
        dt  = time.monotonic() - t0
        _check_tao_error(raw, cmd)
        # Log raw type and content so we can debug parse issues
        raw_repr = repr(raw)[:200]
        self.progress.emit(f"    raw: type={type(raw).__name__} len={len(raw) if hasattr(raw,'__len__') else '?'}  {raw_repr}")
        preview = _response_preview(raw)
        self.progress.emit(f"    → {preview}  [{dt*1000:.0f} ms]")
        return raw

    def run(self):
        ele  = self.element
        data = {}
        meta = {}
        try:
            # ── Use pytao's native bunch1() method ────────────────────────
            # tao.bunch1(ele, coord) is the correct pytao API —
            # raw 'pipe bunch1' returns empty list even when beam exists.
            self.progress.emit(f"Fetching bunch data at {ele} via tao.bunch1()…")

            # Check state first to get particle count and alive mask
            self.progress.emit(f"  tao.bunch1({ele!r}, 'state')")
            t0 = time.monotonic()
            state_arr = self.tao.bunch1(ele, "state")
            dt = time.monotonic() - t0
            self.progress.emit(f"    raw: type={type(state_arr).__name__} len={len(state_arr) if hasattr(state_arr,'__len__') else '?'}  [{dt*1000:.0f} ms]")

            state = np.asarray(state_arr, dtype=int)
            alive = state == 1
            n_total = len(state)
            n_alive = int(alive.sum())
            self.progress.emit(f"  state: {n_total} particles, {n_alive} alive, {n_total-n_alive} lost")

            if n_alive == 0:
                self.progress.emit("  WARNING: no alive particles")
                data = {c: np.array([], dtype=np.float64) for c in BMAD_COORDS}
                data["t_derived"] = np.array([], dtype=np.float64)
                meta = {"element": ele, "n_alive": 0, "n_total": n_total, "n_lost": n_total}
                self.finished.emit(data, meta, "")
                return

            # Fetch each coordinate — store both filtered (for plotting) and
            # full unfiltered (for stable particle index tracking)
            full = {}  # original index arrays, NaN where particle is lost
            for coord in BMAD_COORDS:
                self.progress.emit(f"  tao.bunch1({ele!r}, {coord!r})")
                arr = np.asarray(self.tao.bunch1(ele, coord), dtype=np.float64)
                if len(arr) == n_total:
                    full[coord]  = arr.copy()
                    data[coord]  = arr[alive]
                elif len(arr) == n_alive:
                    # Tao already returned only alive — reconstruct full array
                    full_arr = np.full(n_total, np.nan)
                    full_arr[alive] = arr
                    full[coord]  = full_arr
                    data[coord]  = arr
                else:
                    self.progress.emit(f"    WARNING: unexpected length {len(arr)}")
                    full[coord]  = arr
                    data[coord]  = arr

            # Derive t from z
            data["t_derived"] = z_to_t(data["z"], data["pz"])
            if "z" in full and "pz" in full:
                full["t_derived"] = z_to_t(full["z"], full["pz"])

            # Store full arrays and state so caller can do stable-index tracking
            meta["_full"]  = full   # original-index arrays (NaN = lost)
            meta["_state"] = state  # int array: 1=alive, other=lost

            # Bunch params (non-critical)
            self.progress.emit("  tao.bunch_params()…")
            try:
                bp = self.tao.bunch_params(ele)
                meta = {k: v for k, v in bp.items()} if isinstance(bp, dict) else {}
            except Exception as e:
                self.progress.emit(f"    bunch_params failed: {e}")
                meta = {}

            meta["element"] = ele
            meta["n_alive"] = n_alive
            meta["n_total"] = n_total
            meta["n_lost"]  = n_total - n_alive

            self.progress.emit(f"Done — {n_alive:,} particles fetched")
            self.finished.emit(data, meta, "")

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.progress.emit(f"EXCEPTION: {exc}")
            self.progress.emit(tb)
            self.finished.emit({}, {}, str(exc))


def _check_tao_error(raw, cmd: str = ""):
    """
    pytao often returns Tao error messages as strings/lists rather than raising.
    Detect them and raise RuntimeError so the worker catches them cleanly.
    """
    if raw is None:
        return
    if isinstance(raw, (list, tuple)):
        check = " ".join(str(r) for r in raw[:5])
    else:
        check = str(raw)
    if "ERROR detected" in check or "[ERROR |" in check:
        import re
        m = re.search(r'tao_pipe_cmd:.*', check)
        msg = m.group(0) if m else check[:200]
        if "BEAM NOT SAVED" in check:
            raise RuntimeError(
                f"Beam not saved at this element.\n"
                f"Use 'Beam Setup… → Apply & Query' or click 'Re-run Tracking' "
                f"to track the beam through the lattice first.\n\n"
                f"(Tao: {msg})"
            )
        raise RuntimeError(f"Command: {cmd} causes error: {msg}")


def _first_error_line(msg: str) -> str:
    """
    Tao errors repeat the same message hundreds of times (once per element).
    Extract just the first unique error line for display.
    """
    lines = msg.splitlines()
    seen  = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line not in seen:
            seen.append(line)
        if len(seen) >= 3:   # show at most 3 unique lines
            break
    result = "  |  ".join(seen)
    if len(result) > 300:
        result = result[:297] + "…"
    return result
    """Return a short human-readable preview of a pytao response."""
    if raw is None:
        return "(None)"
    if isinstance(raw, (list, tuple)):
        n = len(raw)
        if n == 0:
            return "(empty list)"
        sample = str(raw[0])[:40]
        return f"list[{n}]  first={sample!r}"
    s = str(raw)
    if len(s) > 80:
        s = s[:77] + "…"
    return repr(s)


def _response_preview(raw) -> str:
    """Return a short human-readable preview of a pytao response."""
    if raw is None:
        return "(None)"
    if isinstance(raw, (list, tuple)):
        n = len(raw)
        if n == 0:
            return "(empty list)"
        sample = str(raw[0])[:40]
        return f"list[{n}]  first={sample!r}"
    s = str(raw)
    if len(s) > 80:
        s = s[:77] + "…"
    return repr(s)


def _parse_tao_values(raw) -> np.ndarray:
    """
    Robustly parse a pytao pipe command return value into a float64 array.
    Handles: list[str], list[float], single string with any delimiter, empty list.
    """
    if isinstance(raw, (int, float)):
        return np.array([float(raw)], dtype=np.float64)
    if isinstance(raw, str):
        tokens = raw.replace(";", "\n").replace(",", "\n").split()
        try:
            return np.array([float(t) for t in tokens if t.strip()], dtype=np.float64)
        except ValueError:
            return np.array([], dtype=np.float64)
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            try:
                out.append(float(item))
            except (ValueError, TypeError):
                pass
        return np.array(out, dtype=np.float64)
    try:
        return np.array(list(raw), dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)


def _parse_bunch_params(raw) -> dict:
    """
    Parse the output of 'pipe bunch_params ele|model' into a dict.
    The format is typically lines of:  key;value  or  key: value
    """
    meta = {}
    if not raw:
        return meta
    lines = raw if isinstance(raw, list) else str(raw).splitlines()
    for line in lines:
        line = str(line).strip()
        if not line:
            continue
        # Try semicolon separator first, then colon
        for sep in (";", ":"):
            if sep in line:
                parts = line.split(sep, 1)
                key   = parts[0].strip().lower().replace(" ", "_")
                val   = parts[1].strip()
                try:
                    meta[key] = float(val)
                except ValueError:
                    meta[key] = val
                break
    return meta


# ── Plot Panel ─────────────────────────────────────────────────────────────────

class PlotPanel(QWidget):
    def __init__(self, app, panel_index, parent=None):
        super().__init__(parent)
        self.app   = app
        self.index = panel_index

        self._color      = FILE_COLORS[0]
        self._ax_history = deque()
        self._ax_cols    = None
        self._hmap_cache = {}
        self._hmap_key   = None
        self._has_data   = False
        self._fixed_xlim = None   # (xmin, xmax) when mode == Fixed
        self._fixed_ylim = None   # (ymin, ymax) when mode == Fixed

        pair = DEFAULT_PAIRS[panel_index % len(DEFAULT_PAIRS)]

        self.setObjectName("plotpanel")
        self._set_border_color(FILE_COLORS[0])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # ── Header row 1: axes + mode ─────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet("background: #0a0a1a; border-radius: 4px;")
        hdr.setFixedHeight(30)
        hl  = QHBoxLayout(hdr)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        self.x_combo = QComboBox()
        self.x_combo.addItems([COL_LABELS[c] for c in DISPLAY_COLS])
        self.x_combo.setCurrentIndex(DISPLAY_COLS.index(pair[0]))
        self.x_combo.setFixedWidth(90)
        self.x_combo.currentIndexChanged.connect(self._on_axis_change)

        self.y_combo = QComboBox()
        self.y_combo.addItems([COL_LABELS[c] for c in DISPLAY_COLS])
        self.y_combo.setCurrentIndex(DISPLAY_COLS.index(pair[1]))
        self.y_combo.setFixedWidth(90)
        self.y_combo.currentIndexChanged.connect(self._on_axis_change)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Auto", "Roll", "Track", "Fixed"])
        self.mode_combo.setCurrentText("Roll")
        self.mode_combo.setFixedWidth(80)
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)

        # Lock button — captures current axes into Fixed mode
        self._lock_btn = QPushButton("🔒")
        self._lock_btn.setFixedWidth(28)
        self._lock_btn.setFixedHeight(22)
        self._lock_btn.setToolTip("Lock current axis limits (switches to Fixed mode)")
        self._lock_btn.setStyleSheet("QPushButton { font-size: 11px; padding: 0; }")
        self._lock_btn.clicked.connect(self._lock_axes)

        for w in [self._sep(), QLabel("X"), self.x_combo,
                  self._sep(), QLabel("Y"), self.y_combo,
                  self._sep(), self.mode_combo,
                  self._lock_btn]:
            hl.addWidget(w)
        hl.addStretch()
        layout.addWidget(hdr)

        # ── Header row 2: Fixed axis limit editors (hidden unless Fixed mode) ──
        self._fixed_bar = QWidget()
        self._fixed_bar.setStyleSheet("background: #0a0a1a;")
        self._fixed_bar.setFixedHeight(26)
        fl = QHBoxLayout(self._fixed_bar)
        fl.setContentsMargins(4, 1, 4, 1)
        fl.setSpacing(3)

        def _lim_edit(placeholder):
            e = QLineEdit()
            e.setPlaceholderText(placeholder)
            e.setFixedWidth(80)
            e.setStyleSheet(
                "QLineEdit { background: #161628; border: 1px solid #333366;"
                " border-radius: 3px; padding: 1px 4px; font-size: 10px; }")
            e.editingFinished.connect(self._apply_fixed_limits)
            return e

        fl.addWidget(QLabel("X:", styleSheet="font-size:10px; color:#8ab0d8;"))
        self._xmin_edit = _lim_edit("xmin")
        self._xmax_edit = _lim_edit("xmax")
        fl.addWidget(self._xmin_edit)
        fl.addWidget(QLabel("→", styleSheet="font-size:10px; color:#445566;"))
        fl.addWidget(self._xmax_edit)
        fl.addWidget(self._sep())
        fl.addWidget(QLabel("Y:", styleSheet="font-size:10px; color:#8ab0d8;"))
        self._ymin_edit = _lim_edit("ymin")
        self._ymax_edit = _lim_edit("ymax")
        fl.addWidget(self._ymin_edit)
        fl.addWidget(QLabel("→", styleSheet="font-size:10px; color:#445566;"))
        fl.addWidget(self._ymax_edit)
        fl.addStretch()

        self._fixed_bar.setVisible(False)
        layout.addWidget(self._fixed_bar)

        # ── Matplotlib canvas ─────────────────────────────────────────────
        self.fig    = plt.Figure(facecolor=BG, dpi=96)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self._draw_empty()

    def _set_border_color(self, color: str):
        self.setStyleSheet(
            "QWidget#plotpanel { border: 2px solid " + color + ";"
            " border-radius: 6px; background: #12122a; }"
        )

    def _sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setStyleSheet("color: #252535;")
        f.setFixedWidth(1)
        return f

    def x_col(self) -> str:
        return DISPLAY_COLS[self.x_combo.currentIndex()]

    def y_col(self) -> str:
        return DISPLAY_COLS[self.y_combo.currentIndex()]

    def _on_axis_change(self):
        self._ax_history.clear()
        self._hmap_cache.clear()
        self.app.render_all()

    def _on_mode_change(self):
        self._ax_history.clear()
        is_fixed = self.mode_combo.currentText() == "Fixed"
        self._fixed_bar.setVisible(is_fixed)
        self.app.render_all()

    def _lock_axes(self):
        """Capture current matplotlib axis limits and switch to Fixed mode."""
        if not self.fig.axes:
            return
        ax = self.fig.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self._fixed_xlim = xlim
        self._fixed_ylim = ylim
        # Populate the edit fields
        self._xmin_edit.setText(f"{xlim[0]:.4g}")
        self._xmax_edit.setText(f"{xlim[1]:.4g}")
        self._ymin_edit.setText(f"{ylim[0]:.4g}")
        self._ymax_edit.setText(f"{ylim[1]:.4g}")
        # Switch to Fixed mode
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText("Fixed")
        self.mode_combo.blockSignals(False)
        self._fixed_bar.setVisible(True)
        self.app.render_all()

    def _apply_fixed_limits(self):
        """Parse the limit edit fields and update _fixed_xlim/_fixed_ylim."""
        try:
            xmin = float(self._xmin_edit.text())
            xmax = float(self._xmax_edit.text())
            ymin = float(self._ymin_edit.text())
            ymax = float(self._ymax_edit.text())
            if xmin < xmax and ymin < ymax:
                self._fixed_xlim = (xmin, xmax)
                self._fixed_ylim = (ymin, ymax)
                self.app.render_all()
        except ValueError:
            pass  # incomplete input — ignore

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        ax.text(0.5, 0.5, "Connect to Tao to begin",
                transform=ax.transAxes, ha="center", va="center",
                color="#888888", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()

    def render(self, data: dict, settings: dict):
        """Render the panel with the current data snapshot."""
        if not data:
            if not self._has_data:
                self._draw_empty()
            return

        xn = self.x_col()
        yn = self.y_col()

        if xn not in data or yn not in data:
            if not self._has_data:
                self._draw_empty()
            return

        xd = data[xn].copy()
        yd = data[yn].copy()

        if len(xd) == 0:
            if not self._has_data:
                self._draw_empty()
            return

        xu = COL_UNITS.get(xn, "")
        yu = COL_UNITS.get(yn, "")
        xlbl = f"{COL_LABELS[xn]}  [{xu}]" if xu else COL_LABELS[xn]
        ylbl = f"{COL_LABELS[yn]}  [{yu}]" if yu else COL_LABELS[yn]

        # ── Axis limits ───────────────────────────────────────────────────
        ax_mode  = self.mode_combo.currentText()
        col_key  = (xn, yn)
        smooth_n = settings["smooth_n"]

        if col_key != self._ax_cols:
            self._ax_history.clear()
            self._ax_cols = col_key

        xlim = ylim = None

        if ax_mode == "Fixed":
            # Use stored limits; don't touch _ax_history
            xlim = self._fixed_xlim
            ylim = self._fixed_ylim

        elif ax_mode == "Roll":
            xc    = float(xd.mean())
            yc    = float(yd.mean())
            xhalf = float(xd.max() - xd.min()) / 2.0 or 1e-10
            yhalf = float(yd.max() - yd.min()) / 2.0 or 1e-10
            self._ax_history.append((xhalf * 1.05, yhalf * 1.05))
            while len(self._ax_history) > max(1, smooth_n):
                self._ax_history.popleft()
            arr  = list(self._ax_history)
            sxh  = sum(a[0] for a in arr) / len(arr)
            syh  = sum(a[1] for a in arr) / len(arr)
            xlim = (xc - sxh, xc + sxh)
            ylim = (yc - syh, yc + syh)

        elif ax_mode == "Track":
            nsig = settings["sigma"]
            xc   = float(xd.mean())
            yc   = float(yd.mean())
            xs   = float(xd.std()) or 1e-10
            ys   = float(yd.std()) or 1e-10
            self._ax_history.append((xs, ys))
            while len(self._ax_history) > max(1, smooth_n):
                self._ax_history.popleft()
            arr  = list(self._ax_history)
            sxs  = sum(a[0] for a in arr) / len(arr)
            sys_ = sum(a[1] for a in arr) / len(arr)
            xlim = (xc - nsig * sxs, xc + nsig * sxs)
            ylim = (yc - nsig * sys_, yc + nsig * sys_)

        # ── Figure layout ─────────────────────────────────────────────────
        show_hist    = settings["show_hist"]
        need_rebuild = (
            not self.fig.axes or
            (show_hist and len(self.fig.axes) < 3) or
            (not show_hist and len(self.fig.axes) != 1)
        )

        if need_rebuild:
            self.fig.clear()
            self.fig.patch.set_facecolor(BG)
            if show_hist:
                gs     = GridSpec(2, 2, figure=self.fig,
                                  width_ratios=[4, 1], height_ratios=[1, 4],
                                  hspace=0.03, wspace=0.03,
                                  left=0.13, right=0.97, top=0.97, bottom=0.13)
                ax_s   = self.fig.add_subplot(gs[1, 0])
                ax_hx  = self.fig.add_subplot(gs[0, 0], sharex=ax_s)
                ax_hy  = self.fig.add_subplot(gs[1, 1], sharey=ax_s)
                for ax in (ax_s, ax_hx, ax_hy):
                    _style_ax(ax)
                ax_main = ax_s
            else:
                ax_main = self.fig.add_subplot(111)
                _style_ax(ax_main)
                self.fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.13)
                ax_hx = ax_hy = None
        else:
            if show_hist:
                ax_main, ax_hx, ax_hy = (self.fig.axes[0],
                                          self.fig.axes[1],
                                          self.fig.axes[2])
                for ax in (ax_main, ax_hx, ax_hy):
                    ax.cla()
                    _style_ax(ax)
            else:
                ax_main = self.fig.axes[0]
                ax_main.cla()
                _style_ax(ax_main)
                ax_hx = ax_hy = None

        # ── Draw data ─────────────────────────────────────────────────────
        plot_mode    = settings["plot_mode"]
        cmap         = settings["cmap"]
        pts          = settings["pt_size"]
        alph         = settings["alpha"]
        hbins        = settings["hmap_bins"]
        log_clr      = settings["log_scale"]
        smooth_sigma = settings["smooth_sigma"]

        if plot_mode == "Scatter":
            ax_main.scatter(xd, yd, s=pts, c=self._color,
                            alpha=alph, linewidths=0, zorder=2)
        elif plot_mode == "Heatmap 2D":
            hmap_key = (xn, yn)
            if hmap_key != self._hmap_key:
                self._hmap_cache.clear()
                self._hmap_key = hmap_key
            cache_key = (id(data), hbins, round(smooth_sigma, 2))
            if cache_key in self._hmap_cache:
                h, xe, ye = self._hmap_cache[cache_key]
            else:
                h, xe, ye = np.histogram2d(xd, yd, bins=hbins)
                h = h.T.astype(float)
                if smooth_sigma > 0:
                    h = gaussian_filter(h, sigma=smooth_sigma)
                if len(self._hmap_cache) > 30:
                    self._hmap_cache.pop(next(iter(self._hmap_cache)))
                self._hmap_cache[cache_key] = (h, xe, ye)

            vmin = max(1e-3, h[h > 0].min()) if h.any() else 1e-3
            norm = (LogNorm(vmin=vmin, vmax=h.max())
                    if log_clr else Normalize(vmin=0, vmax=h.max()))
            ax_main.imshow(h, origin="lower", aspect="auto",
                           extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           cmap=cmap, norm=norm,
                           interpolation="gaussian", zorder=2)

        if show_hist and ax_hx is not None:
            hist_bins = settings["hist_bins"]
            ax_hx.hist(xd, bins=hist_bins, color=self._color,
                       alpha=0.75, linewidth=0)
            ax_hx.tick_params(labelbottom=False, colors=TEXT_C, labelsize=8)
            ax_hy.hist(yd, bins=hist_bins, color=self._color,
                       alpha=0.75, orientation="horizontal", linewidth=0)
            ax_hy.tick_params(labelleft=False, colors=TEXT_C, labelsize=8)

        if xlim:
            ax_main.set_xlim(xlim)
        if ylim:
            ax_main.set_ylim(ylim)

        if settings["show_overlay"]:
            _draw_overlay(ax_main, xd, yd, self._color, xn, yn)

        # ── Particle tracking overlay ─────────────────────────────────────
        tracked = self.app._tracked_ids
        if tracked:
            traj_cache = self.app._traj_cache
            history_addrs = list(traj_cache.keys())
            for t_idx, pid in enumerate(tracked):
                tcol = TRACK_COLORS[t_idx % len(TRACK_COLORS)]
                tx, ty, lost_x, lost_y = [], [], [], []
                for addr in history_addrs:
                    snap = traj_cache[addr]
                    if pid not in snap:
                        continue
                    pt = snap[pid]
                    if xn in pt and yn in pt:
                        if pt.get("alive", True):
                            tx.append(pt[xn])
                            ty.append(pt[yn])
                        else:
                            # Mark last known position before loss
                            lost_x.append(pt[xn])
                            lost_y.append(pt[yn])
                if len(tx) > 1:
                    ax_main.plot(tx, ty, color=tcol, linewidth=1.0,
                                 alpha=0.6, zorder=7)
                if tx:
                    ax_main.scatter([tx[-1]], [ty[-1]], s=50, color=tcol,
                                    zorder=8, linewidths=1.2,
                                    edgecolors="white")
                # Red × where particle was lost
                if lost_x:
                    ax_main.scatter(lost_x, lost_y, s=40, color="#ff3333",
                                    marker="x", linewidths=1.5, zorder=9)

        ax_main.set_xlabel(xlbl, color=TEXT_C, fontsize=10)
        ax_main.set_ylabel(ylbl, color=TEXT_C, fontsize=10)
        self._has_data = True
        self.canvas.draw_idle()

    def destroy_panel(self):
        plt.close(self.fig)
        self.deleteLater()


# ── Connection Dialog ──────────────────────────────────────────────────────────

class ConnectDialog(QDialog):
    """
    Dialog to launch or connect to a Tao session.
    Returns (init_file, extra_args) on accept.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Tao")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        # Mode selection
        mode_grp = QGroupBox("Connection mode")
        mode_layout = QVBoxLayout(mode_grp)
        self._mode_file  = QRadioButton("Launch new Tao session from init file")
        self._mode_obj   = QRadioButton(
            "Use already-running pytao Tao object (advanced — edit script)")
        self._mode_file.setChecked(True)
        mode_layout.addWidget(self._mode_file)
        mode_layout.addWidget(self._mode_obj)
        layout.addWidget(mode_grp)

        # Init file row
        self._file_grp = QGroupBox("Tao init file")
        fg = QVBoxLayout(self._file_grp)

        file_row = QWidget()
        fr = QHBoxLayout(file_row)
        fr.setContentsMargins(0, 0, 0, 0)
        self._init_edit = QLineEdit()
        self._init_edit.setPlaceholderText("path/to/tao.init")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        fr.addWidget(self._init_edit, 1)
        fr.addWidget(browse_btn)
        fg.addWidget(file_row)

        args_row = QWidget()
        ar = QHBoxLayout(args_row)
        ar.setContentsMargins(0, 0, 0, 0)
        ar.addWidget(QLabel("Extra args:"))
        self._args_edit = QLineEdit()
        self._args_edit.setPlaceholderText("-beam_init_position_file beam.dat")
        ar.addWidget(self._args_edit, 1)
        fg.addWidget(args_row)
        layout.addWidget(self._file_grp)

        # Particle count + charge
        npart_grp = QGroupBox("Particle count & charge")
        np_layout = QVBoxLayout(npart_grp)

        def _param_row(parent_layout, label, tao_name, widget):
            w  = QWidget()
            wl = QHBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            lb = QLabel(label)
            lb.setFixedWidth(130)
            lb.setToolTip(f"Tao: {tao_name}")
            widget.setToolTip(f"Tao field: {tao_name}")
            tao_lbl = QLabel(f"→ {tao_name}")
            tao_lbl.setStyleSheet("color: #445566; font-size: 9px;")
            tao_lbl.setFixedWidth(140)
            wl.addWidget(lb)
            wl.addWidget(widget, 1)
            wl.addWidget(tao_lbl)
            parent_layout.addWidget(w)
            return widget

        self._npart_spin = QSpinBox()
        self._npart_spin.setRange(1, 10_000_000)
        self._npart_spin.setValue(1000)
        self._npart_spin.setSingleStep(1000)
        _param_row(np_layout, "n_particle", "beam_init%n_particle", self._npart_spin)

        self._charge_spin = SciSpinBox()
        self._charge_spin.setRange(0.0, 1.0)
        self._charge_spin.setValue(1e-9)
        _param_row(np_layout, "bunch_charge [C]", "beam_init%bunch_charge", self._charge_spin)

        note = QLabel("⚠ bunch_charge must be non-zero for beam tracking.")
        note.setStyleSheet("color: #f0e04f; font-size: 10px;")
        np_layout.addWidget(note)
        layout.addWidget(npart_grp)

        # Transverse emittances
        emit_grp = QGroupBox("Transverse emittances")
        eg = QVBoxLayout(emit_grp)

        # Normalized / Geometric switch
        norm_row = QWidget()
        nrl = QHBoxLayout(norm_row)
        nrl.setContentsMargins(0, 0, 0, 0)
        self._emit_norm_rb = QRadioButton("Normalized  [m·rad]")
        self._emit_geom_rb = QRadioButton("Geometric  [m·rad]")
        self._emit_norm_rb.setChecked(True)
        nrl.addWidget(self._emit_norm_rb)
        nrl.addWidget(self._emit_geom_rb)
        nrl.addStretch()
        eg.addWidget(norm_row)

        def _emit_spin(default):
            sp = SciSpinBox()
            sp.setRange(0.0, 1.0)
            sp.setValue(default)
            return sp

        self._emit_x_spin = _emit_spin(1e-6)
        self._emit_y_spin = _emit_spin(1e-6)
        self._emit_x_lbl  = QLabel("emit_nx  [m·rad]")
        self._emit_x_lbl.setFixedWidth(130)
        self._emit_y_lbl  = QLabel("emit_ny  [m·rad]")
        self._emit_y_lbl.setFixedWidth(130)
        self._emit_x_tao  = QLabel("→ beam_init%a_norm_emit")
        self._emit_x_tao.setStyleSheet("color: #445566; font-size: 9px;")
        self._emit_x_tao.setFixedWidth(180)
        self._emit_y_tao  = QLabel("→ beam_init%b_norm_emit")
        self._emit_y_tao.setStyleSheet("color: #445566; font-size: 9px;")
        self._emit_y_tao.setFixedWidth(180)

        ex_row = QWidget(); exl = QHBoxLayout(ex_row); exl.setContentsMargins(0,0,0,0)
        exl.addWidget(self._emit_x_lbl); exl.addWidget(self._emit_x_spin, 1); exl.addWidget(self._emit_x_tao)
        eg.addWidget(ex_row)

        ey_row = QWidget(); eyl = QHBoxLayout(ey_row); eyl.setContentsMargins(0,0,0,0)
        eyl.addWidget(self._emit_y_lbl); eyl.addWidget(self._emit_y_spin, 1); eyl.addWidget(self._emit_y_tao)
        eg.addWidget(ey_row)

        def _update_emit_labels(normalized):
            if normalized:
                self._emit_x_lbl.setText("emit_nx  [m·rad]")
                self._emit_y_lbl.setText("emit_ny  [m·rad]")
                self._emit_x_tao.setText("→ beam_init%a_norm_emit")
                self._emit_y_tao.setText("→ beam_init%b_norm_emit")
            else:
                self._emit_x_lbl.setText("emit_x  [m·rad]")
                self._emit_y_lbl.setText("emit_y  [m·rad]")
                self._emit_x_tao.setText("→ beam_init%a_emit")
                self._emit_y_tao.setText("→ beam_init%b_emit")
        self._emit_norm_rb.toggled.connect(_update_emit_labels)
        layout.addWidget(emit_grp)

        # Longitudinal parameters
        long_grp = QGroupBox("Longitudinal parameters")
        lg = QVBoxLayout(long_grp)

        self._sigma_s_spin = SciSpinBox()
        self._sigma_s_spin.setRange(0.0, 100.0)
        self._sigma_s_spin.setValue(0.01)
        _param_row(lg, "sigma_s  [m]", "beam_init%sig_z", self._sigma_s_spin)

        self._sigma_dp_spin = SciSpinBox()
        self._sigma_dp_spin.setRange(0.0, 1.0)
        self._sigma_dp_spin.setValue(1e-3)
        _param_row(lg, "sigma_dp  [δ]", "beam_init%sig_pz", self._sigma_dp_spin)

        layout.addWidget(long_grp)

        # Distribution options
        dist_grp = QGroupBox("Distribution options")
        dg = QVBoxLayout(dist_grp)

        self._cutoff_spin = QDoubleSpinBox()
        self._cutoff_spin.setDecimals(1); self._cutoff_spin.setRange(0.0, 10.0)
        self._cutoff_spin.setValue(3.0); self._cutoff_spin.setSingleStep(0.5)
        _param_row(dg, "Gaussian cutoff [σ]", "beam_init%random_sigma_cutoff", self._cutoff_spin)

        self._renorm_sigma_cb  = QCheckBox("Enforce RMS values  (renorm_sigma)")
        self._renorm_sigma_cb.setChecked(False)
        self._renorm_sigma_cb.setToolTip("Tao: beam_init%renorm_sigma — rescales distribution to match exact sigma values")
        self._renorm_center_cb = QCheckBox("Re-center distribution  (renorm_center)")
        self._renorm_center_cb.setChecked(True)
        self._renorm_center_cb.setToolTip("Tao: beam_init%renorm_center — shifts centroid to zero")
        dg.addWidget(self._renorm_sigma_cb)
        dg.addWidget(self._renorm_center_cb)

        engine_row = QWidget(); enl = QHBoxLayout(engine_row); enl.setContentsMargins(0,0,0,0)
        enl.addWidget(QLabel("Random engine:"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(["quasi", "pseudo"])
        self._engine_combo.setToolTip("Tao: beam_init%random_engine\nquasi = Sobol (smoother), pseudo = random")
        enl.addWidget(self._engine_combo)
        enl.addWidget(QLabel("→ beam_init%random_engine", styleSheet="color:#445566;font-size:9px;"))
        enl.addStretch()
        dg.addWidget(engine_row)
        layout.addWidget(dist_grp)

        # track_start element
        ts_grp = QGroupBox("Beam track_start element")
        tsl = QVBoxLayout(ts_grp)
        self._track_start_edit = QLineEdit()
        self._track_start_edit.setPlaceholderText("e.g. LTR_MARK_BEG  (leave blank for BEGINNING)")
        tsl.addWidget(self._track_start_edit)
        note3 = QLabel("Must be a marker/element that exists in your lattice.")
        note3.setStyleSheet("color: #556680; font-size: 10px;")
        tsl.addWidget(note3)
        layout.addWidget(ts_grp)

        self._mode_obj.toggled.connect(
            lambda checked: self._file_grp.setEnabled(not checked))

        # Buttons
        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        ok_btn  = QPushButton("Connect")
        ok_btn.setStyleSheet("background: #1a3a1a;")
        cxl_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cxl_btn.clicked.connect(self.reject)
        br.addStretch()
        br.addWidget(cxl_btn)
        br.addWidget(ok_btn)
        layout.addWidget(btn_row)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select tao.init file", "", "Tao init (*.init);;All files (*.*)")
        if path:
            self._init_edit.setText(path)

    def init_file(self) -> str:
        return self._init_edit.text().strip()

    def extra_args(self) -> str:
        return self._args_edit.text().strip()

    def n_particles(self) -> int:
        return self._npart_spin.value()

    def bunch_charge(self) -> float:
        return self._charge_spin.value()

    def sigmas(self) -> dict:
        normalized = self._emit_norm_rb.isChecked()
        return {
            ("a_norm_emit" if normalized else "a_emit"): self._emit_x_spin.value(),
            ("b_norm_emit" if normalized else "b_emit"): self._emit_y_spin.value(),
            "sig_z":  self._sigma_s_spin.value(),
            "sig_pz": self._sigma_dp_spin.value(),
        }

    def dist_options(self) -> dict:
        """Extra beam_init distribution options."""
        return {
            "random_sigma_cutoff": self._cutoff_spin.value(),
            "renorm_sigma":        "T" if self._renorm_sigma_cb.isChecked() else "F",
            "renorm_center":       "T" if self._renorm_center_cb.isChecked() else "F",
            "random_engine":       self._engine_combo.currentText(),
            "random_gauss_converter": "limited" if self._cutoff_spin.value() > 0 else "exact",
        }

    def track_start(self) -> str:
        return self._track_start_edit.text().strip()

    def use_existing(self) -> bool:
        return self._mode_obj.isChecked()


# ── Element key → color for lattice strip ─────────────────────────────────────

ELE_COLORS = {
    "quadrupole":  "#e05050",   # red
    "sbend":       "#5080e0",   # blue
    "rbend":       "#5080e0",   # blue
    "sextupole":   "#50c050",   # green
    "octupole":    "#c0a030",   # yellow
    "kicker":      "#c050c0",   # purple
    "hkicker":     "#c050c0",
    "vkicker":     "#c050c0",
    "rfcavity":    "#40c0c0",   # cyan
    "lcavity":     "#40c0c0",
    "solenoid":    "#e09040",   # orange
    "wiggler":     "#e09040",
    "undulator":   "#e09040",
    "monitor":     "#606060",   # grey
    "instrument":  "#606060",
    "marker":      "#404040",
    "drift":       None,        # invisible
}


def _ele_color(key: str) -> str:
    return ELE_COLORS.get(key.lower(), "#505050")


# ── Optics Window ──────────────────────────────────────────────────────────────

class OpticsWindow(QWidget):
    """
    Standalone window showing Twiss functions, orbit, and dispersion
    vs s-position, with a lattice element strip and hover tooltip.
    """
    def __init__(self, tao, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lattice Optics")
        self.resize(1200, 820)
        self.setStyleSheet(f"background: {BG};")

        self._tao      = tao
        self._optics   = None   # fetched data dict
        self._s_line   = None   # vertical cursor line on each ax

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Toolbar row
        ctrl = QWidget()
        ctrl.setStyleSheet(f"background: #0f0f1e;")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(6, 4, 6, 4)
        cl.setSpacing(8)

        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedHeight(26)
        refresh_btn.clicked.connect(self._fetch_optics)
        cl.addWidget(refresh_btn)

        cl.addWidget(QLabel("Show:"))
        self._cb_beta   = QCheckBox("β");    self._cb_beta.setChecked(True)
        self._cb_alpha  = QCheckBox("α");    self._cb_alpha.setChecked(False)
        self._cb_eta    = QCheckBox("η");    self._cb_eta.setChecked(True)
        self._cb_orbit  = QCheckBox("orbit"); self._cb_orbit.setChecked(True)
        self._cb_phi    = QCheckBox("φ");    self._cb_phi.setChecked(False)
        for cb in [self._cb_beta, self._cb_alpha, self._cb_eta,
                   self._cb_orbit, self._cb_phi]:
            cb.setStyleSheet("color: #c8cde4;")
            cb.toggled.connect(self._rebuild_plots)
            cl.addWidget(cb)

        cl.addStretch()
        self._status = QLabel("Click Refresh to load optics")
        self._status.setStyleSheet("color: #8ab0d8; font-size: 10px;")
        cl.addWidget(self._status)
        layout.addWidget(ctrl)

        # Matplotlib figure
        self.fig = plt.Figure(facecolor=BG, dpi=96)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # Hover tooltip label (overlay on canvas)
        self._tooltip = QLabel("", self.canvas)
        self._tooltip.setStyleSheet(
            "QLabel { background: #1a1a3a; color: #c8cde4; "
            "border: 1px solid #4f9ef0; border-radius: 4px; "
            "padding: 3px 6px; font-size: 10px; font-family: monospace; }")
        self._tooltip.hide()

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("figure_leave_event",
                                lambda e: self._tooltip.hide())

        self._fetch_optics()

    # ── Data fetching ──────────────────────────────────────────────────────

    def _fetch_optics(self):
        self._status.setText("Fetching optics data…")
        QApplication.processEvents()
        try:
            who = ("ele.s,ele.l,ele.key,ele.name,"
                   "ele.a.beta,ele.b.beta,"
                   "ele.a.alpha,ele.b.alpha,"
                   "ele.a.eta,ele.b.eta,"
                   "ele.a.phi,ele.b.phi,"
                   "orbit.vec.1,orbit.vec.3")
            raw = self._tao.cmd(f"pipe lat_list 1@0>>*|model {who}")
            self._optics = self._parse_lat_list(raw, who)
            n = len(self._optics.get("ele.s", []))
            self._status.setText(f"Loaded optics at {n} elements")
            self._rebuild_plots()
        except Exception as e:
            import traceback
            self._status.setText(f"Error: {e}")
            traceback.print_exc()

    def _parse_lat_list(self, raw, who: str) -> dict:
        """
        Parse pipe lat_list output.
        pytao returns 3 list items per element:
          item 0: "ele.s;ele.l"
          item 1: "ele.key"   (no semicolon — string field)
          item 2: "ele.name;ele.a.beta;ele.b.beta;...;orbit.vec.3"
        """
        if not isinstance(raw, list) or len(raw) < 3:
            return {}

        result = {
            "ele.s": [], "ele.l": [], "ele.key": [], "ele.name": [],
            "ele.a.beta": [], "ele.b.beta": [],
            "ele.a.alpha": [], "ele.b.alpha": [],
            "ele.a.eta": [], "ele.b.eta": [],
            "ele.a.phi": [], "ele.b.phi": [],
            "orbit.vec.1": [], "orbit.vec.3": [],
        }

        def floats(s):
            return [float(x.strip()) for x in s.split(";") if x.strip()]

        i = 0
        while i + 2 < len(raw):
            try:
                sl   = floats(str(raw[i]))        # ele.s ; ele.l
                key  = str(raw[i+1]).strip()       # ele.key
                rest = [x.strip() for x in str(raw[i+2]).split(";")]

                if len(sl) < 2 or len(rest) < 11:
                    i += 3
                    continue

                result["ele.s"].append(sl[0])
                result["ele.l"].append(sl[1])
                result["ele.key"].append(key)
                result["ele.name"].append(rest[0])
                result["ele.a.beta"].append(float(rest[1]))
                result["ele.b.beta"].append(float(rest[2]))
                result["ele.a.alpha"].append(float(rest[3]))
                result["ele.b.alpha"].append(float(rest[4]))
                result["ele.a.eta"].append(float(rest[5]))
                result["ele.b.eta"].append(float(rest[6]))
                result["ele.a.phi"].append(float(rest[7]))
                result["ele.b.phi"].append(float(rest[8]))
                result["orbit.vec.1"].append(float(rest[9]))
                result["orbit.vec.3"].append(float(rest[10]))
            except (ValueError, IndexError):
                pass
            i += 3

        return {k: (np.array(v) if k not in ("ele.key", "ele.name") else v)
                for k, v in result.items()}

    # ── Plot building ──────────────────────────────────────────────────────

    def _active_panels(self):
        panels = []
        if self._cb_beta.isChecked():   panels.append("beta")
        if self._cb_alpha.isChecked():  panels.append("alpha")
        if self._cb_eta.isChecked():    panels.append("eta")
        if self._cb_orbit.isChecked():  panels.append("orbit")
        if self._cb_phi.isChecked():    panels.append("phi")
        return panels

    def _rebuild_plots(self):
        if self._optics is None:
            return
        d = self._optics
        s = d.get("ele.s", np.array([]))
        if len(s) == 0:
            return

        panels = self._active_panels()
        if not panels:
            self.fig.clear()
            self.canvas.draw_idle()
            return

        n_panels = len(panels)
        self.fig.clear()

        # Each panel gets a main axes + a thin lattice strip above it
        # Use gridspec with height_ratios: strip=1, plot=8 per panel
        ratios = []
        for _ in panels:
            ratios += [1, 8]

        gs = self.fig.add_gridspec(
            n_panels * 2, 1,
            height_ratios=ratios,
            hspace=0.08,
            left=0.08, right=0.97, top=0.97, bottom=0.06
        )

        self._axes     = []
        self._s_lines  = []
        ref_ax = None   # first plot axis — others share its x

        for i, panel in enumerate(panels):
            ax_strip = self.fig.add_subplot(
                gs[i * 2], sharex=ref_ax)
            ax_plot  = self.fig.add_subplot(
                gs[i * 2 + 1], sharex=ref_ax)
            if ref_ax is None:
                ref_ax = ax_plot

            self._draw_lattice_strip(ax_strip, d)
            self._draw_panel(ax_plot, panel, d, s)

            # Only show x label on bottom panel
            if i < n_panels - 1:
                ax_plot.set_xlabel("")
                ax_plot.tick_params(labelbottom=False)
                ax_strip.tick_params(labelbottom=False)
            else:
                ax_plot.set_xlabel("s  [m]", color=TEXT_C, fontsize=10)
                ax_strip.tick_params(labelbottom=False)

            # Vertical cursor line
            line = ax_plot.axvline(x=s[0], color="#ffffff", linewidth=0.8,
                                   alpha=0.4, zorder=10)
            self._s_lines.append((ax_strip, ax_plot, line))
            self._axes.append((ax_strip, ax_plot))

        self.canvas.draw_idle()

    def _draw_lattice_strip(self, ax, d):
        """Draw colored element bars in a thin strip axes."""
        ax.set_facecolor("#0a0a18")
        ax.set_yticks([])
        ax.tick_params(labelbottom=False, bottom=False,
                       left=False, labelleft=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylim(0, 1)

        s_arr   = d.get("ele.s",   [])
        l_arr   = d.get("ele.l",   [])
        keys    = d.get("ele.key", [])

        for i, (s_end, l, key) in enumerate(zip(s_arr, l_arr, keys)):
            color = _ele_color(key)
            if color is None or l < 1e-6:
                continue
            s_start = s_end - l
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(
                (s_start, 0.05), l, 0.9,
                facecolor=color, edgecolor="none", zorder=2
            ))

    def _draw_panel(self, ax, panel: str, d: dict, s: np.ndarray):
        _style_ax(ax)
        ax.tick_params(colors=TEXT_C, labelsize=9)

        colors_ab = {"a": "#4f9ef0", "b": "#f0904f"}  # blue=a, orange=b

        if panel == "beta":
            for plane, label in [("a", "βₐ [m]"), ("b", "β_b [m]")]:
                key = f"ele.{plane}.beta"
                if key in d:
                    ax.plot(s, d[key], color=colors_ab[plane],
                            linewidth=1.2, label=label)
            ax.set_ylabel("β [m]", color=TEXT_C, fontsize=9)
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=TEXT_C,
                      framealpha=0.8, loc="upper right")

        elif panel == "alpha":
            for plane, label in [("a", "αₐ"), ("b", "α_b")]:
                key = f"ele.{plane}.alpha"
                if key in d:
                    ax.plot(s, d[key], color=colors_ab[plane],
                            linewidth=1.2, label=label)
            ax.set_ylabel("α", color=TEXT_C, fontsize=9)
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=TEXT_C,
                      framealpha=0.8, loc="upper right")
            ax.axhline(0, color=SPINE_C, linewidth=0.5)

        elif panel == "eta":
            for plane, label in [("a", "ηₐ [m]"), ("b", "η_b [m]")]:
                key = f"ele.{plane}.eta"
                if key in d:
                    ax.plot(s, d[key], color=colors_ab[plane],
                            linewidth=1.2, label=label)
            ax.set_ylabel("η [m]", color=TEXT_C, fontsize=9)
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=TEXT_C,
                      framealpha=0.8, loc="upper right")
            ax.axhline(0, color=SPINE_C, linewidth=0.5)

        elif panel == "orbit":
            ox = d.get("orbit.vec.1")
            oy = d.get("orbit.vec.3")
            if ox is not None:
                ax.plot(s, ox * 1e3, color=colors_ab["a"],
                        linewidth=1.2, label="x [mm]")
            if oy is not None:
                ax.plot(s, oy * 1e3, color=colors_ab["b"],
                        linewidth=1.2, label="y [mm]")
            ax.set_ylabel("orbit [mm]", color=TEXT_C, fontsize=9)
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=TEXT_C,
                      framealpha=0.8, loc="upper right")
            ax.axhline(0, color=SPINE_C, linewidth=0.5)

        elif panel == "phi":
            for plane, label in [("a", "φₐ/2π"), ("b", "φ_b/2π")]:
                key = f"ele.{plane}.phi"
                if key in d:
                    ax.plot(s, d[key], color=colors_ab[plane],
                            linewidth=1.2, label=label)
            ax.set_ylabel("φ/2π", color=TEXT_C, fontsize=9)
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=TEXT_C,
                      framealpha=0.8, loc="upper right")

    # ── Hover tooltip ──────────────────────────────────────────────────────

    def _on_hover(self, event):
        if event.inaxes is None or self._optics is None:
            self._tooltip.hide()
            return

        s_arr  = self._optics.get("ele.s",   np.array([]))
        names  = self._optics.get("ele.name", [])
        keys   = self._optics.get("ele.key",  [])
        l_arr  = self._optics.get("ele.l",   np.array([]))
        sx     = event.xdata

        if sx is None or len(s_arr) == 0:
            self._tooltip.hide()
            return

        # Find nearest element by s position
        idx = int(np.argmin(np.abs(s_arr - sx)))
        name  = names[idx] if idx < len(names) else "?"
        key   = keys[idx]  if idx < len(keys)  else "?"
        s_val = s_arr[idx]
        l_val = l_arr[idx] if idx < len(l_arr) else 0.0

        txt = (f"{name}  [{idx}]\n"
               f"{key}\n"
               f"s = {s_val:.4f} m   L = {l_val:.4f} m")
        self._tooltip.setText(txt)
        self._tooltip.adjustSize()

        # Move cursor lines
        for ax_strip, ax_plot, line in self._s_lines:
            line.set_xdata([s_val, s_val])
        self.canvas.draw_idle()

        # Position tooltip near cursor but keep inside window
        px = self.canvas.mapFromGlobal(
            self.canvas.cursor().pos()).x() + 14
        py = self.canvas.mapFromGlobal(
            self.canvas.cursor().pos()).y() - 10
        tw = self._tooltip.width()
        th = self._tooltip.height()
        cw = self.canvas.width()
        ch = self.canvas.height()
        px = min(px, cw - tw - 4)
        py = max(py, 4)
        py = min(py, ch - th - 4)
        self._tooltip.move(px, py)
        self._tooltip.show()

    def closeEvent(self, event):
        plt.close(self.fig)
        event.accept()


# ── Main Window ────────────────────────────────────────────────────────────────

class TaoViewer(QMainWindow):
    # Emitted from worker thread — must be a Qt Signal to cross thread boundary
    _fetch_done = Signal(dict, dict, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tao Bunch Distribution Viewer")
        self.resize(1400, 900)

        # Tao state
        self._tao            = None      # pytao Tao instance
        self._connected      = False
        self._current_ele    = ""
        self._element_list   = []
        self._data           = {}
        self._meta           = {}
        self._history        = []
        self._history_idx    = -1
        self._temp_files     = []        # temp files to clean up on disconnect

        # Live-query state
        self._live_timer     = QTimer(self)
        self._live_timer.timeout.connect(self._live_tick)

        # Playback (step through elements like a movie)
        self._play_timer     = QTimer(self)
        self._play_timer.timeout.connect(self._advance_element)
        self._playing        = False
        self._fetching       = False
        self._worker_thread  = None
        self._worker_obj     = None

        # Particle tracking
        self._tracked_ids    = []   # list of int particle indices to track
        self._traj_cache     = {}   # addr -> {pid: {coord: float}} history

        # UI state
        self._panels         = []
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.render_all)

        self._fetch_done.connect(self._on_fetch_done)

        self._build_ui()
        self._add_panel()

    def _debounce(self):
        self._debounce_timer.start(120)

    def _log(self, message: str, level: str = "info"):
        """Append a timestamped line to the query log."""
        ts = QDateTime.currentDateTime().toString("hh:mm:ss")
        colors = {
            "info":    "#c8cde4",
            "ok":      "#4ff0a0",
            "warn":    "#f0e04f",
            "error":   "#f04f4f",
            "query":   "#4f9ef0",
            "dim":     "#556680",
        }
        color = colors.get(level, "#c8cde4")
        prefix = {"ok": "✓", "warn": "⚠", "error": "✗",
                  "query": "→", "dim": " "}.get(level, " ")
        html = (f'<span style="color:#445566;">[{ts}]</span> '
                f'<span style="color:{color};">{prefix} {message}</span>')
        self._log_widget.append(html)
        # Auto-scroll to bottom
        sb = self._log_widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_log(self):
        self._log_widget.clear()

    def _copy_log(self):
        QApplication.clipboard().setText(self._log_widget.toPlainText())

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        def tbtn(text, slot, color=None, checkable=False, width=None):
            b = QPushButton(text)
            b.setFixedHeight(28)
            if width:
                b.setFixedWidth(width)
            if color:
                b.setStyleSheet(
                    f"QPushButton {{ background: {color}; }} "
                    f"QPushButton:hover {{ background: {color}dd; }}"
                )
            if checkable:
                b.setCheckable(True)
            b.clicked.connect(slot)
            tb.addWidget(b)
            return b

        def tbsep():
            f = QFrame()
            f.setFrameShape(QFrame.VLine)
            f.setStyleSheet("color: #2a2a3a;")
            f.setFixedHeight(28)
            tb.addWidget(f)

        self.connect_btn = tbtn("Connect Tao", self._connect_tao,
                                "#1a3a2a", width=110)
        self.disconnect_btn = tbtn("Disconnect", self._disconnect_tao,
                                   "#3a1a1a", width=100)
        self.disconnect_btn.setEnabled(False)
        tbsep()
        tbtn("+ Panel", self._add_panel,    width=74)
        tbtn("− Panel", self._remove_panel, "#3a1a1a", width=74)
        tbsep()
        self.fetch_btn = tbtn("⟳  Query", self._fetch_current,
                              "#1a2a3a", width=90)
        self.fetch_btn.setEnabled(False)
        self.live_btn  = tbtn("Live", self._toggle_live,
                              "#1a2a1a", checkable=True, width=60)
        self.live_btn.setEnabled(False)
        tbsep()
        self.beam_btn = tbtn("Beam Setup…", self._open_beam_dialog,
                             "#2a1a3a", width=100)
        self.beam_btn.setEnabled(False)
        self.rerun_btn = tbtn("⟳ Re-track", self._rerun_tracking,
                              "#1a2a2a", width=90)
        self.rerun_btn.setEnabled(False)
        tbsep()
        self.optics_btn = tbtn("Optics", self._open_optics,
                               "#1a2a3a", width=70)
        self.optics_btn.setEnabled(False)
        tbsep()
        tbtn("Export", self._export, "#1a3a1a", width=70)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._status_lbl = QLabel("Not connected")
        self._meta_lbl   = QLabel("")
        self._progress   = QProgressBar()
        self._progress.setFixedWidth(120)
        self._progress.setFixedHeight(14)
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        self.status_bar.addWidget(self._status_lbl)
        self.status_bar.addPermanentWidget(self._progress)
        self.status_bar.addPermanentWidget(self._meta_lbl)
        _sig = QLabel("Author: Randika Gamage  |  Support: Absolutely not. Figure it out.")
        _sig.setStyleSheet("color: #c8cde4; font-size: 10px; padding-right: 6px;")
        self.status_bar.addPermanentWidget(_sig)

        # Central layout: sidebar | main area
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(232)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sidebar_inner = QWidget()
        self._sidebar_layout = QVBoxLayout(sidebar_inner)
        self._sidebar_layout.setContentsMargins(0, 4, 0, 4)
        self._sidebar_layout.setSpacing(0)
        sidebar_scroll.setWidget(sidebar_inner)

        self._build_sidebar()
        self._sidebar_layout.addStretch()
        main_layout.addWidget(sidebar_scroll)

        # Right: element selector + plot grid
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Element selector bar
        ele_bar = QWidget()
        ele_bar.setStyleSheet("background: #0f0f1e; border-radius: 4px;")
        el  = QHBoxLayout(ele_bar)
        el.setContentsMargins(6, 4, 6, 4)
        el.setSpacing(6)

        # Play/Pause button
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(32)
        self._play_btn.setFixedHeight(26)
        self._play_btn.setCheckable(True)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)
        el.addWidget(self._play_btn)

        # Element slider
        self._ele_slider = QSlider(Qt.Horizontal)
        self._ele_slider.setMinimum(0)
        self._ele_slider.setMaximum(1)
        self._ele_slider.setFixedWidth(180)
        self._ele_slider.setEnabled(False)
        self._ele_slider.valueChanged.connect(self._on_ele_slider)
        el.addWidget(self._ele_slider)

        # Speed spinner
        self._speed_spin = QSpinBox()
        self._speed_spin.setRange(1, 30)
        self._speed_spin.setValue(5)
        self._speed_spin.setSuffix(" fps")
        self._speed_spin.setFixedWidth(70)
        self._speed_spin.setToolTip("Playback speed (elements per second)")
        el.addWidget(self._speed_spin)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #252535;"); sep.setFixedWidth(1)
        el.addWidget(sep)

        el.addWidget(QLabel("Element:"))

        self._ele_combo = QComboBox()
        self._ele_combo.setMinimumWidth(200)
        self._ele_combo.setEditable(True)
        self._ele_combo.setInsertPolicy(QComboBox.NoInsert)
        self._ele_combo.currentTextChanged.connect(self._on_ele_changed)
        el.addWidget(self._ele_combo, 1)

        self._ele_search = QLineEdit()
        self._ele_search.setPlaceholderText("Filter…")
        self._ele_search.setFixedWidth(120)
        self._ele_search.textChanged.connect(self._filter_elements)
        el.addWidget(self._ele_search)

        # Element step buttons
        prev_btn = QPushButton("◀")
        prev_btn.setFixedWidth(30)
        prev_btn.setFixedHeight(26)
        prev_btn.clicked.connect(self._step_prev)
        next_btn = QPushButton("▶")
        next_btn.setFixedWidth(30)
        next_btn.setFixedHeight(26)
        next_btn.clicked.connect(self._step_next)
        el.addWidget(prev_btn)
        el.addWidget(next_btn)

        # History navigation
        self._hist_prev = QPushButton("↩")
        self._hist_prev.setFixedWidth(26)
        self._hist_prev.setFixedHeight(26)
        self._hist_prev.setToolTip("History back")
        self._hist_prev.clicked.connect(self._history_prev)
        self._hist_next = QPushButton("↪")
        self._hist_next.setFixedWidth(26)
        self._hist_next.setFixedHeight(26)
        self._hist_next.setToolTip("History forward")
        self._hist_next.clicked.connect(self._history_next)
        el.addWidget(self._hist_prev)
        el.addWidget(self._hist_next)
        el.addStretch()

        # s-position and element type labels
        self._s_lbl   = QLabel("")
        self._s_lbl.setStyleSheet("color: #4f9ef0; font-size: 10px;")
        self._ele_lbl = QLabel("")
        self._ele_lbl.setStyleSheet("color: #f0904f; font-size: 10px;")
        el.addWidget(self._s_lbl)
        el.addWidget(self._ele_lbl)

        right_layout.addWidget(ele_bar)

        # Vertical splitter: plot grid (top) | log panel (bottom)
        self._right_split = QSplitter(Qt.Vertical)

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(4)
        self._right_split.addWidget(self._grid_widget)

        # Log panel
        log_container = QWidget()
        log_container.setStyleSheet("background: #0a0a16;")
        lc = QVBoxLayout(log_container)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.setSpacing(0)

        log_header = QWidget()
        log_header.setStyleSheet("background: #0f0f1e;")
        log_header.setFixedHeight(22)
        lh = QHBoxLayout(log_header)
        lh.setContentsMargins(6, 0, 6, 0)
        lh.setSpacing(8)
        log_title = QLabel("Query Log")
        log_title.setStyleSheet("color: #8ab0d8; font-weight: bold; font-size: 10px;")
        copy_log_btn = QPushButton("Copy")
        copy_log_btn.setFixedHeight(18)
        copy_log_btn.setFixedWidth(46)
        copy_log_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 0 4px; background: #1a1a2e; "
            "border: 1px solid #333366; border-radius: 3px; }"
        )
        copy_log_btn.clicked.connect(self._copy_log)
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setFixedHeight(18)
        clear_log_btn.setFixedWidth(46)
        clear_log_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 0 4px; background: #1a1a2e; "
            "border: 1px solid #333366; border-radius: 3px; }"
        )
        clear_log_btn.clicked.connect(self._clear_log)
        lh.addWidget(log_title)
        lh.addStretch()
        lh.addWidget(copy_log_btn)
        lh.addWidget(clear_log_btn)
        lc.addWidget(log_header)

        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setFont(QFont("Monospace", 9))
        self._log_widget.setStyleSheet(
            "QTextEdit { background: #0a0a16; color: #c8cde4; "
            "border: none; padding: 4px; }"
        )
        lc.addWidget(self._log_widget)
        self._right_split.addWidget(log_container)

        # Start with log taking ~18% of vertical space
        self._right_split.setStretchFactor(0, 5)
        self._right_split.setStretchFactor(1, 1)

        right_layout.addWidget(self._right_split, 1)
        main_layout.addWidget(right, 1)

    def _build_sidebar(self):
        sl = self._sidebar_layout

        # ── DISPLAY ──────────────────────────────────────────────────────
        sec = SidebarSection("DISPLAY")
        self._hist_cb    = QCheckBox("Marginal histograms")
        self._overlay_cb = QCheckBox("Stats overlay")
        self._hist_cb.toggled.connect(self.render_all)
        self._overlay_cb.toggled.connect(self.render_all)
        sec.add(self._hist_cb)
        sec.add(self._overlay_cb)

        self._pt_slider     = make_slider(0.5, 12.0, 2.0, 1)
        self._alpha_slider  = make_slider(0.02, 1.0, 0.35, 2)
        self._bins_slider   = make_slider(10, 200, 60)
        self._smooth_slider = make_slider(1, 30, 1)
        self._sigma_slider  = make_slider(0.5, 10.0, 3.0, 1)

        for s in [self._pt_slider, self._alpha_slider, self._bins_slider,
                  self._smooth_slider, self._sigma_slider]:
            s.valueChanged.connect(self._debounce)

        sec.add_row("Point size",             self._pt_slider)
        sec.add_row("Alpha",                  self._alpha_slider)
        sec.add_row("Histogram bins",         self._bins_slider)
        sec.add_row("Axis smoothing (frames)", self._smooth_slider)
        sec.add_row("Track window (±σ)",      self._sigma_slider)
        sl.addWidget(sec)

        # ── PLOT MODE ────────────────────────────────────────────────────
        sec2 = SidebarSection("PLOT MODE")
        self._mode_scatter = QRadioButton("Scatter")
        self._mode_heatmap = QRadioButton("Heatmap 2D")
        self._mode_scatter.setChecked(True)
        self._mode_scatter.toggled.connect(self.render_all)
        sec2.add(self._mode_scatter)
        sec2.add(self._mode_heatmap)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["turbo", "plasma", "inferno", "gist_rainbow",
                                   "jet", "RdYlBu", "Spectral", "gnuplot2",
                                   "CMRmap", "afmhot"])
        self._cmap_combo.currentTextChanged.connect(self.render_all)
        sec2.add_row("Colormap", self._cmap_combo)

        self._hbins_slider   = make_slider(50, 500, 300)
        self._hsmooth_slider = make_slider(0.0, 8.0, 2.0, 1)
        self._hbins_slider.valueChanged.connect(self._debounce)
        self._hsmooth_slider.valueChanged.connect(self._debounce)
        self._log_cb = QCheckBox("Log color scale")
        self._log_cb.setChecked(True)
        self._log_cb.toggled.connect(self.render_all)

        sec2.add_row("Heatmap bins",      self._hbins_slider)
        sec2.add_row("Smoothing (sigma)", self._hsmooth_slider)
        sec2.add(self._log_cb)
        sl.addWidget(sec2)

        # ── LIVE QUERY ───────────────────────────────────────────────────
        sec3 = SidebarSection("LIVE QUERY")
        self._live_interval_slider = make_slider(0.5, 10.0, 2.0, 1)
        self._live_interval_slider.valueChanged.connect(self._update_live_interval)
        sec3.add_row("Interval (s)", self._live_interval_slider)
        self._live_status_lbl = QLabel("Live query: off")
        self._live_status_lbl.setStyleSheet("color: #666688; font-size: 10px;")
        sec3.add(self._live_status_lbl)
        sl.addWidget(sec3)

        # ── PARTICLE TRACKING ────────────────────────────────────────────
        sec5 = SidebarSection("PARTICLE TRACKING")

        self._track_entry = QLineEdit()
        self._track_entry.setPlaceholderText("e.g. 0,42,100")
        track_btn   = QPushButton("Track")
        track_btn.setFixedWidth(60)
        track_btn.clicked.connect(self._set_tracking)
        clear_track_btn = QPushButton("Clear")
        clear_track_btn.setFixedWidth(50)
        clear_track_btn.setStyleSheet("background: #3a1a1a;")
        clear_track_btn.clicked.connect(self._clear_tracking)

        track_row = QWidget()
        trl = QHBoxLayout(track_row)
        trl.setContentsMargins(0, 0, 0, 0)
        trl.addWidget(self._track_entry, 1)
        trl.addWidget(track_btn)
        trl.addWidget(clear_track_btn)

        self._track_lbl = QLabel("No particles tracked")
        self._track_lbl.setStyleSheet("color: #666688; font-size: 10px;")
        self._track_lbl.setWordWrap(True)

        sec5.add(track_row)
        sec5.add(self._track_lbl)
        sl.addWidget(sec5)

        # ── COORDINATE INFO ──────────────────────────────────────────────
        sec4 = SidebarSection("COORDINATES")
        info = QLabel(
            "Bmad coords:\n"
            "  x, px — horizontal\n"
            "  y, py — vertical\n"
            "  z     — path-length dev.\n"
            "  pz    — Δp/p\n"
            "  t[z]  — time from z\n"
            "          (β≈1 approx)"
        )
        info.setStyleSheet("color: #666688; font-size: 10px; font-family: monospace;")
        info.setWordWrap(True)
        sec4.add(info)
        sl.addWidget(sec4)

    # ── Settings accessor ──────────────────────────────────────────────────

    def _get_settings(self) -> dict:
        return {
            "plot_mode":    "Scatter" if self._mode_scatter.isChecked() else "Heatmap 2D",
            "cmap":         self._cmap_combo.currentText(),
            "pt_size":      self._pt_slider.real_value(),
            "alpha":        self._alpha_slider.real_value(),
            "hist_bins":    self._bins_slider.real_value(),
            "smooth_n":     self._smooth_slider.real_value(),
            "sigma":        self._sigma_slider.real_value(),
            "show_hist":    self._hist_cb.isChecked(),
            "show_overlay": self._overlay_cb.isChecked(),
            "hmap_bins":    self._hbins_slider.real_value(),
            "smooth_sigma": self._hsmooth_slider.real_value(),
            "log_scale":    self._log_cb.isChecked(),
        }

    # ── Panel management ───────────────────────────────────────────────────

    def _add_panel(self):
        idx   = len(self._panels)
        color = FILE_COLORS[idx % len(FILE_COLORS)]
        panel = PlotPanel(self, idx)
        panel._color = color
        panel._set_border_color(color)
        self._panels.append(panel)
        self._reflow_grid()
        if self._data:
            self.render_all()

    def _remove_panel(self):
        if len(self._panels) <= 1:
            return
        panel = self._panels.pop()
        panel.destroy_panel()
        self._reflow_grid()

    def _reflow_grid(self):
        n    = len(self._panels)
        cols = 1 if n == 1 else (2 if n <= 4 else 3)

        for i in reversed(range(self._grid_layout.count())):
            self._grid_layout.itemAt(i).widget().setParent(None)

        for i, panel in enumerate(self._panels):
            r, c = divmod(i, cols)
            self._grid_layout.addWidget(panel, r, c)

        for c in range(cols):
            self._grid_layout.setColumnStretch(c, 1)
        rows = (n + cols - 1) // cols
        for r in range(rows):
            self._grid_layout.setRowStretch(r, 1)

    # ── Tao connection ─────────────────────────────────────────────────────

    def _connect_tao(self):
        dlg = ConnectDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return

        if dlg.use_existing():
            QMessageBox.information(
                self, "Advanced mode",
                "To use an existing Tao object, assign it to\n"
                "viewer._tao after creating the TaoViewer instance,\n"
                "then call viewer._post_connect()."
            )
            return

        init_file = dlg.init_file()
        if not init_file:
            QMessageBox.warning(self, "No init file", "Please specify a tao.init file.")
            return
        if not Path(init_file).exists():
            QMessageBox.critical(self, "File not found", f"Cannot find:\n{init_file}")
            return

        try:
            from pytao import Tao
        except ImportError:
            QMessageBox.critical(
                self, "pytao not found",
                "pytao is not installed.\n\n"
                "Install it with:\n    pip install pytao\n\n"
                "Tao itself must also be installed and on your PATH."
            )
            return

        self._status_lbl.setText("Connecting to Tao…")
        self._progress.setVisible(True)
        QApplication.processEvents()

        n           = dlg.n_particles()
        charge      = dlg.bunch_charge()
        sigmas      = dlg.sigmas()
        dist_opts   = dlg.dist_options()
        extra       = dlg.extra_args()
        track_start = dlg.track_start()

        # Build namelist lines via helper, then write to a copy of tao.init
        # in the same directory so relative paths resolve correctly.
        namelist_lines = self._write_beam_init_file(
            n, charge, sigmas, dist_opts, track_start)

        init_dir = os.path.dirname(os.path.abspath(init_file))
        import tempfile
        fd, temp_init = tempfile.mkstemp(suffix=".init", prefix="taobunchplot_",
                                         dir=init_dir)
        with os.fdopen(fd, "w") as f:
            with open(init_file) as orig:
                f.write(orig.read())
            f.write("\n")
            f.write("\n".join(namelist_lines) + "\n")

        self._log(f"Wrote modified init: {temp_init}", "dim")

        # Startup file sets track_type = beam after all namelists load
        fd2, startup_file = tempfile.mkstemp(suffix=".startup", prefix="taobunchplot_")
        with os.fdopen(fd2, "w") as f:
            f.write("set global track_type = beam\n")
        self._log(f"Wrote startup file: {startup_file}", "dim")

        # Track both for cleanup on disconnect/close
        self._temp_files.extend([temp_init, startup_file])

        try:
            cmd = (f"-init {temp_init}"
                   f" -startup_file {startup_file}"
                   f" -noplot -external_plotting")
            if extra:
                cmd += f" {extra}"
            self._log(f"Launching Tao: {cmd}", "query")
            self._tao = Tao(cmd)
            self._log("Tao launched", "ok")
        except Exception as exc:
            self._progress.setVisible(False)
            self._log(f"Connection failed: {exc}", "error")
            QMessageBox.critical(self, "Connection failed", str(exc))
            self._status_lbl.setText("Connection failed")
            return

        self._post_connect()

    def _post_connect(self):
        """Called after self._tao is set — loads element list and enables UI."""
        self._progress.setVisible(False)
        try:
            self._load_element_list()
            self._log(f"Loaded {len(self._element_list)} elements from lattice", "ok")
        except Exception as exc:
            self._log(f"Element list warning: {exc}", "warn")
            QMessageBox.warning(
                self, "Element list warning",
                f"Could not load element list:\n{exc}\n\n"
                "You can still type element names manually."
            )

        self._connected = True
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.fetch_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.beam_btn.setEnabled(True)
        self.rerun_btn.setEnabled(True)
        self._play_btn.setEnabled(True)
        self.optics_btn.setEnabled(True)
        self._status_lbl.setText(f"Connected  ●  {len(self._element_list)} elements")
        self._log(f"Connected — {len(self._element_list)} elements available", "ok")

        # Auto-fetch first real element (index 1) — BEGINNING (index 0) has no
        # particles at its exit end since beam is initialized after it.
        # Then the combo is set to the last element for navigation.
        if self._element_list:
            last_display = self._element_list[-1]
            # Use second element (index 1) if available to confirm beam exists
            first_real_display = (self._element_list[1]
                                  if len(self._element_list) > 1
                                  else self._element_list[0])
            first_real_addr = self._resolve_addr(first_real_display)

            self._ele_combo.blockSignals(True)
            self._ele_combo.setCurrentText(last_display)
            self._ele_combo.blockSignals(False)

            self._log(f"Auto-querying first element: {first_real_display}  ({first_real_addr})", "query")
            self._fetch_element(first_real_addr, label=first_real_display)

    def _cleanup_temp_files(self):
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        self._temp_files = []

    def _disconnect_tao(self):
        self._stop_live()
        self._tao       = None
        self._connected = False
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.fetch_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_btn.setChecked(False)
        self.beam_btn.setEnabled(False)
        self.rerun_btn.setEnabled(False)
        self._play_btn.setEnabled(False)
        self._play_btn.setChecked(False)
        self._ele_slider.setEnabled(False)
        self._playing = False
        self._play_timer.stop()
        self.optics_btn.setEnabled(False)
        self._element_list = []
        self._ele_combo.clear()
        self._data = {}
        self._meta = {}
        self._traj_cache = {}
        self._status_lbl.setText("Disconnected")
        self._meta_lbl.setText("")
        self._log("Disconnected from Tao", "warn")
        self._cleanup_temp_files()
        for panel in self._panels:
            panel._has_data = False
            panel._draw_empty()

    # ── Element list ───────────────────────────────────────────────────────

    def _load_element_list(self):
        # Fetch names and indices together so we can build unambiguous addresses
        raw_names = self._tao.cmd("pipe lat_list 1@0>>*|model ele.name")
        raw_index = self._tao.cmd("pipe lat_list 1@0>>*|model ele.ix_ele")

        if isinstance(raw_names, list):
            names = [str(r).strip() for r in raw_names if str(r).strip()]
        else:
            names = [s.strip() for s in str(raw_names).splitlines() if s.strip()]

        if isinstance(raw_index, list):
            indices = [str(r).strip() for r in raw_index if str(r).strip()]
        else:
            indices = [s.strip() for s in str(raw_index).splitlines() if s.strip()]

        # Build display list as "NAME  (idx)" and address map to "1@0>>idx"
        # If a name appears more than once we must use the index address
        from collections import Counter
        name_counts = Counter(names)

        self._element_names   = []   # display strings shown in combo
        self._element_addrs   = []   # actual tao address used in pipe commands
        self._addr_by_display = {}   # display string -> tao address

        for i, name in enumerate(names):
            idx = indices[i] if i < len(indices) else str(i)
            addr = f"1@0>>{idx}"
            if name_counts[name] > 1:
                display = f"{name}  [{idx}]"
            else:
                display = name
            self._element_names.append(display)
            self._element_addrs.append(addr)
            self._addr_by_display[display] = addr

        self._element_list = self._element_names
        self._ele_slider.setMinimum(0)
        self._ele_slider.setMaximum(max(0, len(self._element_names) - 1))
        self._ele_slider.setValue(0)
        self._ele_slider.setEnabled(True)

        self._log(f"First: {names[0] if names else '?'}  "
                  f"Last: {names[-1] if names else '?'}  "
                  f"({len(names)} total)", "dim")

        self._ele_combo.blockSignals(True)
        self._ele_combo.clear()
        self._ele_combo.addItems(self._element_names)
        self._ele_combo.blockSignals(False)

    def _resolve_addr(self, display: str) -> str:
        """Convert a display name (possibly 'NAME [idx]') to a tao address '1@0>>N'."""
        if hasattr(self, "_addr_by_display") and display in self._addr_by_display:
            return self._addr_by_display[display]
        # Fallback: use as-is (manual entry or pre-list state)
        return display

    def _filter_elements(self, text: str):
        if not self._element_list:
            return
        filt = [n for n in self._element_list
                if text.lower() in n.lower()] if text else self._element_list
        current = self._ele_combo.currentText()
        self._ele_combo.blockSignals(True)
        self._ele_combo.clear()
        self._ele_combo.addItems(filt)
        if current in filt:
            self._ele_combo.setCurrentText(current)
        self._ele_combo.blockSignals(False)

    def _on_ele_changed(self, text: str):
        if not self._connected or not text:
            return
        self._current_ele = text
        # Sync slider position
        if text in self._element_list:
            idx = self._element_list.index(text)
            self._ele_slider.blockSignals(True)
            self._ele_slider.setValue(idx)
            self._ele_slider.blockSignals(False)
        addr = self._resolve_addr(text)
        self._fetch_element(addr, label=text)

    def _step_prev(self):
        if not self._element_list:
            return
        try:
            idx = self._element_list.index(self._ele_combo.currentText())
            idx = max(0, idx - 1)
        except ValueError:
            idx = 0
        self._ele_slider.blockSignals(True)
        self._ele_slider.setValue(idx)
        self._ele_slider.blockSignals(False)
        display = self._element_list[idx]
        self._ele_combo.setCurrentText(display)
        self._fetch_element(self._resolve_addr(display), label=display)

    def _step_next(self):
        if not self._element_list:
            return
        try:
            idx = self._element_list.index(self._ele_combo.currentText())
            idx = min(len(self._element_list) - 1, idx + 1)
        except ValueError:
            idx = 0
        self._ele_slider.blockSignals(True)
        self._ele_slider.setValue(idx)
        self._ele_slider.blockSignals(False)
        display = self._element_list[idx]
        self._ele_combo.setCurrentText(display)
        self._fetch_element(self._resolve_addr(display), label=display)

    # ── Beam init helpers ──────────────────────────────────────────────────

    def _write_beam_init_file(self, n_particles: int, bunch_charge: float,
                              sigmas: dict, dist_opts: dict = None,
                              track_start: str = "") -> str:
        """Write tao_beam_init namelist appended to a copy of tao.init."""
        import tempfile
        lines = ["&tao_beam_init"]
        lines.append(f"  beam_saved_at = \"*\"")
        lines.append(f"  ix_universe = 1")
        if track_start:
            lines.append(f"  track_start = '{track_start}'")
        lines.append(f"  beam_init%n_particle = {n_particles}")
        if bunch_charge != 0.0:
            lines.append(f"  beam_init%bunch_charge = {bunch_charge:.6g}")
        if sigmas:
            field_map = {
                "a_norm_emit": "beam_init%a_norm_emit",
                "b_norm_emit": "beam_init%b_norm_emit",
                "a_emit":      "beam_init%a_emit",
                "b_emit":      "beam_init%b_emit",
                "sig_z":       "beam_init%sig_z",
                "sig_pz":      "beam_init%sig_pz",
            }
            for key, namelist_key in field_map.items():
                if key in sigmas and sigmas[key] != 0.0:
                    lines.append(f"  {namelist_key} = {sigmas[key]:.8g}")
        if dist_opts:
            lines.append(f"  beam_init%random_sigma_cutoff = {dist_opts.get('random_sigma_cutoff', 3.0):.1f}")
            lines.append(f"  beam_init%renorm_sigma  = {dist_opts.get('renorm_sigma', 'F')}")
            lines.append(f"  beam_init%renorm_center = {dist_opts.get('renorm_center', 'T')}")
            lines.append(f"  beam_init%random_engine = \"{dist_opts.get('random_engine', 'quasi')}\"")
            lines.append(f"  beam_init%random_gauss_converter = \"{dist_opts.get('random_gauss_converter', 'limited')}\"")
        lines.append("/")
        return lines

    def _apply_beam_init(self, n_particles: int, bunch_charge: float,
                         sigmas: dict = None):
        """Send beam_init parameters to Tao. Non-fatal — logs warnings on error."""
        cmds = [
            f"set beam_init n_particle = {n_particles}",
        ]
        if bunch_charge != 0.0:
            cmds.append(f"set beam_init bunch_charge = {bunch_charge:.6g}")
        else:
            self._log("  WARNING: bunch_charge = 0 — beam tracking will fail!", "warn")
            self._log("  Set bunch_charge (e.g. 1e-9 C) in the connect dialog or Beam Setup", "dim")

        if sigmas:
            # Field names from 'show beam' output
            field_map = {
                "a_norm_emit": "a_norm_emit",
                "b_norm_emit": "b_norm_emit",
                "a_emit":      "a_emit",
                "b_emit":      "b_emit",
                "sig_z":       "sig_z",
                "sig_pz":      "sig_pz",
            }
            for key, tao_key in field_map.items():
                if key in sigmas and sigmas[key] != 0.0:
                    cmds.append(f"set beam_init {tao_key} = {sigmas[key]:.8g}")

        # Save beam at all elements so pipe bunch1 works everywhere
        cmds.append("set beam saved_at = *")
        cmds.append("reinit beam")

        for tao_cmd in cmds:
            self._log(f"  {tao_cmd}", "query")
            try:
                self._tao.cmd(tao_cmd)
                self._log(f"    OK", "dim")
            except Exception as e:
                self._log(f"    WARNING: {_first_error_line(str(e))}", "warn")

    def _try_run_tracking(self, sigmas: dict = None):
        """
        Set track_type = beam so Tao tracks on every lattice recalculation.
        Note: 'set global track_type = beam' triggers tracking immediately —
        'run' is the Tao optimizer and must NOT be called here.
        Non-fatal — logs a warning if it fails.
        """
        # Guard: refuse if emittances are all zero (Tao will crash)
        if sigmas is not None:
            emit_vals = [v for k, v in sigmas.items() if "emit" in k or "sig" in k]
            if emit_vals and all(v == 0.0 for v in emit_vals):
                self._log("WARNING: all emittances/sigmas are zero — skipping tracking", "warn")
                self._log("  Set non-zero a_norm_emit/b_norm_emit/sig_z/sig_pz in Beam Setup", "dim")
                return

        self._log("set global track_type = beam", "query")
        try:
            self._tao.cmd("set global track_type = beam")
            self._log("Beam tracking enabled — bunch data available at saved elements", "ok")
        except Exception as e:
            self._log(f"WARNING: track_type = beam failed — {_first_error_line(str(e))}", "warn")
            self._log("  Hint: check bunch_charge and beam parameters in Beam Setup", "dim")

    # ── Re-run tracking ────────────────────────────────────────────────────

    def _rerun_tracking(self):
        """Re-run beam tracking so bunch data is refreshed at all elements."""
        if not self._connected:
            return
        self._log("Re-running beam tracking…", "query")
        self._progress.setVisible(True)
        QApplication.processEvents()
        # Pass None for sigmas — user is responsible for setting them via Beam Setup first
        self._try_run_tracking(sigmas=None)
        self._progress.setVisible(False)
        if self._connected:
            self._fetch_current()

    # ── Beam setup dialog ──────────────────────────────────────────────────

    def _open_beam_dialog(self):
        """
        Dialog to configure beam_init parameters and send them to Tao.
        Covers the common case where tao.init has no bunch distribution defined.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Beam Setup")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)

        # ── Particle count ────────────────────────────────────────────────
        grp1 = QGroupBox("Particle count & charge")
        g1   = QVBoxLayout(grp1)
        row1 = QWidget()
        r1l  = QHBoxLayout(row1)
        r1l.setContentsMargins(0, 0, 0, 0)
        r1l.addWidget(QLabel("n_particle:"))
        n_spin = QSpinBox()
        n_spin.setRange(1, 10_000_000)
        n_spin.setValue(1000)
        n_spin.setSingleStep(1000)
        r1l.addWidget(n_spin, 1)
        g1.addWidget(row1)

        row_c = QWidget()
        rcl = QHBoxLayout(row_c)
        rcl.setContentsMargins(0, 0, 0, 0)
        rcl.addWidget(QLabel("bunch_charge (C):"))
        charge_spin = QDoubleSpinBox()
        charge_spin.setDecimals(6)
        charge_spin.setRange(0.0, 1.0)
        charge_spin.setValue(1e-9)
        charge_spin.setSingleStep(1e-10)
        charge_spin.setSpecialValueText("0 (skip)")
        rcl.addWidget(charge_spin, 1)
        g1.addWidget(row_c)
        layout.addWidget(grp1)

        # ── Distribution type ─────────────────────────────────────────────
        grp2 = QGroupBox("Distribution type")
        g2   = QVBoxLayout(grp2)
        dist_combo = QComboBox()
        dist_combo.addItems(["gaussian", "uniform", "kv", "grid"])
        dist_combo.setCurrentText("gaussian")
        g2.addWidget(dist_combo)
        note2 = QLabel("Sets beam_init%distribution_type")
        note2.setStyleSheet("color: #556680; font-size: 10px;")
        g2.addWidget(note2)
        layout.addWidget(grp2)

        # ── Gaussian sigmas ───────────────────────────────────────────────
        grp3 = QGroupBox("Beam parameters  (used when distribution = gaussian)")
        g3   = QVBoxLayout(grp3)

        def sigma_row(label, default, decimals=8):
            w   = QWidget()
            wl  = QHBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setFixedWidth(120)
            sp  = QDoubleSpinBox()
            sp.setDecimals(decimals)
            sp.setRange(0.0, 1.0)
            sp.setValue(default)
            sp.setSingleStep(default * 0.1 or 1e-8)
            wl.addWidget(lbl)
            wl.addWidget(sp, 1)
            g3.addWidget(w)
            return sp

        sig_emit_a = sigma_row("a_norm_emit  [m·rad]", 1e-6)
        sig_emit_b = sigma_row("b_norm_emit  [m·rad]", 1e-6)
        sig_z      = sigma_row("sig_z        [m]",     1e-2)
        sig_pz     = sigma_row("sig_pz       []",      1e-3)

        note3 = QLabel("a/b_norm_emit = normalized transverse emittances.\n"
                        "sig_z/pz = longitudinal.")
        note3.setStyleSheet("color: #556680; font-size: 10px;")
        g3.addWidget(note3)
        layout.addWidget(grp3)

        # Toggle sigma group visibility
        def _toggle_sigma(text):
            grp3.setVisible(text == "gaussian")
        dist_combo.currentTextChanged.connect(_toggle_sigma)

        # ── Custom commands ───────────────────────────────────────────────
        grp4 = QGroupBox("Additional Tao commands  (one per line, sent after above)")
        g4   = QVBoxLayout(grp4)
        extra_edit = QTextEdit()
        extra_edit.setFixedHeight(70)
        extra_edit.setPlaceholderText(
            "e.g.\nset beam_init energy = 1e9\nset beam_init spin = T")
        extra_edit.setFont(QFont("Monospace", 9))
        extra_edit.setStyleSheet(
            "QTextEdit { background: #0a0a18; color: #c8cde4; border: 1px solid #333366; }")
        g4.addWidget(extra_edit)
        layout.addWidget(grp4)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 4, 0, 0)
        apply_btn  = QPushButton("Apply & Re-init")
        apply_btn.setStyleSheet("background: #1a3a1a;")
        query_btn  = QPushButton("Apply & Query")
        query_btn.setStyleSheet("background: #1a2a3a;")
        cancel_btn = QPushButton("Cancel")
        br.addStretch()
        br.addWidget(cancel_btn)
        br.addWidget(apply_btn)
        br.addWidget(query_btn)
        layout.addWidget(btn_row)

        def _apply(also_query: bool):
            cmds = []
            cmds.append(f"set beam_init n_particle = {n_spin.value()}")
            if charge_spin.value() != 0.0:
                cmds.append(f"set beam_init bunch_charge = {charge_spin.value():.6g}")
            cmds.append(f"set beam_init distribution_type = {dist_combo.currentText()}")
            if dist_combo.currentText() == "gaussian":
                cmds += [
                    f"set beam_init a_norm_emit = {sig_emit_a.value():.8g}",
                    f"set beam_init b_norm_emit = {sig_emit_b.value():.8g}",
                    f"set beam_init sig_z       = {sig_z.value():.8g}",
                    f"set beam_init sig_pz      = {sig_pz.value():.8g}",
                ]
            # Extra user commands
            for line in extra_edit.toPlainText().splitlines():
                line = line.strip()
                if line:
                    cmds.append(line)
            # Reinit and re-enable tracking at end
            cmds.append("reinit beam")
            cmds.append("set global track_type = beam")

            # Wrap all set beam_init commands in lattice_calc_on=F so Tao
            # doesn't try to inject beam mid-sequence and fail
            self._log("tao> set global lattice_calc_on = F", "query")
            try:
                self._tao.cmd("set global lattice_calc_on = F")
            except Exception as e:
                self._log(f"  WARNING: {_first_error_line(str(e))}", "warn")

            errors = []
            for cmd in cmds:
                self._log(f"tao> {cmd}", "query")
                try:
                    result = self._tao.cmd(cmd)
                    preview = _response_preview(result)
                    self._log(f"  {preview}", "dim")
                except Exception as e:
                    err_short = _first_error_line(str(e))
                    self._log(f"  WARNING: {err_short}", "warn")
                    if cmd not in ("set global track_type = beam", "reinit beam"):
                        errors.append(f"{cmd}: {err_short}")

            # Re-enable lattice calc — triggers one clean tracking pass
            self._log("tao> set global lattice_calc_on = T", "query")
            try:
                self._tao.cmd("set global lattice_calc_on = T")
                self._log("  OK", "dim")
            except Exception as e:
                self._log(f"  WARNING: {_first_error_line(str(e))}", "warn")

            if errors:
                QMessageBox.warning(dlg, "Some commands failed",
                                    "\n".join(errors))
            else:
                self._log("Beam init applied successfully", "ok")

            dlg.accept()
            if also_query:
                self._fetch_current()

        apply_btn.clicked.connect(lambda: _apply(False))
        query_btn.clicked.connect(lambda: _apply(True))
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()

    # ── Playback ───────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._play_btn.isChecked():
            self._playing = True
            fps = max(1, self._speed_spin.value())
            self._play_timer.start(int(1000 / fps))
            self._play_btn.setText("⏸")
            self._log("Playback started", "dim")
        else:
            self._playing = False
            self._play_timer.stop()
            self._play_btn.setText("▶")
            self._log("Playback stopped", "dim")

    def _advance_element(self):
        """Step to next element during playback, wrapping at end."""
        if self._fetching or not self._element_list:
            return
        try:
            idx = self._element_list.index(self._ele_combo.currentText())
        except ValueError:
            idx = 0
        idx = (idx + 1) % len(self._element_list)
        # Update slider without triggering fetch (slider triggers fetch)
        self._ele_slider.blockSignals(True)
        self._ele_slider.setValue(idx)
        self._ele_slider.blockSignals(False)
        display = self._element_list[idx]
        self._ele_combo.blockSignals(True)
        self._ele_combo.setCurrentText(display)
        self._ele_combo.blockSignals(False)
        self._fetch_element(self._resolve_addr(display), label=display)

    def _on_ele_slider(self, idx: int):
        """Slider moved — jump to that element index."""
        if not self._element_list or idx >= len(self._element_list):
            return
        display = self._element_list[idx]
        self._ele_combo.blockSignals(True)
        self._ele_combo.setCurrentText(display)
        self._ele_combo.blockSignals(False)
        self._fetch_element(self._resolve_addr(display), label=display)

    # ── Particle tracking ──────────────────────────────────────────────────

    def _set_tracking(self):
        text = self._track_entry.text().strip()
        if not text:
            self._clear_tracking()
            return
        ids = []
        for tok in text.replace(",", " ").split():
            try:
                ids.append(int(tok))
            except ValueError:
                pass
        if not ids:
            self._track_lbl.setText("No valid indices entered")
            return
        self._tracked_ids = ids[:8]  # max 8 like SDDS viewer
        self._traj_cache  = {}       # reset history when tracking changes
        self._track_lbl.setText(f"Tracking: {', '.join(str(i) for i in self._tracked_ids)}")
        self.render_all()

    def _clear_tracking(self):
        self._tracked_ids = []
        self._traj_cache  = {}
        self._track_lbl.setText("No particles tracked")
        self._track_entry.clear()
        self.render_all()

    # ── Data fetching ──────────────────────────────────────────────────────

    def _fetch_current(self):
        display = self._ele_combo.currentText().strip()
        if not display or not self._connected:
            return
        addr = self._resolve_addr(display)
        self._log(f"Element: {display}  →  addr: {addr}", "dim")
        self._fetch_element(addr, label=display)

    def _fetch_element(self, element: str, label: str = ""):
        if self._fetching:
            return
        self._fetching = True
        self._progress.setVisible(True)
        self._fetch_btn_state(False)
        display = label or element
        self._status_lbl.setText(f"Querying  {display}…")
        self._log(f"Querying: {display}  [{element}]", "query")

        worker  = FetchWorker(self._tao, element)
        thread  = QThread(self)
        worker.moveToThread(thread)
        # Wire progress signal — runs in worker thread, crosses to GUI via Qt queued connection
        worker.progress.connect(lambda msg: self._log(msg, "dim"),
                                Qt.QueuedConnection)
        worker.finished.connect(self._fetch_done.emit)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        worker.finished.connect(thread.quit)
        # Keep references alive until thread finishes
        self._worker_thread = thread
        self._worker_obj    = worker
        thread.start()

    def _fetch_btn_state(self, enabled: bool):
        self.fetch_btn.setEnabled(enabled and self._connected)

    def _on_fetch_done(self, data: dict, meta: dict, error: str):
        self._fetching = False
        self._progress.setVisible(False)
        self._fetch_btn_state(True)

        if error:
            self._status_lbl.setText(f"Fetch error: {error}")
            self._log(f"Fetch error: {error}", "error")
            return

        self._data = data
        self._meta = meta

        ele     = meta.get("element", self._ele_combo.currentText())
        n_alive = meta.get("n_alive", len(next(iter(data.values()), [])))
        n_lost  = meta.get("n_lost", 0)
        n_total = meta.get("n_total", n_alive + n_lost)

        # Log result summary
        coord_shapes = {c: len(v) for c, v in data.items() if c != "t_derived"}
        self._log(
            f"  {ele}: {n_alive:,} alive / {n_total:,} total"
            + (f"  ({n_lost:,} lost)" if n_lost else ""),
            "ok"
        )
        s_val = meta.get("s", meta.get("s_position", None))
        if isinstance(s_val, float):
            self._log(f"  s = {s_val:.4f} m", "dim")

        # Push to history
        self._history = self._history[:self._history_idx + 1]
        self._history.append({"element": ele, "data": data, "meta": meta})
        self._history_idx = len(self._history) - 1

        # Update s-position and element type labels
        s_val   = meta.get("s", meta.get("s_position", None))
        ele_key = meta.get("key", meta.get("ele_type", ""))
        self._s_lbl.setText(f"s = {s_val:.4f} m" if isinstance(s_val, float) else "")
        self._ele_lbl.setText(str(ele_key) if ele_key else "")

        self._status_lbl.setText(
            f"●  {ele}   |   {n_alive:,} alive"
            + (f"  ({n_lost:,} lost)" if n_lost else "")
        )

        # Store trajectory snapshot for tracked particles using original indices
        if self._tracked_ids and data:
            full  = meta.get("_full", {})
            state = meta.get("_state", None)
            snap  = {}
            for pid in self._tracked_ids:
                if full and pid < len(next(iter(full.values()), [])):
                    # Use original-index arrays so index is stable regardless of losses
                    alive_now = (state[pid] == 1) if (state is not None and pid < len(state)) else True
                    snap[pid] = {
                        "alive": alive_now,
                        **{c: float(full[c][pid])
                           for c in list(BMAD_COORDS) + ["t_derived"]
                           if c in full and not np.isnan(full[c][pid])}
                    }
                elif pid < n_alive:
                    # Fallback: use filtered arrays
                    snap[pid] = {
                        "alive": True,
                        **{c: float(data[c][pid])
                           for c in list(BMAD_COORDS) + ["t_derived"]
                           if c in data and pid < len(data[c])}
                    }
            if snap:
                self._traj_cache[ele] = snap

        self.render_all()

    # ── History navigation ─────────────────────────────────────────────────

    def _history_prev(self):
        if self._history_idx > 0:
            self._history_idx -= 1
            snap = self._history[self._history_idx]
            self._data = snap["data"]
            self._meta = snap["meta"]
            self._ele_combo.blockSignals(True)
            self._ele_combo.setCurrentText(snap["element"])
            self._ele_combo.blockSignals(False)
            self.render_all()

    def _history_next(self):
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            snap = self._history[self._history_idx]
            self._data = snap["data"]
            self._meta = snap["meta"]
            self._ele_combo.blockSignals(True)
            self._ele_combo.setCurrentText(snap["element"])
            self._ele_combo.blockSignals(False)
            self.render_all()

    # ── Live query ─────────────────────────────────────────────────────────

    def _toggle_live(self):
        if self.live_btn.isChecked():
            interval_s = self._live_interval_slider.real_value()
            self._live_timer.start(int(interval_s * 1000))
            self._live_status_lbl.setText(f"Live: every {interval_s:.1f} s")
            self._log(f"Live query started — interval {interval_s:.1f} s", "ok")
        else:
            self._stop_live()

    def _stop_live(self):
        self._live_timer.stop()
        self._live_status_lbl.setText("Live query: off")
        if self._connected:
            self._log("Live query stopped", "warn")

    def _live_tick(self):
        if not self._fetching and self._connected:
            self._fetch_current()

    def _update_live_interval(self):
        if self.live_btn.isChecked():
            interval_s = self._live_interval_slider.real_value()
            self._live_timer.setInterval(int(interval_s * 1000))
            self._live_status_lbl.setText(f"Live: every {interval_s:.1f} s")

    # ── Rendering ──────────────────────────────────────────────────────────

    def render_all(self):
        if not self._data:
            return
        # Don't re-render while a fetch is in progress — keeps previous frame visible
        if self._fetching:
            return
        settings = self._get_settings()
        for panel in self._panels:
            panel.render(self._data, settings)

    # ── Optics window ──────────────────────────────────────────────────────

    def _open_optics(self):
        if not self._connected:
            return
        self._optics_win = OpticsWindow(self._tao, parent=None)
        self._optics_win.show()

    # ── Export ─────────────────────────────────────────────────────────────

    def _export(self):
        if not self._data:
            QMessageBox.warning(self, "Export", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export panels", "", "PNG image (*.png);;PDF (*.pdf)")
        if not path:
            return

        n    = len(self._panels)
        cols = 1 if n == 1 else (2 if n <= 4 else 3)
        rows = (n + cols - 1) // cols
        fig, axes_arr = plt.subplots(rows, cols,
                                     figsize=(cols * 5, rows * 4),
                                     facecolor=BG)
        axes_flat = np.array(axes_arr).flatten() if n > 1 else [axes_arr]
        settings  = self._get_settings()

        for i, panel in enumerate(self._panels):
            ax  = axes_flat[i]
            ax.set_facecolor(AX_BG)
            xn  = panel.x_col()
            yn  = panel.y_col()
            if xn not in self._data or yn not in self._data:
                continue
            xd = self._data[xn]
            yd = self._data[yn]
            xu = COL_UNITS.get(xn, "")
            yu = COL_UNITS.get(yn, "")
            if settings["plot_mode"] == "Scatter":
                ax.scatter(xd, yd, s=settings["pt_size"],
                           c=panel._color, alpha=settings["alpha"],
                           linewidths=0, zorder=2)
            ax.set_xlabel(f"{COL_LABELS[xn]}  [{xu}]" if xu else COL_LABELS[xn],
                          color=TEXT_C, fontsize=10)
            ax.set_ylabel(f"{COL_LABELS[yn]}  [{yu}]" if yu else COL_LABELS[yn],
                          color=TEXT_C, fontsize=10)
            ax.tick_params(colors=TEXT_C, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor(SPINE_C)
            ax.set_facecolor(AX_BG)
            ele = self._meta.get("element", "")
            ax.set_title(ele, color=TEXT_C, fontsize=9, pad=3)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
        if not path.endswith(".pdf"):
            pdf_path = path.rsplit(".", 1)[0] + ".pdf"
            fig.savefig(pdf_path, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        QMessageBox.information(self, "Export", f"Saved to {path}")

    # ── Close ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_live()
        self._live_timer.stop()
        self._play_timer.stop()
        self._cleanup_temp_files()
        self._tao = None
        for panel in self._panels:
            plt.close(panel.fig)
        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Tao Bunch Distribution Viewer")
    apply_dark_palette(app)
    window = TaoViewer()
    window.show()
    sys.exit(app.exec())
