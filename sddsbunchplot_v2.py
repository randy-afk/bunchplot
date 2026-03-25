#!/usr/bin/env python3

"""
SDDS Bunch Distribution Viewer v2
===================================
PySide6 + Matplotlib GUI for viewing SDDS binary particle data files.

Install dependencies:
    pip install PySide6 matplotlib numpy scipy

Run:
    python SDDSbunchplot_v2.py
"""

import struct
import re
import threading
import time
import json
from collections import deque
from pathlib import Path

import numpy as np
import os
os.environ['QT_API'] = 'pyside6'
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
    QToolBar, QStatusBar, QGroupBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup,
    QStackedWidget,
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, QObject, QThread, QSize, QPoint,
)
from PySide6.QtGui import (
    QColor, QPalette, QFont, QAction, QIcon, QPixmap,
)

import sys

# ── Constants ─────────────────────────────────────────────────────────────────

COLUMNS = ["x", "xp", "y", "yp", "t", "p", "dt", "particleID"]
COL_UNITS = {
    "x": "m", "xp": "", "y": "m", "yp": "",
    "t": "s", "p": "m·βγ", "dt": "s", "particleID": "",
}

PARTICLE_DTYPE = np.dtype([
    ("x",          "<f8"),
    ("xp",         "<f8"),
    ("y",          "<f8"),
    ("yp",         "<f8"),
    ("t",          "<f8"),
    ("p",          "<f8"),
    ("dt",         "<f8"),
    ("particleID", "<u8"),
])
PARTICLE_SIZE = PARTICLE_DTYPE.itemsize  # 64 bytes

DEFAULT_PAIRS = [("x", "xp"), ("y", "yp"), ("t", "p"), ("x", "y")]

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

# Matplotlib dark theme colours
BG      = "#1a1a2e"
AX_BG   = "#12122a"
GRID_C  = "#2a2a50"
TEXT_C  = "#c8cde4"
SPINE_C = "#333366"

PARTICLE_MASSES = {
    "Electron": 0.51099895,
    "Proton":   938.27208816,
}

_RMS95 = 2.4477  # sqrt(-2*ln(0.05)) for 95% ellipse


# ── SDDS Parser ───────────────────────────────────────────────────────────────

def _parse_header(raw: bytes):
    data_pos = raw.find(b"&data")
    if data_pos == -1:
        raise ValueError("Could not find '&data' — not a valid SDDS file.")
    nl = raw.find(b"\n", data_pos)
    if nl == -1:
        raise ValueError("Malformed header: no newline after '&data'.")
    binary_start = nl + 1
    header_text  = raw[:binary_start].decode("latin-1", errors="replace")

    param_blocks = re.findall(r'&parameter(.*?)&end', header_text, re.DOTALL)
    param_defs, fixed_params = [], {}

    for block in param_blocks:
        name_m  = re.search(r'name\s*=\s*(\w+)',        block)
        type_m  = re.search(r'type\s*=\s*(\w+)',        block)
        fixed_m = re.search(r'fixed_value\s*=\s*(\S+)', block)
        if not name_m or not type_m:
            continue
        name, ptype = name_m.group(1), type_m.group(1)
        if fixed_m:
            fv = fixed_m.group(1).rstrip(',')
            try:
                fixed_params[name] = float(fv) if ptype == "double" else int(fv)
            except ValueError:
                fixed_params[name] = fv
        else:
            param_defs.append((name, ptype))

    return binary_start, param_defs, fixed_params


def read_sdds_file(filepath: str) -> list:
    with open(filepath, "rb") as f:
        header_raw = f.read(65536)
    binary_start, param_defs, fixed_params = _parse_header(header_raw)

    pages = []
    with open(filepath, "rb") as f:
        f.seek(binary_start)
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            n_rows = struct.unpack("<i", hdr)[0]
            if n_rows < 0 or n_rows > 10_000_000:
                break
            if n_rows == 0:
                continue

            params = dict(fixed_params)
            ok = True
            for pname, ptype in param_defs:
                try:
                    if ptype == "double":
                        raw = f.read(8)
                        if len(raw) < 8: raise EOFError
                        params[pname] = struct.unpack("<d", raw)[0]
                    elif ptype in ("long", "short"):
                        raw = f.read(4)
                        if len(raw) < 4: raise EOFError
                        params[pname] = struct.unpack("<i", raw)[0]
                    elif ptype == "string":
                        raw = f.read(4)
                        if len(raw) < 4: raise EOFError
                        slen = struct.unpack("<i", raw)[0]
                        if slen < 0 or slen > 1_000_000:
                            raise ValueError
                        params[pname] = f.read(slen).decode("latin-1", errors="replace")
                except (EOFError, struct.error, ValueError):
                    ok = False
                    break
            if not ok:
                break

            byte_count = n_rows * PARTICLE_SIZE
            chunk = f.read(byte_count)
            if len(chunk) < PARTICLE_SIZE:
                break
            if len(chunk) < byte_count:
                n_rows = len(chunk) // PARTICLE_SIZE
                chunk  = chunk[:n_rows * PARTICLE_SIZE]

            structured = np.frombuffer(chunk, dtype=PARTICLE_DTYPE).copy()
            data = np.column_stack([structured[col].astype(np.float64)
                                    for col in COLUMNS])
            del structured, chunk
            pages.append({"params": params, "data": data})

    return pages


# ── RF Bucket Physics ─────────────────────────────────────────────────────────

def compute_rf_separatrix_full(cavities, alphac, p_central, mass_mev,
                                f_rev_hz, n_points=600):
    if not cavities:
        return None, None

    bg        = float(p_central)
    gamma     = np.sqrt(1.0 + bg**2)
    E0_eV     = mass_mev * 1e6 * gamma
    eta       = float(alphac) - 1.0 / gamma**2
    omega_rev = 2.0 * np.pi * f_rev_hz

    h1, phi_s, V1 = cavities[0][1], cavities[0][2], cavities[0][0]
    phi_ufp = np.pi - phi_s

    def F_single(phi):
        return (-np.cos(phi) - np.cos(phi_s)
                + (np.pi - phi - phi_s) * np.sin(phi_s))

    factor = V1 / (np.pi * h1 * abs(eta) * E0_eV)

    if len(cavities) > 1:
        def F_func(phi):
            val = F_single(phi)
            for V, h, phi_sv in cavities[1:]:
                ratio = h / h1
                phi_i = ratio * phi + (phi_sv - ratio * phi_s)
                val += (V / V1) * (
                    -np.cos(phi_i) - np.cos(phi_sv)
                    + (np.pi - phi_i - phi_sv) * np.sin(phi_sv)
                )
            return val
    else:
        F_func = F_single

    eps     = 0.002
    phi_arr = np.linspace(phi_ufp + eps, 2.0 * np.pi - phi_ufp - eps, n_points)
    Fvals   = np.array([F_func(p) for p in phi_arr])
    delta2  = factor * Fvals
    mask    = delta2 >= 0

    if not mask.any():
        return None, None

    phi_valid   = phi_arr[mask]
    delta_valid = np.sqrt(np.maximum(delta2[mask], 0))
    dt_valid    = (phi_valid - phi_s) / (h1 * omega_rev)

    dt_sep    = np.concatenate([dt_valid,   dt_valid[::-1]])
    delta_sep = np.concatenate([delta_valid, -delta_valid[::-1]])

    return dt_sep, delta_sep


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _compute_twiss(xd, yd):
    if len(xd) < 3:
        return None
    xc, yc = xd - xd.mean(), yd - yd.mean()
    s11 = float(np.mean(xc**2))
    s12 = float(np.mean(xc * yc))
    s22 = float(np.mean(yc**2))
    det = s11 * s22 - s12**2
    if det <= 0:
        return None
    emit = float(np.sqrt(det))
    return {"emit": emit, "beta": s11/emit,
            "alpha": -s12/emit, "gamma": s22/emit}


def _style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.grid(True, color=GRID_C, linewidth=0.4, zorder=0)


def _draw_overlay(ax, xd, yd, color, xn="x", yn="y", p_central=None):
    xc = float(xd.mean()); yc = float(yd.mean())
    xs = float(xd.std()) or 1e-10
    ys = float(yd.std()) or 1e-10

    # Crosshairs
    ax.axvline(xc, color=color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=5)
    ax.axhline(yc, color=color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=5)

    # Covariance ellipses
    xdev = xd - xd.mean(); ydev = yd - yd.mean()
    cov  = np.array([[np.mean(xdev**2), np.mean(xdev*ydev)],
                     [np.mean(xdev*ydev), np.mean(ydev**2)]])
    vals, vecs = np.linalg.eigh(cov)
    vals  = np.maximum(vals, 0)
    theta = np.linspace(0, 2*np.pi, 200)
    unit  = np.array([np.cos(theta), np.sin(theta)])
    basis = vecs @ (np.sqrt(vals)[:, None] * unit)

    ax.plot(xc + basis[0], yc + basis[1],
            color=color, linewidth=1.2, alpha=0.9, zorder=6)
    e95 = basis * _RMS95
    ax.plot(xc + e95[0], yc + e95[1],
            color=color, linewidth=1.0, alpha=0.45, linestyle="--", zorder=6)

    # Twiss box (conjugate pairs only)
    CONJUGATE = {("x","xp"),("xp","x"),("y","yp"),("yp","y")}
    if (xn, yn) in CONJUGATE:
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

    # σ box top right
    def _fmt(col, sigma):
        if col == "p" and p_central and float(p_central) != 0:
            return f"σ_δ = {sigma/abs(float(p_central)):.4g}"
        return f"σ_{col} = {sigma:.4g}"

    sigma_txt = _fmt(xn, xs) + "\n" + _fmt(yn, ys) + "\n[RMS]"
    ax.text(0.98, 0.98, sigma_txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=10.5, family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a18",
                      edgecolor="#555577", alpha=0.75), zorder=10)


# ── Qt Dark Palette ───────────────────────────────────────────────────────────

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
    """)


# ── Collapsible Sidebar Section ───────────────────────────────────────────────

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
            ("▾  " if checked else "▸  ") +
            self._toggle.text()[3:]
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


def make_slider(mn, mx, val, decimals=0):
    s = QSlider(Qt.Horizontal)
    s.setMinimum(0)
    s.setMaximum(1000)
    s._mn = mn; s._mx = mx; s._decimals = decimals
    s.setValue(int((val - mn) / (mx - mn) * 1000))

    def real_value():
        v = s.value() / 1000.0 * (s._mx - s._mn) + s._mn
        return round(v, decimals) if decimals > 0 else int(round(v))

    s.real_value = real_value
    return s


# ── Plot Panel ────────────────────────────────────────────────────────────────

class PlotPanel(QWidget):
    def __init__(self, app, panel_index, parent=None):
        super().__init__(parent)
        self.app   = app
        self.index = panel_index
        self._color = FILE_COLORS[0]
        self._ax_history = deque()
        self._ax_cols    = None
        self._hmap_cache = {}
        self._hmap_key   = None
        self._cbar       = None   # active colorbar, or None

        pair = DEFAULT_PAIRS[panel_index % len(DEFAULT_PAIRS)]

        self.setObjectName("plotpanel")
        self.setStyleSheet(
            "QWidget#plotpanel { border: 2px solid " + FILE_COLORS[0] + ";"
            " border-radius: 6px; background: #12122a; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet("background: #0a0a1a; border-radius: 4px;")
        hdr.setFixedHeight(30)
        hl  = QHBoxLayout(hdr)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        self.file_combo = QComboBox()
        self.file_combo.setFixedWidth(110)
        self.file_combo.currentTextChanged.connect(self._on_file_change)

        self.x_combo = QComboBox()
        self.x_combo.addItems(COLUMNS)
        self.x_combo.setCurrentText(pair[0])
        self.x_combo.setFixedWidth(70)
        self.x_combo.currentTextChanged.connect(self._on_axis_change)

        self.y_combo = QComboBox()
        self.y_combo.addItems(COLUMNS)
        self.y_combo.setCurrentText(pair[1])
        self.y_combo.setFixedWidth(70)
        self.y_combo.currentTextChanged.connect(self._on_axis_change)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Auto", "Roll", "Track"])
        self.mode_combo.setCurrentText("Roll")
        self.mode_combo.setFixedWidth(80)
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)

        self.rf_lbl = QLabel("RF")
        self.rf_lbl.setStyleSheet(
            "color: #1a3a2a; background: #1a3a2a; "
            "border-radius: 3px; padding: 1px 4px; font-size: 10px; font-weight: bold;"
        )
        self.rf_lbl.setFixedHeight(18)

        self.bkt_btn = QPushButton("Bkt")
        self.bkt_btn.setCheckable(True)
        self.bkt_btn.setFixedWidth(36)
        self.bkt_btn.setFixedHeight(20)
        self.bkt_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 0; }"
        )
        self.bkt_btn.toggled.connect(lambda _: self.app.render_all())

        for w in [self.file_combo,
                  self._sep(), QLabel("X"), self.x_combo,
                  self._sep(), QLabel("Y"), self.y_combo,
                  self._sep(), self.mode_combo,
                  self.rf_lbl, self.bkt_btn]:
            hl.addWidget(w)
        hl.addStretch()

        layout.addWidget(hdr)

        # ── Matplotlib canvas ─────────────────────────────────────────────
        self.fig = plt.Figure(facecolor=BG, dpi=96)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self._draw_empty()
        self._update_mode_options()

    def _sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setStyleSheet("color: #252535;")
        f.setFixedWidth(1)
        return f

    def _on_file_change(self, label):
        finfo = self.app._file_by_label(label)
        color = finfo["color"] if finfo else FILE_COLORS[0]
        self._color = color
        self.setStyleSheet(
            "QWidget#plotpanel { border: 2px solid " + color + ";"
            " border-radius: 6px; background: #12122a; }"
        )
        self._ax_history.clear()
        self._hmap_cache.clear()
        self.app.render_all()

    def _on_axis_change(self):
        self._ax_history.clear()
        self._hmap_cache.clear()
        self._update_mode_options()
        self.app.render_all()

    def _on_mode_change(self):
        self._ax_history.clear()
        self.app.render_all()

    def _update_mode_options(self):
        xn = self.x_combo.currentText()
        yn = self.y_combo.currentText()
        has_ref = xn in ("p", "t") or yn in ("p", "t")
        cur = self.mode_combo.currentText()
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        if has_ref:
            self.mode_combo.addItems(["Auto", "Roll", "Track",
                                       "Roll+Δ", "Track+Δ"])
        else:
            self.mode_combo.addItems(["Auto", "Roll", "Track"])
            if "Δ" in cur:
                cur = "Roll"
        self.mode_combo.setCurrentText(cur
            if cur in [self.mode_combo.itemText(i)
                       for i in range(self.mode_combo.count())] else "Roll")
        self.mode_combo.blockSignals(False)

    def update_file_list(self, labels, default=None):
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_combo.addItems(labels)
        if default and default in labels:
            self.file_combo.setCurrentText(default)
        self.file_combo.blockSignals(False)
        finfo = self.app._file_by_label(self.file_combo.currentText())
        if finfo:
            self._color = finfo["color"]

    def get_pages(self):
        finfo = self.app._file_by_label(self.file_combo.currentText())
        return finfo["pages"] if finfo else None

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        ax.text(0.5, 0.5, "Open an SDDS file to begin",
                transform=ax.transAxes, ha="center", va="center",
                color="#888888", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

    def render(self, page_idx, settings):
        pages = self.get_pages()
        if not pages:
            self._draw_empty()
            return

        pg_idx = min(page_idx, len(pages) - 1)
        data   = pages[pg_idx]["data"]
        params = pages[pg_idx]["params"]

        xn = self.x_combo.currentText()
        yn = self.y_combo.currentText()
        xi = COLUMNS.index(xn)
        yi = COLUMNS.index(yn)
        xd = data[:, xi].copy()
        yd = data[:, yi].copy()

        # Reference centering (Δ modes)
        center_ref = "Δ" in self.mode_combo.currentText()
        p_central  = params.get("pCentral", None)
        t_central  = params.get("PassCentralTime", None)
        if center_ref:
            if xn == "p" and p_central is not None:
                xd = xd - float(p_central)
            if yn == "p" and p_central is not None:
                yd = yd - float(p_central)
            if xn == "t" and t_central is not None:
                xd = xd - float(t_central)
            if yn == "t" and t_central is not None:
                yd = yd - float(t_central)

        xu   = COL_UNITS.get(xn, "")
        yu   = COL_UNITS.get(yn, "")
        pfx  = "Δ" if center_ref else ""
        xlbl = (pfx + xn + "  [" + xu + "]") if xu else (pfx + xn)
        ylbl = (pfx + yn + "  [" + yu + "]") if yu else (pfx + yn)

        # Axis limits
        ax_mode  = self.mode_combo.currentText()
        col_key  = (xn, yn)
        smooth_n = settings["smooth_n"]

        if col_key != self._ax_cols:
            self._ax_history.clear()
            self._ax_cols = col_key

        xlim = ylim = None

        base_mode = ax_mode.replace("+Δ", "")
        if base_mode == "Roll":
            xc = float(xd.mean()); yc = float(yd.mean())
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

        elif base_mode == "Track":
            nsig = settings["sigma"]
            xc, xs = float(xd.mean()), float(xd.std()) or 1e-10
            yc, ys = float(yd.mean()), float(yd.std()) or 1e-10
            self._ax_history.append((xs, ys))
            while len(self._ax_history) > max(1, smooth_n):
                self._ax_history.popleft()
            arr  = list(self._ax_history)
            sxs  = sum(a[0] for a in arr) / len(arr)
            sys_ = sum(a[1] for a in arr) / len(arr)
            xlim = (xc - nsig * sxs, xc + nsig * sxs)
            ylim = (yc - nsig * sys_, yc + nsig * sys_)

        # Layout
        show_hist  = settings["show_hist"]
        show_cbar  = settings["show_cbar"] and settings["plot_mode"] == "Heatmap 2D"

        # Count only "real" data axes (exclude the colorbar axes matplotlib appends)
        real_axes = [a for a in self.fig.axes if not getattr(a, "_colorbar_axes", False)]
        n_real    = len(real_axes)

        need_rebuild = (
            not real_axes or
            (show_hist and n_real < 3) or
            (not show_hist and n_real != 1) or
            (self._cbar is not None) != show_cbar   # colorbar toggled
        )

        if need_rebuild:
            self.fig.clear()
            self._cbar = None
            self.fig.patch.set_facecolor(BG)
            right_margin = 0.82 if show_cbar else 0.97
            if show_hist:
                gs    = GridSpec(2, 2, figure=self.fig,
                                 width_ratios=[4,1], height_ratios=[1,4],
                                 hspace=0.03, wspace=0.03,
                                 left=0.12, right=right_margin, top=0.97, bottom=0.12)
                ax_s  = self.fig.add_subplot(gs[1, 0])
                ax_hx = self.fig.add_subplot(gs[0, 0], sharex=ax_s)
                ax_hy = self.fig.add_subplot(gs[1, 1], sharey=ax_s)
                for ax in (ax_s, ax_hx, ax_hy):
                    _style_ax(ax)
                ax_main = ax_s
            else:
                ax_main = self.fig.add_subplot(111)
                _style_ax(ax_main)
                self.fig.subplots_adjust(left=0.12, right=right_margin, top=0.97, bottom=0.12)
                ax_hx = ax_hy = None
        else:
            if show_hist:
                ax_main, ax_hx, ax_hy = (self.fig.axes[0],
                                          self.fig.axes[1],
                                          self.fig.axes[2])
                for ax in (ax_main, ax_hx, ax_hy):
                    ax.cla(); _style_ax(ax)
            else:
                ax_main = self.fig.axes[0]
                ax_main.cla(); _style_ax(ax_main)
                ax_hx = ax_hy = None

        # Draw data
        mode = settings["plot_mode"]
        cmap = settings["cmap"]
        pts  = settings["pt_size"]
        alph = settings["alpha"]
        hbins = settings["hmap_bins"]
        log_clr = settings["log_scale"]
        smooth_sigma = settings["smooth_sigma"]

        if mode == "Scatter":
            ax_main.scatter(xd, yd, s=pts, c=self._color,
                            alpha=alph, linewidths=0, zorder=2)
        elif mode == "Heatmap 2D":
            hmap_key = (xn, yn, self.file_combo.currentText(), center_ref)
            if hmap_key != self._hmap_key:
                self._hmap_cache.clear()
                self._hmap_key = hmap_key
            cache_key = (pg_idx, hbins, round(smooth_sigma, 2))
            if cache_key in self._hmap_cache:
                h, xe, ye = self._hmap_cache[cache_key]
            else:
                h, xe, ye = np.histogram2d(xd, yd, bins=hbins)
                h = h.T.astype(float)
                if smooth_sigma > 0:
                    h = gaussian_filter(h, sigma=smooth_sigma)
                if len(self._hmap_cache) > 50:
                    self._hmap_cache.pop(next(iter(self._hmap_cache)))
                self._hmap_cache[cache_key] = (h, xe, ye)

            # Mask after retrieval so smoothed near-zero edges don't show as
            # solid blocks. Use a small threshold rather than exactly 0 so
            # gaussian blur residuals at the boundary also get masked.
            h = np.ma.masked_where(h < h[h > 0].min() * 0.01, h)

            norm = (LogNorm(vmin=max(1e-3, h[h > 0].min()), vmax=h.max())
                    if log_clr else Normalize(vmin=h.min(), vmax=h.max()))
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad(color=AX_BG)
            im = ax_main.imshow(h, origin="lower", aspect="auto",
                                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                                cmap=cmap_obj, norm=norm,
                                interpolation="gaussian", zorder=2)
            # Clamp axes tightly to the data extent so empty border bins don't show.
            # The axis-mode limits below will override this if Roll/Track is active.
            ax_main.set_xlim(xe[0], xe[-1])
            ax_main.set_ylim(ye[0], ye[-1])
            if show_cbar:
                if need_rebuild or self._cbar is None:
                    # First time or settings changed — create the colorbar
                    self._cbar = self.fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.02)
                    self._cbar.ax.tick_params(colors=TEXT_C, labelsize=8)
                    self._cbar.outline.set_edgecolor(SPINE_C)
                    self._cbar.ax.yaxis.set_tick_params(color=TEXT_C)
                    plt.setp(self._cbar.ax.yaxis.get_ticklabels(), color=TEXT_C)
                    self._cbar.ax._colorbar_axes = True
                else:
                    # Reuse existing colorbar — just update its mappable and redraw
                    self._cbar.update_normal(im)

        if show_hist and ax_hx is not None:
            hist_bins = settings["hist_bins"]
            ax_hx.hist(xd, bins=hist_bins, color=self._color, alpha=0.75, linewidth=0)
            ax_hx.tick_params(labelbottom=False, colors=TEXT_C, labelsize=8)
            ax_hy.hist(yd, bins=hist_bins, color=self._color, alpha=0.75,
                       orientation="horizontal", linewidth=0)
            ax_hy.tick_params(labelleft=False, colors=TEXT_C, labelsize=8)

        # Axis limits
        if xlim: ax_main.set_xlim(xlim)
        if ylim: ax_main.set_ylim(ylim)

        # Stats overlay
        if settings["show_overlay"]:
            _draw_overlay(ax_main, xd, yd, self._color, xn, yn, p_central)

        # Particle tracking
        tracked_ids = self.app._tracked_ids
        if tracked_ids:
            fkey = self.file_combo.currentText()
            if fkey not in self.app._traj_cache:
                self.app._traj_cache[fkey] = {}
            for t_idx, tid in enumerate(tracked_ids):
                tcol = TRACK_COLORS[t_idx % len(TRACK_COLORS)]
                if tid not in self.app._traj_cache[fkey]:
                    traj = []
                    for pg in pages:
                        pid_col = pg["data"][:, COLUMNS.index("particleID")]
                        mask    = (pid_col.astype(int) == tid)
                        traj.append(pg["data"][np.where(mask)[0][0]]
                                    if mask.any() else None)
                    self.app._traj_cache[fkey][tid] = traj
                traj = self.app._traj_cache[fkey][tid]
                tx, ty = [], []
                for row in traj[:pg_idx + 1]:
                    if row is not None:
                        tx.append(float(row[xi])); ty.append(float(row[yi]))
                if len(tx) > 1:
                    ax_main.plot(tx, ty, color=tcol, linewidth=0.8,
                                 alpha=0.5, zorder=7)
                if traj[pg_idx] is not None:
                    row = traj[pg_idx]
                    ax_main.scatter([float(row[xi])], [float(row[yi])],
                                    s=60, color=tcol, zorder=8,
                                    linewidths=1.5, edgecolors=self._color)

        # Beam loss overlay
        if self.app._show_loss:
            fkey = self.file_combo.currentText()
            lmap = self.app._loss_cache.get(fkey, {})
            if lmap:
                lost = [pages[pg_idx]["data"][
                            np.where((pages[pg_idx]["data"][:,
                                COLUMNS.index("particleID")].astype(int) == pid))[0][0]]
                        for pid, last_pg in lmap.items()
                        if last_pg == pg_idx and
                        (pages[pg_idx]["data"][:,
                            COLUMNS.index("particleID")].astype(int) == pid).any()]
                if lost:
                    lx = np.array([r[xi] for r in lost])
                    ly = np.array([r[yi] for r in lost])
                    ax_main.scatter(lx, ly, s=12, color="#ff3333",
                                    alpha=0.8, linewidths=0, zorder=9, marker="x")

        # RF bucket
        rf_active = (self.app._show_rf_bucket and center_ref and
                     {xn, yn} == {"t", "p"})
        self.rf_lbl.setStyleSheet(
            "color: #44ff88; background: #0a2a14; "
            "border-radius: 3px; padding: 1px 4px; font-size: 10px; font-weight: bold;"
            if rf_active else
            "color: #1a3a2a; background: #1a3a2a; "
            "border-radius: 3px; padding: 1px 4px; font-size: 10px; font-weight: bold;"
        )

        if rf_active:
            t_mean = float(xd.mean()) if xn == "t" else float(yd.mean())
            p_mean = float(yd.mean()) if yn == "p" else float(xd.mean())
            self._draw_rf(ax_main, params, xn, yn, t_mean, p_mean)

        # Bucket view axis override
        if self.bkt_btn.isChecked() and rf_active:
            self._apply_bucket_view(ax_main, params, xn, yn, xd, yd)

        ax_main.set_xlabel(xlbl, color=TEXT_C, fontsize=10)
        ax_main.set_ylabel(ylbl, color=TEXT_C, fontsize=10)
        self.canvas.draw_idle()

    def _draw_rf(self, ax, params, xn, yn, t_mean, p_mean):
        rf = self.app._rf_params
        if rf is None:
            return
        p_central = params.get("pCentral", None)
        t_central = params.get("PassCentralTime", 0.0)
        if p_central is None:
            return

        cavities_raw = rf.get("cavities", [])
        if self.app._rf_ramp_data is not None:
            ramp  = self.app._rf_ramp_data
            idx   = int(np.argmin(np.abs(ramp["Time"] - float(t_central))))
            cavities_raw = ramp["cavities"][idx]
        if not cavities_raw:
            return

        cavities = [(V, int(h), float(phi_s) * np.pi / 180.0)
                    for V, h, phi_s in cavities_raw]

        dt_sep, delta_sep = compute_rf_separatrix_full(
            cavities, rf.get("alphac", 0.0), p_central,
            rf.get("mass_mev", 0.51099895), rf.get("f_rev_hz", 1e6))
        if dt_sep is None:
            return

        dp_sep = delta_sep * float(p_central)
        if xn == "t" and yn == "p":
            ax.plot(dt_sep + t_mean, dp_sep + p_mean,
                    color="#ffdd44", linewidth=1.4, linestyle="--",
                    alpha=0.85, zorder=8)
        elif xn == "p" and yn == "t":
            ax.plot(dp_sep + p_mean, dt_sep + t_mean,
                    color="#ffdd44", linewidth=1.4, linestyle="--",
                    alpha=0.85, zorder=8)

    def _apply_bucket_view(self, ax, params, xn, yn, xd, yd):
        rf = self.app._rf_params
        if rf is None:
            return
        p_central = params.get("pCentral", None)
        t_central = params.get("PassCentralTime", 0.0)
        if p_central is None:
            return

        cavities_raw = rf.get("cavities", [])
        if self.app._rf_ramp_data is not None:
            ramp = self.app._rf_ramp_data
            idx  = int(np.argmin(np.abs(ramp["Time"] - float(t_central))))
            cavities_raw = ramp["cavities"][idx]
        if not cavities_raw:
            return

        cavities = [(V, int(h), float(phi_s) * np.pi / 180.0)
                    for V, h, phi_s in cavities_raw]
        dt_s, dl_s = compute_rf_separatrix_full(
            cavities, rf.get("alphac", 0.0), p_central,
            rf.get("mass_mev", 0.51099895), rf.get("f_rev_hz", 1e6))
        if dt_s is None:
            return

        dp_s   = dl_s * float(p_central)
        t_mean = float(xd.mean()) if xn == "t" else float(yd.mean())
        p_mean = float(yd.mean()) if yn == "p" else float(xd.mean())
        pad_t  = abs(dt_s).max() * 0.15
        pad_p  = abs(dp_s).max() * 0.15
        ax.set_xlim(t_mean + dt_s.min() - pad_t, t_mean + dt_s.max() + pad_t)
        ax.set_ylim(p_mean + dp_s.min() - pad_p, p_mean + dp_s.max() + pad_p)

    def destroy_panel(self):
        plt.close(self.fig)
        self.deleteLater()


# ── Main Window ───────────────────────────────────────────────────────────────

class SDDSViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDDS Bunch Distribution Viewer v2")
        self.resize(1400, 900)

        # State
        self._files          = []
        self._panels         = []
        self.current_page    = 0
        self._playing        = False
        self._play_timer     = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._tracked_ids    = []
        self._traj_cache     = {}
        self._stats_cache    = {}
        self._loss_cache     = {}
        self._show_loss      = False
        self._rf_params      = None
        self._rf_ramp_data   = None
        self._show_rf_bucket = False
        self._selected_file  = None
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.render_all)

        self._build_ui()
        self._add_panel()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _file_by_label(self, label):
        for f in self._files:
            if f["label"] == label:
                return f
        return None

    def _file_labels(self):
        return [f["label"] for f in self._files]

    def _max_pages(self):
        if not self._files:
            return 0
        return max(len(f["pages"]) for f in self._files)

    def _debounce(self):
        self._debounce_timer.start(120)

    # ── UI Construction ───────────────────────────────────────────────────

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

        tbtn("Open File",     self._open_file,    width=90)
        tbsep()
        tbtn("+ Panel",       self._add_panel,    width=74)
        tbtn("− Panel",       self._remove_panel, "#3a1a1a", width=74)
        tbsep()
        tbtn("Export",        self._export,       "#1a3a1a", width=70)
        tbsep()
        tbtn("Save Session",  self._save_session, "#1a2a1a", width=100)
        tbtn("Load Session",  self._load_session, "#1a1a2a", width=100)
        tbsep()
        self.corr_btn = tbtn("Corr Matrix",  self._open_corr_matrix,
                             "#1a1a3a", width=96)
        self.stats_btn = tbtn("Stats",       self._toggle_stats,
                              "#2a1a3a", checkable=True, width=60)
        self.loss_btn  = tbtn("Beam Loss",   self._toggle_beam_loss,
                              "#3a1a1a", checkable=True, width=86)
        tbsep()
        self.rf_btn = tbtn("RF Bucket",  self._open_rf_dialog,
                           "#1a3a2a", width=86)

        # File legend area
        self._legend_widget = QWidget()
        self._legend_layout = QHBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(8, 0, 8, 0)
        self._legend_layout.setSpacing(8)
        tb.addWidget(self._legend_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._page_lbl  = QLabel("No file loaded")
        self._param_lbl = QLabel("")
        self.status_bar.addWidget(self._page_lbl)
        self.status_bar.addPermanentWidget(self._param_lbl)

        # Central splitter: sidebar | main area
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(228)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sidebar_inner = QWidget()
        self._sidebar_layout = QVBoxLayout(sidebar_inner)
        self._sidebar_layout.setContentsMargins(0, 4, 0, 4)
        self._sidebar_layout.setSpacing(0)
        sidebar_scroll.setWidget(sidebar_inner)

        self._build_sidebar()
        self._sidebar_layout.addStretch()

        main_layout.addWidget(sidebar_scroll)

        # Right area: page slider + grid + stats
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Page slider
        slider_row = QWidget()
        sr_layout  = QHBoxLayout(slider_row)
        sr_layout.setContentsMargins(4, 0, 4, 0)
        self._page_slider = QSlider(Qt.Horizontal)
        self._page_slider.setMinimum(0)
        self._page_slider.setMaximum(1)
        self._page_slider.valueChanged.connect(self._on_slider)
        sr_layout.addWidget(self._page_slider)
        right_layout.addWidget(slider_row)

        # Vertical splitter: panel grid | stats panel
        self._vsplit = QSplitter(Qt.Vertical)

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(4)
        self._vsplit.addWidget(self._grid_widget)

        # Stats panel (hidden by default)
        self._stats_fig    = plt.Figure(facecolor=BG, dpi=96)
        self._stats_canvas = FigureCanvas(self._stats_fig)
        self._stats_canvas.setVisible(False)
        self._vsplit.addWidget(self._stats_canvas)

        right_layout.addWidget(self._vsplit, 1)
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

        self._pt_slider    = make_slider(0.5, 12.0, 2.0, 1)
        self._alpha_slider = make_slider(0.02, 1.0, 0.35, 2)
        self._bins_slider  = make_slider(10, 200, 60)
        self._smooth_slider = make_slider(1, 30, 1)
        self._sigma_slider = make_slider(0.5, 10.0, 3.0, 1)

        for s in [self._pt_slider, self._alpha_slider,
                  self._bins_slider, self._smooth_slider, self._sigma_slider]:
            s.valueChanged.connect(self._debounce)

        sec.add_row("Point size",           self._pt_slider)
        sec.add_row("Alpha",                self._alpha_slider)
        sec.add_row("Histogram bins",       self._bins_slider)
        sec.add_row("Axis smoothing (frames)", self._smooth_slider)
        sec.add_row("Track window (±σ)",    self._sigma_slider)
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

        self._hbins_slider  = make_slider(50, 500, 300)
        self._hsmooth_slider = make_slider(0.0, 8.0, 2.0, 1)
        self._hbins_slider.valueChanged.connect(self._debounce)
        self._hsmooth_slider.valueChanged.connect(self._debounce)
        self._log_cb = QCheckBox("Log color scale")
        self._log_cb.setChecked(True)
        self._log_cb.toggled.connect(self.render_all)

        self._cbar_cb = QCheckBox("Show colorbar")
        self._cbar_cb.setChecked(False)
        self._cbar_cb.toggled.connect(self.render_all)

        sec2.add_row("Heatmap bins",      self._hbins_slider)
        sec2.add_row("Smoothing (sigma)", self._hsmooth_slider)
        sec2.add(self._log_cb)
        sec2.add(self._cbar_cb)
        sl.addWidget(sec2)

        # ── PLAYBACK ─────────────────────────────────────────────────────
        sec3 = SidebarSection("PLAYBACK")

        play_row = QWidget()
        pr_layout = QHBoxLayout(play_row)
        pr_layout.setContentsMargins(0, 0, 0, 0)
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedWidth(90)
        self._play_btn.clicked.connect(self._toggle_play)
        self._speed_slider = make_slider(1, 30, 5)
        pr_layout.addWidget(self._play_btn)
        sec3.add(play_row)
        sec3.add_row("Speed (fps)", self._speed_slider)
        sl.addWidget(sec3)

        # ── PARTICLE TRACKING ────────────────────────────────────────────
        sec4 = SidebarSection("PARTICLE TRACKING")

        self._track_entry = QLineEdit()
        self._track_entry.setPlaceholderText("e.g. 42,100,203")
        track_btn = QPushButton("Track")
        track_btn.setFixedWidth(60)
        track_btn.clicked.connect(self._set_tracking)
        clear_btn = QPushButton("Clear tracking")
        clear_btn.clicked.connect(self._clear_tracking)
        clear_btn.setStyleSheet("background: #3a1a1a;")

        track_row = QWidget()
        trl = QHBoxLayout(track_row)
        trl.setContentsMargins(0, 0, 0, 0)
        trl.addWidget(self._track_entry, 1)
        trl.addWidget(track_btn)

        self._track_lbl = QLabel("No particle tracked")
        self._track_lbl.setStyleSheet("color: #666688; font-size: 10px;")

        sec4.add(track_row)
        sec4.add(clear_btn)
        sec4.add(self._track_lbl)
        sl.addWidget(sec4)

    # ── Settings accessor ─────────────────────────────────────────────────

    def _get_settings(self):
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
            "show_cbar":    self._cbar_cb.isChecked(),
        }

    # ── Panel management ──────────────────────────────────────────────────

    def _add_panel(self):
        idx   = len(self._panels)
        panel = PlotPanel(self, idx)
        self._panels.append(panel)
        labels = self._file_labels()
        if labels:
            panel.update_file_list(labels, labels[0])
        self._reflow_grid()
        if self._files:
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

        # Clear grid
        for i in reversed(range(self._grid_layout.count())):
            self._grid_layout.itemAt(i).widget().setParent(None)

        for i, panel in enumerate(self._panels):
            r, c = divmod(i, cols)
            self._grid_layout.addWidget(panel, r, c)

        # Equal stretch
        for c in range(cols):
            self._grid_layout.setColumnStretch(c, 1)
        rows = (n + cols - 1) // cols
        for r in range(rows):
            self._grid_layout.setRowStretch(r, 1)

    # ── File legend ───────────────────────────────────────────────────────

    def _update_legend(self):
        # Clear
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for finfo in self._files:
            btn = QPushButton(f"● {finfo['label']}")
            btn.setStyleSheet(
                f"QPushButton {{ color: {finfo['color']}; background: transparent; "
                f"border: none; font-size: 11px; padding: 2px 6px; }}"
                f"QPushButton:checked {{ background: #1e1e3a; border-radius: 4px; }}"
            )
            btn.setCheckable(True)
            btn.setChecked(finfo["label"] == self._selected_file)
            label = finfo["label"]
            btn.clicked.connect(lambda checked, l=label: self._select_file(l))
            self._legend_layout.addWidget(btn)

    def _select_file(self, label):
        self._selected_file = label
        self._update_legend()
        if self.stats_btn.isChecked():
            self._draw_stats_panel()

    # ── File opening ──────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SDDS file", "", "All files (*.*)")
        if not path:
            return
        try:
            pages = read_sdds_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Parse error", str(e))
            return
        if not pages:
            QMessageBox.warning(self, "Empty file", "No data pages found.")
            return

        color = FILE_COLORS[len(self._files) % len(FILE_COLORS)]
        label = Path(path).stem[:18]
        self._files.append({"label": label, "path": path,
                             "color": color, "pages": pages})
        self._traj_cache[label]  = {}
        self._stats_cache[label] = None
        if self._selected_file is None:
            self._selected_file = label

        labels = self._file_labels()
        for panel in self._panels:
            panel.update_file_list(labels, labels[-1])

        n = self._max_pages()
        self._page_slider.setMaximum(max(1, n - 1))
        self._page_slider.setValue(0)
        self.current_page = 0
        self._update_legend()
        self.render_all()

    # ── Rendering ─────────────────────────────────────────────────────────

    def render_all(self):
        if not self._files:
            return
        settings = self._get_settings()
        for panel in self._panels:
            panel.render(self.current_page, settings)

        # Status bar
        pg   = None
        for f in self._files:
            if len(f["pages"]) > self.current_page:
                pg = f["pages"][self.current_page]
                break
        if pg:
            n     = self._max_pages()
            step  = pg["params"].get("Step", "?")
            s_val = pg["params"].get("s")
            npart = pg["data"].shape[0]
            s_str = f"   s = {s_val:.4f} m" if isinstance(s_val, float) else ""
            self._page_lbl.setText(f"Page {self.current_page + 1} / {n}")
            self._param_lbl.setText(
                f"Step {step}{s_str}   |   {npart:,} particles")

        if self.stats_btn.isChecked():
            self._draw_stats_panel()

    # ── Slider / playback ─────────────────────────────────────────────────

    def _on_slider(self, val):
        if not self._files:
            return
        self.current_page = val
        self.render_all()

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._play_timer.stop()
            self._play_btn.setText("▶  Play")
        else:
            if not self._files:
                return
            self._playing = True
            self._play_btn.setText("⏸  Pause")
            fps = max(1, self._speed_slider.real_value())
            self._play_timer.start(int(1000 / fps))

    def _advance_frame(self):
        if not self._playing:
            return
        n = self._max_pages()
        if n == 0:
            return
        fps = max(1, self._speed_slider.real_value())
        self._play_timer.setInterval(int(1000 / fps))
        self.current_page = (self.current_page + 1) % n
        self._page_slider.blockSignals(True)
        self._page_slider.setValue(self.current_page)
        self._page_slider.blockSignals(False)
        self.render_all()

    # ── Particle tracking ─────────────────────────────────────────────────

    def _set_tracking(self):
        txt = self._track_entry.text().strip()
        try:
            ids = [int(x.strip()) for x in txt.split(",") if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Invalid IDs",
                                "Enter comma-separated integers.")
            return
        self._tracked_ids = ids
        self._traj_cache  = {l: {} for l in self._file_labels()}
        self._track_lbl.setText(f"Tracking: {ids}" if ids else "No particle tracked")
        if ids:
            self._track_lbl.setStyleSheet("color: #44ff88; font-size: 10px;")
        self.render_all()

    def _clear_tracking(self):
        self._tracked_ids = []
        self._traj_cache  = {l: {} for l in self._file_labels()}
        self._track_lbl.setText("No particle tracked")
        self._track_lbl.setStyleSheet("color: #666688; font-size: 10px;")
        self.render_all()

    # ── Beam loss ─────────────────────────────────────────────────────────

    def _toggle_beam_loss(self, checked):
        if not checked:
            self._show_loss = False
            self.render_all()
            return

        total = sum(len(f["pages"]) * 10000 for f in self._files)
        if total > 5_000_000:
            r = QMessageBox.question(self, "Beam Loss",
                f"Large dataset (~{total:,} particle-pages). Continue?",
                QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                self.loss_btn.setChecked(False)
                return

        for finfo in self._files:
            label, pages = finfo["label"], finfo["pages"]
            lmap = {}
            for pg_idx, pg in enumerate(pages):
                pid_col = pg["data"][:, COLUMNS.index("particleID")].astype(int)
                for pid in pid_col:
                    lmap[pid] = pg_idx
            # Keep only lost particles (not in last page)
            last_pids = set(pages[-1]["data"][:,
                COLUMNS.index("particleID")].astype(int))
            self._loss_cache[label] = {
                pid: pg for pid, pg in lmap.items() if pid not in last_pids}

        self._show_loss = True
        self.render_all()

    # ── Stats over time ───────────────────────────────────────────────────

    def _toggle_stats(self, checked):
        self._stats_canvas.setVisible(checked)
        if checked:
            total = self._vsplit.height()
            self._vsplit.setSizes([int(total * 0.60), int(total * 0.40)])
            self._draw_stats_panel()
        else:
            total = self._vsplit.height()
            self._vsplit.setSizes([total, 0])

    def _draw_stats_panel(self):
        if not self._files:
            return

        files_to_show = self._files
        if self._selected_file:
            sel = self._file_by_label(self._selected_file)
            if sel:
                files_to_show = [sel]

        STAT_COLS = ["x", "xp", "y", "yp", "t", "p", "dt"]
        CONJ      = [("x","xp"), ("y","yp")]

        self._stats_fig.clear()
        self._stats_fig.patch.set_facecolor(BG)
        gs = GridSpec(3, 3, figure=self._stats_fig,
                      hspace=0.6, wspace=0.4,
                      left=0.07, right=0.97, top=0.95, bottom=0.12)
        axes = [self._stats_fig.add_subplot(gs[i//3, i%3])
                for i in range(9)]
        titles = ["x", "xp", "y", "yp", "t", "p", "dt",
                  "ε_x", "ε_y"]

        for ax, title in zip(axes, titles):
            _style_ax(ax)
            ax.set_title(title, color=TEXT_C, fontsize=9, pad=2)

        for finfo in files_to_show:
            label  = finfo["label"]
            pages  = finfo["pages"]
            color  = finfo["color"]

            if self._stats_cache.get(label) is None:
                sc = {col: {"mean":[], "sigma":[], "mn":[], "mx":[]}
                      for col in STAT_COLS}
                emit = {pair: [] for pair in CONJ}
                for pg in pages:
                    data = pg["data"]
                    for col in STAT_COLS:
                        ci = COLUMNS.index(col)
                        cd = data[:, ci]
                        sc[col]["mean"].append(float(cd.mean()))
                        sc[col]["sigma"].append(float(cd.std()))
                        sc[col]["mn"].append(float(cd.min()))
                        sc[col]["mx"].append(float(cd.max()))
                    for cn, cm in CONJ:
                        xd = data[:, COLUMNS.index(cn)] - data[:, COLUMNS.index(cn)].mean()
                        yd = data[:, COLUMNS.index(cm)] - data[:, COLUMNS.index(cm)].mean()
                        det = float(np.mean(xd**2)) * float(np.mean(yd**2)) - float(np.mean(xd*yd))**2
                        emit[(cn,cm)].append(float(np.sqrt(max(det, 0))))
                for col in STAT_COLS:
                    for k in sc[col]:
                        sc[col][k] = np.array(sc[col][k])
                for pair in CONJ:
                    emit[pair] = np.array(emit[pair])
                self._stats_cache[label] = {"stats": sc, "emit": emit,
                                             "n_pages": len(pages)}

            sc_data = self._stats_cache[label]
            px      = np.arange(1, sc_data["n_pages"] + 1)

            for i, col in enumerate(STAT_COLS):
                ax  = axes[i]
                s   = sc_data["stats"][col]
                mu  = s["mean"]; sig = s["sigma"]
                ax.plot(px, mu, color=color, linewidth=1.0)
                ax.fill_between(px, mu - sig, mu + sig,
                                alpha=0.25, color=color)
                ax.plot(px, s["mn"], color=color, linewidth=0.5,
                        linestyle=":")
                ax.plot(px, s["mx"], color=color, linewidth=0.5,
                        linestyle=":")

            for i, pair in enumerate(CONJ):
                ax = axes[7 + i]
                ax.plot(px, sc_data["emit"][pair],
                        color=color, linewidth=1.0)

        # Current page line — only draw if axes has a non-degenerate x range
        # (single-page files produce a singular transform matrix that crashes axvline)
        cp = self.current_page + 1
        for ax in axes:
            xlim = ax.get_xlim()
            if xlim[1] - xlim[0] > 0:
                ax.axvline(cp, color="white", linewidth=0.8, alpha=0.6)

        self._stats_canvas.draw_idle()

    # ── Correlation matrix ────────────────────────────────────────────────

    def _open_corr_matrix(self):
        if not self._files:
            QMessageBox.warning(self, "Corr Matrix", "Load a file first.")
            return

        finfo = None
        for f in self._files:
            if f["label"] == self._selected_file:
                finfo = f
                break
        if finfo is None:
            finfo = self._files[0]

        pg   = finfo["pages"][self.current_page
               if self.current_page < len(finfo["pages"])
               else 0]
        data = pg["data"]
        cols = ["x", "xp", "y", "yp", "t", "p"]
        nc   = len(cols)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Correlation Matrix — {finfo['label']}")
        dlg.resize(700, 650)
        layout = QVBoxLayout(dlg)

        fig  = plt.Figure(facecolor=BG, dpi=96)
        canv = FigureCanvas(fig)
        layout.addWidget(canv)

        gs = GridSpec(nc, nc, figure=fig,
                      hspace=0.15, wspace=0.15,
                      left=0.08, right=0.97, top=0.97, bottom=0.08)

        color = finfo["color"]
        for i in range(nc):
            for j in range(nc):
                ax = fig.add_subplot(gs[i, j])
                _style_ax(ax)
                ax.tick_params(labelsize=7)
                xd = data[:, COLUMNS.index(cols[j])]
                yd = data[:, COLUMNS.index(cols[i])]
                if i == j:
                    ax.hist(xd, bins=60, color=color, alpha=0.8, linewidth=0)
                else:
                    ax.scatter(xd, yd, s=0.3, c=color, alpha=0.3, linewidths=0)
                    r = float(np.corrcoef(xd, yd)[0, 1])
                    ax.text(0.05, 0.95, f"r={r:.2f}",
                            transform=ax.transAxes, fontsize=8,
                            color=TEXT_C, va="top")
                if i == nc - 1:
                    ax.set_xlabel(cols[j], color=TEXT_C, fontsize=8)
                if j == 0:
                    ax.set_ylabel(cols[i], color=TEXT_C, fontsize=8)

        canv.draw()
        dlg.exec()
        plt.close(fig)

    # ── RF Bucket ─────────────────────────────────────────────────────────

    def _open_rf_dialog(self):
        ex       = self._rf_params or {}
        ex_cavs  = ex.get("cavities", [(1e6, 1, 0.0)])
        ex_mass  = ex.get("mass_mev", 0.51099895)
        ex_ac    = ex.get("alphac", 0.0)
        ex_frev  = ex.get("f_rev_hz", 1e6) / 1e6
        ex_mode  = ex.get("mode", "Static")
        ex_species = next((sp for sp, m in PARTICLE_MASSES.items()
                           if abs(m - ex_mass) < 0.001), "Custom")

        dlg = QDialog(self)
        dlg.setWindowTitle("RF Bucket Configuration" +
                           (" — Edit" if ex else ""))
        dlg.resize(500, 580)
        layout = QVBoxLayout(dlg)

        # Particle & lattice
        grp1 = QGroupBox("Particle & Lattice")
        gl1  = QGridLayout(grp1)

        gl1.addWidget(QLabel("Species:"), 0, 0)
        species_combo = QComboBox()
        species_combo.addItems(["Electron", "Proton", "Custom"])
        species_combo.setCurrentText(ex_species)
        gl1.addWidget(species_combo, 0, 1)

        gl1.addWidget(QLabel("Mass (MeV):"), 0, 2)
        mass_spin = QDoubleSpinBox()
        mass_spin.setDecimals(6); mass_spin.setRange(0.001, 10000)
        mass_spin.setValue(ex_mass)
        gl1.addWidget(mass_spin, 0, 3)

        def on_species(text):
            if text in PARTICLE_MASSES:
                mass_spin.setValue(PARTICLE_MASSES[text])
        species_combo.currentTextChanged.connect(on_species)

        gl1.addWidget(QLabel("alphac:"), 1, 0)
        ac_spin = QDoubleSpinBox()
        ac_spin.setDecimals(8); ac_spin.setRange(-1, 1)
        ac_spin.setValue(ex_ac)
        gl1.addWidget(ac_spin, 1, 1)

        gl1.addWidget(QLabel("f_rev (MHz):"), 1, 2)
        frev_spin = QDoubleSpinBox()
        frev_spin.setDecimals(6); frev_spin.setRange(0.001, 1000)
        frev_spin.setValue(ex_frev)
        gl1.addWidget(frev_spin, 1, 3)
        layout.addWidget(grp1)

        # RF Mode
        grp2   = QGroupBox("RF Mode")
        gl2    = QHBoxLayout(grp2)
        static_rb = QRadioButton("Static")
        ramp_rb   = QRadioButton("Ramp (CSV)")
        static_rb.setChecked(ex_mode == "Static")
        ramp_rb.setChecked(ex_mode == "Ramp")
        gl2.addWidget(static_rb); gl2.addWidget(ramp_rb)
        layout.addWidget(grp2)

        # Cavities
        grp3 = QGroupBox("Cavities  (V in Volts, phi_s in degrees)")
        gl3  = QVBoxLayout(grp3)

        hdr_row = QWidget()
        hrl = QHBoxLayout(hdr_row)
        hrl.setContentsMargins(0,0,0,0)
        for txt, w in [("#", 30), ("Voltage (V)", 120),
                       ("Harmonic h", 100), ("phi_s (deg)", 110)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("color: #666688; font-size: 10px;")
            hrl.addWidget(lbl)
        gl3.addWidget(hdr_row)

        cavity_rows = []
        cav_container = QWidget()
        cav_layout = QVBoxLayout(cav_container)
        cav_layout.setContentsMargins(0, 0, 0, 0)
        cav_layout.setSpacing(2)
        gl3.addWidget(cav_container)

        def add_cavity(V=1e6, h=1, phi=0.0):
            row = QWidget()
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            num = QLabel(str(len(cavity_rows)+1))
            num.setFixedWidth(20)
            v_spin = QDoubleSpinBox()
            v_spin.setDecimals(0); v_spin.setRange(0, 1e9); v_spin.setValue(V)
            v_spin.setFixedWidth(110)
            h_spin = QSpinBox()
            h_spin.setRange(1, 100000); h_spin.setValue(int(h))
            h_spin.setFixedWidth(90)
            p_spin = QDoubleSpinBox()
            p_spin.setDecimals(4); p_spin.setRange(-360, 360); p_spin.setValue(phi)
            p_spin.setFixedWidth(100)
            for w in [num, v_spin, h_spin, p_spin]: rl.addWidget(w)
            cav_layout.addWidget(row)
            cavity_rows.append((v_spin, h_spin, p_spin))

        for V, h, phi in ex_cavs:
            add_cavity(V, h, phi)

        add_cav_btn = QPushButton("+ Cavity")
        add_cav_btn.setFixedWidth(90)
        add_cav_btn.clicked.connect(lambda: add_cavity())
        gl3.addWidget(add_cav_btn)
        layout.addWidget(grp3)

        # Ramp CSV
        grp4 = QGroupBox("Ramp CSV")
        gl4  = QHBoxLayout(grp4)
        ramp_lbl = QLabel("No ramp file loaded")
        ramp_lbl.setStyleSheet("color: #666688; font-size: 10px;")
        load_csv_btn = QPushButton("Load CSV")
        load_csv_btn.setFixedWidth(90)
        gl4.addWidget(ramp_lbl, 1)
        gl4.addWidget(load_csv_btn)
        layout.addWidget(grp4)

        def load_ramp():
            path, _ = QFileDialog.getOpenFileName(
                dlg, "Load RF ramp CSV", "",
                "CSV files (*.csv *.txt *.dat);;All files (*.*)")
            if not path:
                return
            try:
                with open(path) as f:
                    first = f.readline().strip()
                delim = "," if "," in first else ("\t" if "\t" in first else None)
                data_rows = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        vals = ([v.strip() for v in line.split(delim)]
                                if delim else line.split())
                        data_rows.append([float(v) for v in vals])
                if not data_rows:
                    raise ValueError("No data rows found")
                arr    = np.array(data_rows)
                times  = arr[:, 0]
                n_cav  = (arr.shape[1] - 1) // 3
                cav_data = []
                for row in arr:
                    cavs = []
                    for i in range(n_cav):
                        b = 1 + i * 3
                        cavs.append((float(row[b]), int(row[b+1]), float(row[b+2])))
                    cav_data.append(cavs)
                self._rf_ramp_data = {"Time": times, "cavities": cav_data}
                ramp_lbl.setText(
                    Path(path).name +
                    f" ({len(data_rows)} steps, {n_cav} cavit" +
                    ("y)" if n_cav == 1 else "ies)"))
            except Exception as e:
                QMessageBox.critical(dlg, "Ramp load error", str(e))

        load_csv_btn.clicked.connect(load_ramp)

        # Buttons
        btn_row = QWidget()
        brl = QHBoxLayout(btn_row)
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background: #1a4a2a;")
        clear_btn = QPushButton("Clear & Close")
        clear_btn.setStyleSheet("background: #3a1a1a;")
        cancel_btn = QPushButton("Cancel")
        brl.addWidget(apply_btn)
        brl.addWidget(clear_btn)
        brl.addWidget(cancel_btn)
        layout.addWidget(btn_row)

        def apply():
            try:
                cavs = [(v_sp.value(), h_sp.value(), p_sp.value())
                        for v_sp, h_sp, p_sp in cavity_rows]
                self._rf_params = {
                    "mass_mev": mass_spin.value(),
                    "alphac":   ac_spin.value(),
                    "f_rev_hz": frev_spin.value() * 1e6,
                    "cavities": cavs,
                    "mode":     "Static" if static_rb.isChecked() else "Ramp",
                }
                self._show_rf_bucket = True
                self.rf_btn.setStyleSheet("background: #226633;")
                self.rf_btn.setText("* RF Bucket")
                self.render_all()
                dlg.accept()
            except Exception as e:
                QMessageBox.critical(dlg, "RF config error", str(e))

        def clear():
            self._rf_params      = None
            self._rf_ramp_data   = None
            self._show_rf_bucket = False
            self.rf_btn.setStyleSheet("")
            self.rf_btn.setText("RF Bucket")
            self.render_all()
            dlg.accept()

        apply_btn.clicked.connect(apply)
        clear_btn.clicked.connect(clear)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()

    # ── Export ────────────────────────────────────────────────────────────

    def _export(self):
        if not self._files:
            QMessageBox.warning(self, "Export", "No data loaded.")
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
            pages = panel.get_pages()
            if not pages:
                continue
            pg_idx = min(self.current_page, len(pages) - 1)
            data   = pages[pg_idx]["data"]
            xn = panel.x_combo.currentText()
            yn = panel.y_combo.currentText()
            xd = data[:, COLUMNS.index(xn)]
            yd = data[:, COLUMNS.index(yn)]
            xu = COL_UNITS.get(xn, "")
            yu = COL_UNITS.get(yn, "")

            if settings["plot_mode"] == "Scatter":
                ax.scatter(xd, yd, s=settings["pt_size"],
                           c=panel._color, alpha=settings["alpha"],
                           linewidths=0, zorder=2)
            ax.set_xlabel((xn+"  ["+xu+"]") if xu else xn,
                          color=TEXT_C, fontsize=10)
            ax.set_ylabel((yn+"  ["+yu+"]") if yu else yn,
                          color=TEXT_C, fontsize=10)
            ax.tick_params(colors=TEXT_C, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor(SPINE_C)
            ax.set_facecolor(AX_BG)
            ax.set_title(panel.file_combo.currentText(),
                         color=TEXT_C, fontsize=9, pad=3)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
        if not path.endswith(".pdf"):
            pdf_path = path.rsplit(".", 1)[0] + ".pdf"
            fig.savefig(pdf_path, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        QMessageBox.information(self, "Export", f"Saved to {path}")

    # ── Session save/load ─────────────────────────────────────────────────

    def _save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "Session (*.json)")
        if not path:
            return
        try:
            session = {
                "files":        [{"path": f["path"], "label": f["label"]}
                                  for f in self._files],
                "current_page": self.current_page,
                "panels": [{
                    "file_label": p.file_combo.currentText(),
                    "x":          p.x_combo.currentText(),
                    "y":          p.y_combo.currentText(),
                    "ax_mode":    p.mode_combo.currentText(),
                    "bkt":        p.bkt_btn.isChecked(),
                } for p in self._panels],
                "plot_mode":    "Scatter" if self._mode_scatter.isChecked()
                                else "Heatmap 2D",
                "cmap":         self._cmap_combo.currentText(),
                "hmap_bins":    self._hbins_slider.real_value(),
                "smooth_sigma": self._hsmooth_slider.real_value(),
                "log_scale":    self._log_cb.isChecked(),
                "show_hist":    self._hist_cb.isChecked(),
                "hist_bins":    self._bins_slider.real_value(),
                "pt_size":      self._pt_slider.real_value(),
                "alpha":        self._alpha_slider.real_value(),
                "smooth_n":     self._smooth_slider.real_value(),
                "sigma":        self._sigma_slider.real_value(),
                "overlay":      self._overlay_cb.isChecked(),
                "rf_params":    self._rf_params,
                "show_rf":      self._show_rf_bucket,
            }
            with open(path, "w") as f:
                json.dump(session, f, indent=2)
            QMessageBox.information(self, "Session saved",
                                    f"Saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "Session (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                session = json.load(f)

            for fentry in session.get("files", []):
                fpath = fentry["path"]
                if not Path(fpath).exists():
                    QMessageBox.warning(self, "File missing",
                        f"{fpath} not found — skipping.")
                    continue
                pages = read_sdds_file(fpath)
                color = FILE_COLORS[len(self._files) % len(FILE_COLORS)]
                label = Path(fpath).stem[:18]
                self._files.append({"label": label, "path": fpath,
                                     "color": color, "pages": pages})
                self._traj_cache[label]  = {}
                self._stats_cache[label] = None

            if not self._files:
                return

            # Restore settings
            if session.get("plot_mode") == "Heatmap 2D":
                self._mode_heatmap.setChecked(True)
            else:
                self._mode_scatter.setChecked(True)
            self._cmap_combo.setCurrentText(session.get("cmap", "turbo"))
            self._log_cb.setChecked(session.get("log_scale", True))
            self._hist_cb.setChecked(session.get("show_hist", False))
            self._overlay_cb.setChecked(session.get("overlay", False))

            self._rf_params      = session.get("rf_params", None)
            self._show_rf_bucket = session.get("show_rf", False)
            if self._rf_params:
                self.rf_btn.setText("* RF Bucket")
                self.rf_btn.setStyleSheet("background: #226633;")

            # Panels
            saved_panels = session.get("panels", [])
            labels = self._file_labels()
            while len(self._panels) < len(saved_panels):
                self._add_panel()
            while len(self._panels) > len(saved_panels) and len(self._panels) > 1:
                p = self._panels.pop()
                p.destroy_panel()

            for panel, pdata in zip(self._panels, saved_panels):
                panel.update_file_list(labels, pdata.get("file_label", labels[0]))
                panel.x_combo.setCurrentText(pdata.get("x", "t"))
                panel.y_combo.setCurrentText(pdata.get("y", "p"))
                panel._update_mode_options()
                panel.mode_combo.setCurrentText(pdata.get("ax_mode", "Roll"))
                panel.bkt_btn.setChecked(pdata.get("bkt", False))

            self._reflow_grid()
            self._update_legend()

            n = self._max_pages()
            self.current_page = min(session.get("current_page", 0), n - 1)
            self._page_slider.setMaximum(max(1, n - 1))
            self._page_slider.setValue(self.current_page)
            self.render_all()

        except Exception as e:
            QMessageBox.critical(self, "Load session error", str(e))

    def closeEvent(self, event):
        self._playing = False
        self._play_timer.stop()
        for panel in self._panels:
            plt.close(panel.fig)
        plt.close(self._stats_fig)
        event.accept()

    def keyPressEvent(self, event):
        key  = event.key()
        mods = event.modifiers()
        if key == Qt.Key_O and mods & Qt.ControlModifier:
            self._open_file()
        elif key == Qt.Key_Space:
            self._toggle_play()
        elif key in (Qt.Key_Right, Qt.Key_Up):
            if self._files:
                n = self._max_pages()
                self.current_page = min(self.current_page + 1, n - 1)
                self._page_slider.blockSignals(True)
                self._page_slider.setValue(self.current_page)
                self._page_slider.blockSignals(False)
                self.render_all()
        elif key in (Qt.Key_Left, Qt.Key_Down):
            if self._files:
                self.current_page = max(self.current_page - 1, 0)
                self._page_slider.blockSignals(True)
                self._page_slider.setValue(self.current_page)
                self._page_slider.blockSignals(False)
                self.render_all()
        else:
            super().keyPressEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("SDDS Bunch Distribution Viewer v2")
    apply_dark_palette(app)
    window = SDDSViewer()
    window.show()
    sys.exit(app.exec())
