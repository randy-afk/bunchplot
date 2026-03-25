#!/usr/bin/env python3

"""
SDDS Bunch Distribution Viewer
================================
CustomTkinter GUI for viewing SDDS binary particle data files.

Features:
  - Scatter plot with any two columns on X / Y (fully custom dropdowns)
  - Toggleable marginal histograms aligned to each axis
  - Page slider + Play/Pause animation with adjustable speed
  - Live per-axis stats (mean, std) in sidebar

SDDS format (little-endian binary):
  8 columns per particle: x, xp, y, yp, t, p, dt, particleID
  19 parameters per page (stored in background)

Install dependencies:
    pip install customtkinter matplotlib numpy

Run:
    python sdds_viewer.py
"""

import struct
from collections import deque
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Column / parameter definitions ───────────────────────────────────────────
COLUMNS = ["x", "xp", "y", "yp", "t", "p", "dt", "particleID"]
COL_UNITS = {
    "x": "m", "xp": "", "y": "m", "yp": "",
    "t": "s", "p": "m·βγ", "dt": "s", "particleID": "",
}

# 7 little-endian doubles + 1 uint64
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

PARAM_DEFS = [
    ("Step",                     "long"),
    ("pCentral",                 "double"),
    ("Charge",                   "double"),
    ("Particles",                "long"),
    ("IDSlotsPerBunch",          "long"),
    ("SVNVersion",               "string"),
    ("SampledCharge",            "double"),
    ("SampledParticles",         "long"),
    ("Pass",                     "long"),
    ("PassLength",               "double"),
    ("PassCentralTime",          "double"),
    ("ElapsedTime",              "double"),
    ("ElapsedCoreTime",          "double"),
    ("MemoryUsage",              "long"),
    ("s",                        "double"),
    ("Description",              "string"),
    ("PreviousElementName",      "string"),
    ("PreviousElementOccurence", "long"),
    ("PreviousElementTag",       "string"),
]

PARAM_TYPE_MAP = {name: ptype for name, ptype in PARAM_DEFS}


# ── SDDS Parser ───────────────────────────────────────────────────────────────

def _read_param(raw: bytes, pos: int, ptype: str):
    """Read one parameter value from raw bytes at pos. Returns (value, new_pos)."""
    if ptype == "double":
        val = struct.unpack_from("<d", raw, pos)[0]
        return val, pos + 8
    elif ptype in ("long", "short"):
        val = struct.unpack_from("<i", raw, pos)[0]
        return val, pos + 4
    elif ptype == "string":
        slen = struct.unpack_from("<i", raw, pos)[0]
        pos += 4
        if slen < 0 or slen > 1_000_000:
            raise ValueError(f"Implausible string length {slen}")
        val = raw[pos:pos + slen].decode("latin-1", errors="replace")
        return val, pos + slen
    return None, pos


def _parse_header(raw: bytes):
    """
    Parse the ASCII header of a binary SDDS file.
    Returns (binary_start_pos, param_defs, fixed_params) where:
      - binary_start_pos: byte offset where binary data begins
      - param_defs: list of (name, type) for parameters that ARE in binary
      - fixed_params: dict of {name: value} for fixed_value parameters
    """
    import re

    # Find the &data line which terminates the header
    data_pos = raw.find(b"&data")
    if data_pos == -1:
        raise ValueError("Could not find '&data' in header — not a valid SDDS file.")
    nl = raw.find(b"\n", data_pos)
    if nl == -1:
        raise ValueError("Malformed header: no newline after '&data'.")
    binary_start = nl + 1

    header_text = raw[:binary_start].decode("latin-1", errors="replace")

    # Extract all &parameter blocks
    param_blocks = re.findall(r'&parameter(.*?)&end', header_text, re.DOTALL)

    param_defs   = []   # parameters stored in binary (no fixed_value)
    fixed_params = {}   # parameters with fixed_value (not in binary)

    for block in param_blocks:
        name_m  = re.search(r'name\s*=\s*(\w+)',        block)
        type_m  = re.search(r'type\s*=\s*(\w+)',        block)
        fixed_m = re.search(r'fixed_value\s*=\s*(\S+)', block)
        if not name_m or not type_m:
            continue
        name  = name_m.group(1)
        ptype = type_m.group(1)
        if fixed_m:
            # Convert fixed value to correct Python type
            fv = fixed_m.group(1).rstrip(',')
            try:
                fixed_params[name] = float(fv) if ptype == "double" else int(fv)
            except ValueError:
                fixed_params[name] = fv
        else:
            param_defs.append((name, ptype))

    return binary_start, param_defs, fixed_params


def read_sdds_file(filepath: str) -> list:
    """
    Parse a binary SDDS file. Returns list of page dicts:
        {"params": {name: value, ...}, "data": np.ndarray shape (N, 8)}

    Layout per page (little-endian):
      [int32 n_rows] [parameters in header order, skipping fixed_value ones] [N × particle rows]
    """
    # Read header first (small — read only up to &data line)
    with open(filepath, "rb") as f:
        header_raw = f.read(65536)  # 64 KB is always enough for the header
    binary_start, param_defs, fixed_params = _parse_header(header_raw)

    pages = []

    with open(filepath, "rb") as f:
        f.seek(binary_start)

        while True:
            # Row count (4 bytes)
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            n_rows = struct.unpack("<i", hdr)[0]
            if n_rows < 0 or n_rows > 10_000_000:
                break
            if n_rows == 0:
                continue

            # Parameters (only those without fixed_value)
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
                        if slen < 0 or slen > 1_000_000: raise ValueError(f"Bad string len {slen}")
                        params[pname] = f.read(slen).decode("latin-1", errors="replace")
                except (EOFError, struct.error, ValueError):
                    ok = False
                    break
            if not ok:
                break

            # Particle data — read directly from file, copy so buffer is self-owned
            byte_count = n_rows * PARTICLE_SIZE
            chunk = f.read(byte_count)
            if len(chunk) < PARTICLE_SIZE:
                break
            if len(chunk) < byte_count:
                n_rows = len(chunk) // PARTICLE_SIZE
                chunk  = chunk[:n_rows * PARTICLE_SIZE]

            structured = np.frombuffer(chunk, dtype=PARTICLE_DTYPE).copy()
            data = np.column_stack([structured[col].astype(np.float64) for col in COLUMNS])
            del structured, chunk  # free intermediates immediately

            pages.append({"params": params, "data": data})

    return pages


# ── RF bucket physics ────────────────────────────────────────────────────────

# Particle rest masses in MeV
PARTICLE_MASSES = {
    "Electron": 0.51099895,
    "Proton":   938.27208816,
}

def compute_rf_separatrix_full(cavities, alphac, p_central, mass_mev,
                                f_rev_hz, t_central=0.0, n_points=600):
    """
    Compute RF bucket separatrix in (dt [s], delta=Dp/p [dimensionless]) coordinates.
    Uses standard longitudinal Hamiltonian (above transition).

    cavities:   list of (V_volts, h_int, phi_s_rad)
    alphac:     momentum compaction factor
    p_central:  reference beta*gamma
    mass_mev:   particle rest mass in MeV
    f_rev_hz:   revolution frequency in Hz
    """
    if not cavities:
        return None, None

    bg        = float(p_central)
    gamma     = np.sqrt(1.0 + bg**2)
    beta      = bg / gamma
    E0_eV     = mass_mev * 1e6 * gamma
    eta       = float(alphac) - 1.0 / gamma**2
    omega_rev = 2.0 * np.pi * f_rev_hz

    h1    = cavities[0][1]
    phi_s = cavities[0][2]   # synchronous phase of primary cavity [rad]
    V1    = cavities[0][0]

    # UFP for single/primary cavity: pi - phi_s
    phi_ufp = np.pi - phi_s

    # Standard separatrix F-function (single cavity, above transition):
    # delta^2 = (eV1 / (pi * h1 * |eta| * E0)) * F(phi)
    # F(phi)  = -cos(phi) - cos(phi_s) + (pi - phi - phi_s)*sin(phi_s)
    def F_single(phi):
        return (-np.cos(phi) - np.cos(phi_s)
                + (np.pi - phi - phi_s) * np.sin(phi_s))

    factor = V1 / (np.pi * h1 * abs(eta) * E0_eV)

    # For multi-cavity, add higher-harmonic correction to F
    if len(cavities) > 1:
        def F_multi(phi):
            val = F_single(phi)
            for V, h, phi_sv in cavities[1:]:
                ratio = h / h1
                # phi of cavity i as function of phi of cavity 1
                phi_i   = ratio * phi + (phi_sv - ratio * phi_s)
                # Standard contribution: same form as F_single
                # (Vi/V1) * [-cos(phi_i) - cos(phi_si) + (pi-phi_i-phi_si)*sin(phi_si)]
                val += (V / V1) * (
                    -np.cos(phi_i) - np.cos(phi_sv)
                    + (np.pi - phi_i - phi_sv) * np.sin(phi_sv)
                )
            return val
        F_func = F_multi
    else:
        F_func = F_single

    # Sweep phi across the full bucket: phi_ufp to 2*pi - phi_ufp
    # (for near-stationary bucket this is ~0 to ~2*pi)
    eps = 0.002
    phi_lo  = phi_ufp + eps
    phi_hi  = 2.0 * np.pi - phi_ufp - eps
    phi_arr = np.linspace(phi_lo, phi_hi, n_points)

    Fvals  = np.array([F_func(p) for p in phi_arr])
    delta2 = factor * Fvals
    mask   = delta2 >= 0

    if not mask.any():
        return None, None

    phi_valid   = phi_arr[mask]
    delta_valid = np.sqrt(np.maximum(delta2[mask], 0))

    # Convert phi to dt, centered on synchronous phase phi_s
    dt_valid = (phi_valid - phi_s) / (h1 * omega_rev)

    # Close the separatrix: upper half forward, lower half backward
    dt_sep    = np.concatenate([dt_valid, dt_valid[::-1]])
    delta_sep = np.concatenate([delta_valid, -delta_valid[::-1]])

    return dt_sep, delta_sep


# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
AX_BG     = "#12122a"
GRID_C    = "#2a2a50"
TEXT_C    = "#c8cde4"
SPINE_C   = "#333366"

# File accent colours — assigned in order as files are loaded
FILE_COLORS = [
    "#4f9ef0",  # blue
    "#f0904f",  # orange
    "#4ff0a0",  # green
    "#f04f90",  # pink
    "#c0a0f0",  # purple
    "#f0e04f",  # yellow
]

DEFAULT_PAIRS = [
    ("x",  "xp"),
    ("y",  "yp"),
    ("t",  "p"),
    ("x",  "y"),
]


# ── Plot drawing helper ────────────────────────────────────────────────────────

def _draw_on_axes(ax_main, xd, yd, mode, cmap, pts, alph, hbins, log_clr,
                  scatter_color, show_hist, ax_hx=None, ax_hy=None, hist_bins=60,
                  smooth_sigma=0.0, precomputed_h=None):
    from matplotlib.colors import LogNorm, Normalize

    if mode == "Scatter":
        ax_main.scatter(xd, yd, s=pts, c=scatter_color,
                        alpha=alph, linewidths=0, zorder=2)

    elif mode == "Heatmap 2D":
        from scipy.ndimage import gaussian_filter
        if precomputed_h is not None:
            h, xedges, yedges = precomputed_h
        else:
            h, xedges, yedges = np.histogram2d(xd, yd, bins=hbins)
            h = h.T
            if smooth_sigma > 0:
                h = gaussian_filter(h.astype(float), sigma=smooth_sigma)
        norm = (LogNorm(vmin=max(1e-3, h[h > 0].min()), vmax=h.max())
                if log_clr else Normalize(vmin=0, vmax=h.max()))
        _im = ax_main.imshow(h, origin="lower", aspect="auto",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap=cmap, norm=norm,
                       interpolation="gaussian", zorder=2)
        ax_main._last_im = _im  # store for colorbar

    if show_hist and ax_hx is not None and ax_hy is not None:
        ax_hx.hist(xd, bins=hist_bins, color=scatter_color, alpha=0.75, linewidth=0)
        ax_hx.tick_params(labelbottom=False, colors=TEXT_C, labelsize=8)
        ax_hy.hist(yd, bins=hist_bins, color=scatter_color, alpha=0.75,
                   orientation="horizontal", linewidth=0)
        ax_hy.tick_params(labelleft=False, colors=TEXT_C, labelsize=8)


# ── RF bucket drawing helper ─────────────────────────────────────────────────

def _draw_rf_bucket(ax, app, params, xn, yn, t_mean=0.0, p_mean=0.0):
    """Draw RF separatrix on ax. Handles static and ramp cases."""
    rf = app._rf_params
    if rf is None:
        return

    p_central = params.get("pCentral", None)
    t_central = params.get("PassCentralTime", 0.0)
    if p_central is None:
        return

    # Get cavities — static or interpolated from ramp
    cavities_raw = rf.get("cavities", [])  # list of (V, h, phi_s_deg)
    if app._rf_ramp_data is not None:
        # Nearest-neighbor interpolation in time
        ramp  = app._rf_ramp_data
        times = ramp["Time"]
        idx   = int(np.argmin(np.abs(times - float(t_central))))
        cavities_raw = ramp["cavities"][idx]

    if not cavities_raw:
        return

    # Convert phi_s from degrees to radians
    cavities = [(V, int(h), float(phi_s) * np.pi / 180.0)
                for V, h, phi_s in cavities_raw]

    mass_mev = rf.get("mass_mev", 0.51099895)
    alphac   = rf.get("alphac", 0.0)
    f_rev_hz = rf.get("f_rev_hz", 1e6)

    dt_sep, delta_sep = compute_rf_separatrix_full(
        cavities, alphac, p_central, mass_mev, f_rev_hz,
        t_central=float(t_central)
    )
    if dt_sep is None:
        return

    # p-axis: convert delta -> absolute or keep as delta
    # yn == "p" means y is momentum — we showed delta = (p-p0)/p0
    # xn == "t" means x is time — centered on t_central already
    dp_sep = delta_sep * float(p_central)
    if xn == "t" and yn == "p":
        ax.plot(dt_sep + t_mean, dp_sep + p_mean,
                color="#ffdd44", linewidth=1.4, linestyle="--",
                alpha=0.85, zorder=8, label="RF bucket")
    elif xn == "p" and yn == "t":
        ax.plot(dp_sep + p_mean, dt_sep + t_mean,
                color="#ffdd44", linewidth=1.4, linestyle="--",
                alpha=0.85, zorder=8, label="RF bucket")


# ── Analysis helpers ─────────────────────────────────────────────────────────

def _compute_twiss(xd, yd):
    """
    Compute RMS Twiss parameters from a conjugate pair (q, q').
    Returns dict with emittance, alpha, beta, gamma.
    Only meaningful when xd=position, yd=angle/momentum-deviation.
    """
    n   = len(xd)
    if n < 3:
        return None
    xc  = xd - xd.mean()
    yc  = yd - yd.mean()
    s11 = float(np.mean(xc**2))
    s12 = float(np.mean(xc * yc))
    s22 = float(np.mean(yc**2))
    det = s11 * s22 - s12**2
    if det <= 0:
        return None
    emit  = float(np.sqrt(det))
    beta  = s11 / emit
    alpha = -s12 / emit
    gamma = s22 / emit
    return {"emit": emit, "beta": beta, "alpha": alpha, "gamma": gamma}


# Scale factor for 95% containment in 2D Gaussian: sqrt(-2*ln(1-0.95))
_RMS95 = 2.4477

def _draw_overlay(ax, xd, yd, color, show_overlay, xn='x', yn='y', p_central=None):
    """Draw mean crosshairs, 1-sigma RMS ellipse, and 95% ellipse on ax."""
    if not show_overlay:
        return
    xc = float(xd.mean())
    yc = float(yd.mean())

    xs = float(xd.std()) or 1e-10
    ys = float(yd.std()) or 1e-10

    # Crosshairs
    ax.axvline(xc, color=color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=5)
    ax.axhline(yc, color=color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=5)

    # Covariance matrix
    xdev = xd - xd.mean()
    ydev = yd - yd.mean()
    cov  = np.array([[np.mean(xdev**2), np.mean(xdev*ydev)],
                     [np.mean(xdev*ydev), np.mean(ydev**2)]])
    vals, vecs = np.linalg.eigh(cov)
    vals  = np.maximum(vals, 0)
    theta = np.linspace(0, 2*np.pi, 200)
    unit  = np.array([np.cos(theta), np.sin(theta)])
    basis = vecs @ (np.sqrt(vals)[:, None] * unit)

    # 1-sigma RMS ellipse — bright, solid
    ax.plot(xc + basis[0], yc + basis[1],
            color=color, linewidth=1.2, alpha=0.9, zorder=6,
            label="1σ RMS")

    # 95% ellipse — same color but dimmer and dashed
    e95 = basis * _RMS95
    ax.plot(xc + e95[0], yc + e95[1],
            color=color, linewidth=1.0, alpha=0.45,
            linestyle="--", zorder=6, label="95%")

    # σ text box — top right, uses actual axis column names
    # For p axis show relative spread dp/p instead of absolute
    def _fmt_sigma(col, sigma):
        if col == 'p' and p_central is not None and float(p_central) != 0:
            rel = sigma / abs(float(p_central))
            return f"\u03c3_\u03b4 = {rel:.4g}"
        return f"\u03c3_{col} = {sigma:.4g}"
    sigma_txt = (
        _fmt_sigma(xn, xs) + "\n"
        + _fmt_sigma(yn, ys) + "\n"
        + "[RMS]"
    )
    ax.text(0.98, 0.98, sigma_txt,
            transform=ax.transAxes,
            va="top", ha="right",
            fontsize=10.5, family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a18",
                      edgecolor="#555577", alpha=0.75),
            zorder=10)


def _draw_twiss_box(ax, xd, yd, xn, yn, color):
    """Draw a Twiss parameter text box inside the axes if pair is conjugate."""
    CONJUGATE = {("x","xp"), ("xp","x"), ("y","yp"), ("yp","y")}
    if (xn, yn) not in CONJUGATE:
        return
    tw = _compute_twiss(xd, yd)
    if tw is None:
        return
    lines = (
        "emit = " + f"{tw['emit']:.3g}" + " m\n"
        + "beta = " + f"{tw['beta']:.3g}" + " m\n"
        + "alph = " + f"{tw['alpha']:.3g}" + "\n"
        + "gamm = " + f"{tw['gamma']:.3g}" + " /m"
    )
    ax.text(0.02, 0.98, lines,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10.5, family="monospace",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a18",
                      edgecolor=color, alpha=0.75),
            zorder=10)


# ── PlotPanel ─────────────────────────────────────────────────────────────────

class PlotPanel:
    """A single plot panel with its own file selector and X/Y axis dropdowns."""

    def __init__(self, parent, app, panel_index):
        self.app   = app
        self.index = panel_index
        self._current_color = FILE_COLORS[0]

        pair = DEFAULT_PAIRS[panel_index % len(DEFAULT_PAIRS)]

        # Outer frame — border color reflects selected file
        self.frame = ctk.CTkFrame(parent, corner_radius=8,
                                  border_width=2, border_color=FILE_COLORS[0])

        # ── Header row ────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self.frame, height=28, fg_color="#0a0a1a",
                           corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        MNU = dict(height=20, font=ctk.CTkFont(size=13), dynamic_resizing=False)

        # File selector — fixed width, truncates long names
        self.file_var = ctk.StringVar(value="—")
        self.file_menu = ctk.CTkOptionMenu(
            hdr, variable=self.file_var, values=["—"],
            command=self._on_file_change, width=100, **MNU)
        self.file_menu.pack(side="left", padx=(5, 1), pady=4)

        def _vsep():
            ctk.CTkFrame(hdr, width=1, fg_color="#252535").pack(
                side="left", fill="y", padx=3, pady=5)
        _vsep()

        # X axis
        ctk.CTkLabel(hdr, text="X", font=ctk.CTkFont(size=13),
                     text_color="#607080", width=12).pack(side="left", padx=(1,0))
        self.x_var = ctk.StringVar(value=pair[0])
        self.x_menu = ctk.CTkOptionMenu(
            hdr, variable=self.x_var, values=COLUMNS,
            command=lambda _: (self._update_ax_mode_options(), self.app._redraw()),
            width=64, **MNU)
        self.x_menu.pack(side="left", padx=1, pady=4)

        _vsep()

        # Y axis
        ctk.CTkLabel(hdr, text="Y", font=ctk.CTkFont(size=13),
                     text_color="#6e6a9a", width=12).pack(side="left", padx=(1,0))
        self.y_var = ctk.StringVar(value=pair[1])
        self.y_menu = ctk.CTkOptionMenu(
            hdr, variable=self.y_var, values=COLUMNS,
            command=lambda _: (self._update_ax_mode_options(), self.app._redraw()),
            width=64, **MNU)
        self.y_menu.pack(side="left", padx=1, pady=4)

        _vsep()

        # Axis mode — Roll/Track/Auto + optional Δref variants for p/t
        self.ax_mode_var = ctk.StringVar(value="Roll")
        self.ax_mode_menu = ctk.CTkOptionMenu(
            hdr, variable=self.ax_mode_var,
            values=["Auto", "Roll", "Track"],
            command=lambda _: (self._ax_history.clear(),
                               self._update_ax_mode_options(),
                               self.app._redraw()),
            width=78, **MNU)
        self.ax_mode_menu.pack(side="left", padx=(1, 2), pady=4)

        # RF bucket indicator — shown when separatrix is active
        self.rf_indicator = ctk.CTkLabel(
            hdr, text="RF", width=26, height=18,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#1a3a2a", fg_color="#1a3a2a",
            corner_radius=3)
        self.rf_indicator.pack(side="left", padx=(1, 2), pady=5)

        # Bucket view toggle
        self.bucket_view_var = ctk.BooleanVar(value=False)
        self.bucket_btn = ctk.CTkButton(
            hdr, text="Bkt", width=32, height=18,
            font=ctk.CTkFont(size=10),
            fg_color="#1a1a3a", hover_color="#2a2a5a",
            command=self._toggle_bucket_view)
        self.bucket_btn.pack(side="left", padx=(1, 4), pady=5)


        # ── Matplotlib figure ─────────────────────────────────────────────
        self.fig = plt.Figure(facecolor=BG, dpi=96)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)

        # Rolling axis limit history — deque of (xlo,xhi,ylo,yhi)
        self._ax_history  = deque()
        self._ax_cols     = None   # track which (xcol,ycol) the history is for
        # Heatmap cache: {(page_idx, hbins, sigma): (h, xedges, yedges)}
        self._hmap_cache  = {}
        self._hmap_key    = None   # (xcol, ycol, file_label)
        self.app.after(50, self._update_ax_mode_options)  # deferred so widgets are ready
        self._draw_empty()

    def _toggle_bucket_view(self):
        self.bucket_view_var.set(not self.bucket_view_var.get())
        if self.bucket_view_var.get():
            self.bucket_btn.configure(
                text="Bkt", fg_color="#2a4a8a", hover_color="#3a5a9a")
        else:
            self.bucket_btn.configure(
                text="Bkt", fg_color="#1a1a3a", hover_color="#2a2a5a")
        self.app._redraw()

    def _update_ax_mode_options(self):
        """Add/remove Δ variants in axis mode dropdown based on axis selection."""
        xn = self.x_var.get()
        yn = self.y_var.get()
        has_ref = xn in ("p", "t") or yn in ("p", "t")
        if has_ref:
            self.ax_mode_menu.configure(
                values=["Auto", "Roll", "Track", "Roll+\u0394", "Track+\u0394"])
        else:
            # If currently on a Δ mode, reset to base mode
            if "\u0394" in self.ax_mode_var.get():
                self.ax_mode_var.set("Roll")
            self.ax_mode_menu.configure(
                values=["Auto", "Roll", "Track"])

    def _on_file_change(self, choice):
        """Update border color when file selection changes."""
        finfo = self.app._file_by_label(choice)
        color = finfo["color"] if finfo else FILE_COLORS[0]
        self._current_color = color
        self.frame.configure(border_color=color)
        self._ax_history.clear()  # reset smoothing on file change
        self._hmap_cache.clear()
        self.app._redraw()

    def update_file_list(self, labels, default=None):
        """Refresh the file dropdown options."""
        if not labels:
            self.file_menu.configure(values=["—"])
            self.file_var.set("—")
            return
        self.file_menu.configure(values=labels)
        # If current selection no longer valid, pick default or first
        if self.file_var.get() not in labels:
            self.file_var.set(default if default in labels else labels[0])
        # Update border color
        finfo = self.app._file_by_label(self.file_var.get())
        color = finfo["color"] if finfo else FILE_COLORS[0]
        self._current_color = color
        self.frame.configure(border_color=color)

    def get_pages(self):
        """Return the pages list for the currently selected file, or None."""
        finfo = self.app._file_by_label(self.file_var.get())
        return finfo["pages"] if finfo else None

    def _style_ax(self, ax):
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT_C, labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_C)
        ax.grid(True, color=GRID_C, linewidth=0.4, zorder=0)

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_ax(ax)
        ax.text(0.5, 0.5, "Open an SDDS file to begin",
                transform=ax.transAxes, ha="center", va="center",
                color="#888888", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()

    def render(self, page_idx, mode, cmap, pts, alph, hbins, log_clr,
               show_hist, hist_bins, smooth_n=1):
        pages = self.get_pages()
        if not pages:
            self._draw_empty()
            return

        pg_idx = min(page_idx, len(pages) - 1)
        data   = pages[pg_idx]["data"]

        xn = self.x_var.get()
        yn = self.y_var.get()
        xi = COLUMNS.index(xn)
        yi = COLUMNS.index(yn)
        xd = data[:, xi].copy()
        yd = data[:, yi].copy()

        # ── Reference centering ───────────────────────────────────────────
        center_ref = "\u0394" in self.ax_mode_var.get()
        params     = pages[pg_idx]["params"]
        if center_ref:
            p_central = params.get("pCentral", None)
            t_central = params.get("PassCentralTime", None)
            if xn == "p" and p_central is not None:
                xd = xd - float(p_central)
            if yn == "p" and p_central is not None:
                yd = yd - float(p_central)
            if xn == "t" and t_central is not None:
                xd = xd - float(t_central)
            if yn == "t" and t_central is not None:
                yd = yd - float(t_central)

        xu = COL_UNITS.get(xn, "")
        yu = COL_UNITS.get(yn, "")
        # Add delta prefix to label when centered
        def _lbl(n, u, centered):
            prefix = "\u0394" if centered and n in ("p", "t") else ""
            return (prefix + n + "  [" + u + "]") if u else (prefix + n)
        xlbl = _lbl(xn, xu, center_ref)
        ylbl = _lbl(yn, yu, center_ref)

        # ── Axis limit calculation (mode-aware) ──────────────────────────
        col_key  = (xn, yn)
        ax_mode  = self.ax_mode_var.get()
        if col_key != self._ax_cols:
            self._ax_history.clear()
            self._ax_cols = col_key

        if ax_mode == "Auto":
            xlim = None
            ylim = None

        elif ax_mode == "Roll":
            # Smooth the window SIZE but always anchor to current centroid
            # so a ramping axis never loses the data off-screen
            xc = float(xd.mean()); yc = float(yd.mean())
            xhalf = float(xd.max() - xd.min()) / 2.0 or 1e-10
            yhalf = float(yd.max() - yd.min()) / 2.0 or 1e-10
            pad_x = xhalf * 0.05
            pad_y = yhalf * 0.05
            # Store half-widths in history, not absolute limits
            self._ax_history.append((xhalf + pad_x, yhalf + pad_y))
            while len(self._ax_history) > max(1, smooth_n):
                self._ax_history.popleft()
            arr   = list(self._ax_history)
            sxh   = sum(a[0] for a in arr) / len(arr)
            syh   = sum(a[1] for a in arr) / len(arr)
            xlim  = (xc - sxh, xc + sxh)
            ylim  = (yc - syh, yc + syh)

        else:  # Track — mean ± N*sigma, always centered on current frame
            nsig = self.app.sigma_var.get() if hasattr(self.app, 'sigma_var') else 3.0
            xc, xs = float(xd.mean()), float(xd.std()) or 1e-10
            yc, ys = float(yd.mean()), float(yd.std()) or 1e-10
            # Store sigma (spread) in history, smooth only the width
            self._ax_history.append((xs, ys))
            while len(self._ax_history) > max(1, smooth_n):
                self._ax_history.popleft()
            arr  = list(self._ax_history)
            sxs  = sum(a[0] for a in arr) / len(arr)
            sys_ = sum(a[1] for a in arr) / len(arr)
            # Center always = current frame mean — never lags behind the ramp
            xlim = (xc - nsig * sxs, xc + nsig * sxs)
            ylim = (yc - nsig * sys_, yc + nsig * sys_)

        # Detect layout changes that require full figure rebuild
        _heatmap_now = (mode == "Heatmap 2D")
        _heatmap_was = getattr(self, '_last_mode_was_heatmap', False)
        self._last_mode_was_heatmap = _heatmap_now
        need_rebuild = (
            not self.fig.axes or
            (show_hist and len(self.fig.axes) < 3) or
            (not show_hist and len(self.fig.axes) != 1) or
            (_heatmap_now != _heatmap_was)  # switching modes clears colorbar
        )
        if need_rebuild:
            self.fig.clear()
            self.fig.patch.set_facecolor(BG)
            if show_hist:
                gs = GridSpec(2, 2, figure=self.fig,
                              width_ratios=[4, 1], height_ratios=[1, 4],
                              hspace=0.03, wspace=0.03,
                              left=0.12, right=0.97, top=0.97, bottom=0.12)
                ax_s  = self.fig.add_subplot(gs[1, 0])
                ax_hx = self.fig.add_subplot(gs[0, 0], sharex=ax_s)
                ax_hy = self.fig.add_subplot(gs[1, 1], sharey=ax_s)
                for ax in (ax_s, ax_hx, ax_hy):
                    self._style_ax(ax)
                ax_main = ax_s
            else:
                ax_main = self.fig.add_subplot(111)
                self._style_ax(ax_main)
                self.fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.12)
                ax_hx = ax_hy = None
        else:
            # Reuse existing axes — just clear their contents
            if show_hist:
                ax_main = self.fig.axes[0]
                ax_hx   = self.fig.axes[1]
                ax_hy   = self.fig.axes[2]
                for ax in (ax_main, ax_hx, ax_hy):
                    ax.cla()
                    self._style_ax(ax)
            else:
                ax_main = self.fig.axes[0]
                ax_main.cla()
                self._style_ax(ax_main)
                ax_hx = ax_hy = None

        # Heatmap cache — recompute only when data/settings change
        precomputed_h = None
        if mode == "Heatmap 2D":
            smooth_sigma = self.app.smooth_sigma_var.get() if hasattr(self.app, 'smooth_sigma_var') else 0.0
            hmap_data_key = (xn, yn, self.file_var.get(), self.ax_mode_var.get())
            if hmap_data_key != self._hmap_key:
                self._hmap_cache.clear()
                self._hmap_key = hmap_data_key
            cache_key = (pg_idx, hbins, round(smooth_sigma, 2))
            if cache_key in self._hmap_cache:
                precomputed_h = self._hmap_cache[cache_key]
            else:
                from scipy.ndimage import gaussian_filter
                h, xe, ye = np.histogram2d(xd, yd, bins=hbins)
                h = h.T.astype(float)
                if smooth_sigma > 0:
                    h = gaussian_filter(h, sigma=smooth_sigma)
                precomputed_h = (h, xe, ye)
                # Limit cache size to 50 entries per panel
                if len(self._hmap_cache) > 50:
                    self._hmap_cache.pop(next(iter(self._hmap_cache)))
                self._hmap_cache[cache_key] = precomputed_h
        else:
            smooth_sigma = 0.0

        _draw_on_axes(ax_main, xd, yd, mode, cmap, pts, alph, hbins, log_clr,
                      self._current_color, show_hist, ax_hx, ax_hy, hist_bins,
                      smooth_sigma=smooth_sigma, precomputed_h=precomputed_h)

        # Bucket view: expand axes to show full separatrix
        bucket_view = self.bucket_view_var.get()
        _rf_active2 = (getattr(self.app, '_show_rf_bucket', False)
                       and center_ref and {xn, yn} == {'t', 'p'})
        if bucket_view and _rf_active2:
            # Compute separatrix extent for axis limits
            _dt_s, _dl_s = compute_rf_separatrix_full(
                [(float(vv), int(hh), float(pp) * np.pi / 180.0)
                 for vv, hh, pp in self.app._rf_params.get('cavities', [])],
                self.app._rf_params.get('alphac', 0),
                params.get('pCentral', 1),
                self.app._rf_params.get('mass_mev', 0.511),
                self.app._rf_params.get('f_rev_hz', 1e6)
            )
            if _dt_s is not None:
                _dp_s  = _dl_s * float(params.get('pCentral', 1))
                _tmean = float(xd.mean()) if xn == 't' else float(yd.mean())
                _pmean = float(yd.mean()) if yn == 'p' else float(xd.mean())
                pad_t  = abs(_dt_s).max() * 0.15
                pad_p  = abs(_dp_s).max() * 0.15
                ax_main.set_xlim(_tmean + _dt_s.min() - pad_t,
                                 _tmean + _dt_s.max() + pad_t)
                ax_main.set_ylim(_pmean + _dp_s.min() - pad_p,
                                 _pmean + _dp_s.max() + pad_p)
        else:
            if xlim is not None:
                ax_main.set_xlim(xlim)
            if ylim is not None:
                ax_main.set_ylim(ylim)

        # ── Overlay & Twiss ───────────────────────────────────────────────
        show_overlay = getattr(self.app, 'overlay_var', None)
        show_overlay = show_overlay.get() if show_overlay else False
        _draw_overlay(ax_main, xd, yd, self._current_color, show_overlay, xn, yn,
                      p_central=params.get('pCentral', None))
        if show_overlay:
            _draw_twiss_box(ax_main, xd, yd, xn, yn, self._current_color)

        # ── Particle tracking ─────────────────────────────────────────────
        if self.app.tracked_ids:
            pages = self.get_pages()
            if pages:
                fkey = self.file_var.get()
                if fkey not in self.app._traj_cache:
                    self.app._traj_cache[fkey] = {}
                tcolors = self.app._track_colors
                for t_idx, tid in enumerate(self.app.tracked_ids):
                    tcol = tcolors[t_idx % len(tcolors)]
                    # Build trajectory cache
                    if tid not in self.app._traj_cache[fkey]:
                        traj = []
                        for pg in pages:
                            pid_col = pg["data"][:, COLUMNS.index("particleID")]
                            mask = (pid_col.astype(int) == tid)
                            if mask.any():
                                idx = int(np.where(mask)[0][0])
                                traj.append(pg["data"][idx, :])
                            else:
                                traj.append(None)
                        self.app._traj_cache[fkey][tid] = traj
                    traj = self.app._traj_cache[fkey][tid]
                    # Draw trajectory line
                    tx, ty = [], []
                    for row in traj[:pg_idx + 1]:
                        if row is not None:
                            tx.append(float(row[xi]))
                            ty.append(float(row[yi]))
                    if len(tx) > 1:
                        ax_main.plot(tx, ty, color=tcol, linewidth=0.8,
                                     alpha=0.5, zorder=7)
                    # Draw current position dot
                    if traj[pg_idx] is not None:
                        row = traj[pg_idx]
                        ax_main.scatter([float(row[xi])], [float(row[yi])],
                                        s=60, color=tcol, zorder=8,
                                        linewidths=1.5,
                                        edgecolors=self._current_color)

        # ── Beam loss overlay ─────────────────────────────────────────────
        if getattr(self.app, '_show_loss', False):
            pages = self.get_pages()
            if pages:
                fkey  = self.file_var.get()
                lmap  = self.app._loss_cache.get(fkey, {})
                if lmap:
                    # Find particles lost on THIS page transition
                    # i.e. last seen on pg_idx
                    lost_rows = []
                    for pid, last_pg in lmap.items():
                        if last_pg == pg_idx:
                            pg_data = pages[pg_idx]["data"]
                            pid_col = pg_data[:, COLUMNS.index("particleID")]
                            mask = (pid_col.astype(int) == pid)
                            if mask.any():
                                lost_rows.append(pg_data[np.where(mask)[0][0]])
                    if lost_rows:
                        lx = np.array([r[xi] for r in lost_rows])
                        ly = np.array([r[yi] for r in lost_rows])
                        ax_main.scatter(lx, ly, s=12, color="#ff3333",
                                        alpha=0.8, linewidths=0,
                                        zorder=9, marker="x")
        # ── RF bucket separatrix ──────────────────────────────────────────
        _rf_show  = getattr(self.app, '_show_rf_bucket', False)
        _rf_axes  = {xn, yn} == {'t', 'p'}
        _rf_active = _rf_show and center_ref and _rf_axes
        # Update panel header indicator
        if _rf_active:
            self.rf_indicator.configure(
                text="RF", text_color="#44ff88", fg_color="#0a2a14")
            # Shift separatrix to beam centroid so it tracks the bunch
            _t_mean = float(xd.mean()) if xn == 't' else float(yd.mean())
            _p_mean = float(yd.mean()) if yn == 'p' else float(xd.mean())
            _draw_rf_bucket(ax_main, self.app, params, xn, yn,
                            t_mean=_t_mean, p_mean=_p_mean)
        else:
            self.rf_indicator.configure(
                text="RF", text_color="#1a3a2a", fg_color="#1a3a2a")

        ax_main.set_xlabel(xlbl, color=TEXT_C, fontsize=13)
        ax_main.set_ylabel(ylbl, color=TEXT_C, fontsize=13)

        # Colorbar for heatmap — only on fresh build to avoid overlap
        if mode == "Heatmap 2D" and hasattr(ax_main, '_last_im') and need_rebuild:
            try:
                cb = self.fig.colorbar(ax_main._last_im, ax=ax_main,
                                       fraction=0.035, pad=0.02)
                cb.ax.tick_params(colors=TEXT_C, labelsize=8)
                cb.outline.set_edgecolor(SPINE_C)
                cb.ax.yaxis.set_tick_params(color=TEXT_C)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_C)
            except Exception:
                pass

        self.canvas.draw_idle()

    def destroy(self):
        plt.close(self.fig)
        self.frame.destroy()


# ── Main Application ──────────────────────────────────────────────────────────

class SDDSViewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SDDS Bunch Distribution Viewer")
        self.geometry("1280x820")
        self.minsize(900, 600)

        # files: list of {label, path, color, pages}
        self._files       = []
        self.current_page = 0
        self._playing     = False
        self._play_thread = None
        self._panels      = []

        self.tracked_ids   = []     # list of tracked particleIDs
        self._traj_cache   = {}     # {file_label: {pid: list of rows}}
        self._stats_cache  = {}     # {file_label: stats_dict}
        self._stats_visible = False
        self._loss_cache   = {}     # {file_label: {pid: last_page_idx}}
        self._show_loss    = False  # whether to overlay lost particles
        self._selected_file_label = None  # file selected in legend for stats
        self._redraw_after_id     = None   # debounce timer id
        self._frame_rendering     = False  # throttle flag
        self._rf_params           = None   # static RF params dict
        self._rf_ramp_data        = None   # loaded ramp CSV data
        self._show_rf_bucket      = False  # RF bucket overlay toggle
        # Colors cycled per tracked particle
        self._track_colors = [
            "#ffffff", "#ffdd44", "#ff6644", "#44ffaa",
            "#ff44ff", "#44ddff", "#ffaa44", "#aaffaa",
        ]
        self._corr_window = None   # correlation matrix toplevel

        self._build_ui()
        self._add_panel()

    # ── Helpers ───────────────────────────────────────────────────────────────

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

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        toolbar = ctk.CTkFrame(self, height=46, corner_radius=0, fg_color="#0f0f1e")
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)

        # Standard toolbar button style
        BTN  = dict(height=28, corner_radius=6,
                    font=ctk.CTkFont(size=13))
        PAD  = dict(side="left", padx=2, pady=9)
        # Separator helper
        def tbsep():
            ctk.CTkFrame(toolbar, width=1, fg_color="#2a2a3a").pack(
                side="left", fill="y", padx=4, pady=8)

        ctk.CTkButton(toolbar, text="Open File", width=90,
                      **BTN, command=self._open_file).pack(side="left", padx=(10,2), pady=9)
        tbsep()
        ctk.CTkButton(toolbar, text="+ Panel", width=74,
                      **BTN, command=self._add_panel).pack(**PAD)
        ctk.CTkButton(toolbar, text="- Panel", width=74,
                      fg_color="#3a1a1a", hover_color="#552222",
                      **BTN, command=self._remove_panel).pack(**PAD)
        tbsep()
        ctk.CTkButton(toolbar, text="Export", width=70,
                      fg_color="#1a3a1a", hover_color="#225522",
                      **BTN, command=self._export).pack(**PAD)
        tbsep()
        ctk.CTkButton(toolbar, text="Save Session", width=100,
                      fg_color="#1a2a1a", hover_color="#223322",
                      **BTN, command=self._save_session).pack(**PAD)
        ctk.CTkButton(toolbar, text="Load Session", width=100,
                      fg_color="#1a1a2a", hover_color="#222233",
                      **BTN, command=self._load_session).pack(**PAD)
        tbsep()
        self.corr_btn = ctk.CTkButton(toolbar, text="Corr Matrix", width=94,
                      fg_color="#1a1a3a", hover_color="#222255",
                      **BTN, command=self._open_corr_matrix)
        self.corr_btn.pack(**PAD)

        self.stats_btn = ctk.CTkButton(toolbar, text="Stats", width=64,
                      fg_color="#2a1a3a", hover_color="#3a2a5a",
                      **BTN, command=self._toggle_stats_panel)
        self.stats_btn.pack(**PAD)

        self.loss_btn = ctk.CTkButton(toolbar, text="Beam Loss", width=84,
                      fg_color="#3a1a1a", hover_color="#552222",
                      **BTN, command=self._toggle_beam_loss)
        self.loss_btn.pack(**PAD)
        tbsep()
        self.rf_btn = ctk.CTkButton(toolbar, text="RF Bucket", width=84,
                      fg_color="#1a3a2a", hover_color="#224433",
                      **BTN, command=self._open_rf_dialog)
        self.rf_btn.pack(**PAD)

        # Loaded files legend — colored dots + short names
        self.legend_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        self.legend_frame.pack(side="left", padx=12, fill="y")

        # Body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        sidebar = ctk.CTkFrame(body, width=220, corner_radius=12)
        sidebar.pack(side="left", fill="y", padx=(0, 6))
        sidebar.pack_propagate(False)

        right = ctk.CTkFrame(body, corner_radius=12, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        # Info bar
        info = ctk.CTkFrame(right, height=28, fg_color="transparent")
        info.pack(fill="x", pady=(0, 4))
        info.pack_propagate(False)
        self.page_label = ctk.CTkLabel(info, text="Page — / —",
                                       font=ctk.CTkFont(size=13, weight="bold"),
                                       text_color="#a0c0e0")
        self.page_label.pack(side="left")
        ctk.CTkFrame(info, width=1, fg_color="#2a2a3a").pack(
            side="left", fill="y", padx=10, pady=4)
        self.param_label = ctk.CTkLabel(info, text="", text_color="#607080",
                                        font=ctk.CTkFont(size=13))
        self.param_label.pack(side="left")

        self.page_slider = ctk.CTkSlider(right, from_=0, to=1,
                                         command=self._on_slider, height=14)
        self.page_slider.pack(fill="x", pady=(0, 4))

        # Vertical paned window — grid on top, stats on bottom
        import tkinter as tk
        self._paned = tk.PanedWindow(right, orient=tk.VERTICAL,
                                     bg="#1a1a2e", sashwidth=5,
                                     sashrelief="flat", borderwidth=0)
        self._paned.pack(fill="both", expand=True)

        self.grid_frame = ctk.CTkFrame(self._paned, fg_color="transparent")
        self._paned.add(self.grid_frame, stretch="always")

        # Stats over time panel — added to paned window when toggled on
        self.stats_panel = ctk.CTkFrame(self._paned, fg_color="transparent")
        self.stats_fig   = plt.Figure(facecolor=BG)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=self.stats_panel)
        self.stats_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.stats_canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)

        self._build_sidebar(sidebar)

        # ── Keyboard shortcuts ────────────────────────────────────────────
        self.bind("<space>",      lambda e: self._toggle_play())
        self.bind("<Right>",      lambda e: self._step_page(+1))
        self.bind("<Left>",       lambda e: self._step_page(-1))
        self.bind("<Control-o>",  lambda e: self._open_file())

    def _build_sidebar(self, parent):
        def section(text):
            # Row: colored accent bar + label
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(9, 2))
            ctk.CTkFrame(row, width=3, height=14, corner_radius=2,
                         fg_color="#4a6fa5").pack(side="left", padx=(0, 6))
            ctk.CTkLabel(row, text=text,
                         font=ctk.CTkFont(size=13, weight="bold"),
                         text_color="#8ab0d8").pack(side="left", anchor="w")

        def lbl(text):
            ctk.CTkLabel(parent, text=text,
                         font=ctk.CTkFont(size=13),
                         text_color="gray60").pack(padx=14, pady=(3, 0), anchor="w")

        def slider(var, lo, hi, cmd):
            ctk.CTkSlider(parent, from_=lo, to=hi, variable=var,
                          command=cmd, width=186).pack(padx=14, pady=1)

        def sep():
            ctk.CTkFrame(parent, height=1, fg_color="#1e1e2e").pack(
                fill="x", padx=10, pady=3)

        section("DISPLAY")
        self.hist_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(parent, text="Marginal histograms",
                      variable=self.hist_var, command=self._redraw,
                      height=24).pack(padx=12, pady=(3, 1), anchor="w")

        self.overlay_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(parent, text="Stats overlay",
                      variable=self.overlay_var, command=self._redraw,
                      height=24).pack(padx=12, pady=(3,1), anchor="w")


        lbl("Point size")
        self.pt_size_var = ctk.DoubleVar(value=2.0)
        slider(self.pt_size_var, 0.5, 12, lambda _: self._redraw())

        lbl("Alpha")
        self.alpha_var = ctk.DoubleVar(value=0.35)
        slider(self.alpha_var, 0.02, 1.0, lambda _: self._redraw())

        lbl("Histogram bins")
        self.bins_var = ctk.IntVar(value=60)
        slider(self.bins_var, 10, 200, lambda _: self._redraw())

        lbl("Axis smoothing (frames)")
        self.smooth_var = ctk.IntVar(value=1)
        slider(self.smooth_var, 1, 30, lambda _: self._redraw())

        lbl("Track window (\u00b1 sigma)")
        self.sigma_var = ctk.DoubleVar(value=3.0)
        slider(self.sigma_var, 0.5, 10.0, lambda _: self._redraw())

        sep()

        section("PLOT MODE")
        self.plot_mode_var = ctk.StringVar(value="Scatter")
        for mode in ("Scatter", "Heatmap 2D"):
            ctk.CTkRadioButton(parent, text=mode, variable=self.plot_mode_var,
                               value=mode, command=self._redraw,
                               height=24).pack(padx=16, pady=1, anchor="w")

        lbl("Colormap")
        self.cmap_var = ctk.StringVar(value="turbo")
        ctk.CTkOptionMenu(parent, variable=self.cmap_var,
                          values=["turbo","plasma","inferno","gist_rainbow",
                                  "jet","RdYlBu","Spectral","gnuplot2",
                                  "CMRmap","afmhot"],
                          command=lambda _: self._redraw(),
                          width=188, height=28).pack(padx=12, pady=1)

        lbl("Heatmap bins")
        self.hmap_bins_var = ctk.IntVar(value=300)
        slider(self.hmap_bins_var, 20, 500, lambda _: self._redraw())

        lbl("Heatmap smoothing (sigma)")
        self.smooth_sigma_var = ctk.DoubleVar(value=2.0)
        slider(self.smooth_sigma_var, 0.0, 8.0, lambda _: self._redraw())

        self.log_scale_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(parent, text="Log color scale",
                      variable=self.log_scale_var, command=self._redraw,
                      height=24).pack(padx=12, pady=(3, 1), anchor="w")

        sep()

        section("PLAYBACK")
        pb = ctk.CTkFrame(parent, fg_color="transparent")
        pb.pack(padx=12, fill="x", pady=(2, 0))
        self.play_btn = ctk.CTkButton(pb, text="> Play", width=90, height=28,
                                      command=self._toggle_play)
        self.play_btn.pack(side="left")

        lbl("Speed (fps)")
        self.speed_var = ctk.DoubleVar(value=5)
        slider(self.speed_var, 1, 30, None)

        sep()

        section("PARTICLE TRACKING")
        lbl("Particle ID (manual)")
        self.track_id_var = ctk.StringVar(value="")
        id_row = ctk.CTkFrame(parent, fg_color="transparent")
        id_row.pack(padx=12, fill="x", pady=1)
        ctk.CTkEntry(id_row, textvariable=self.track_id_var,
                     width=120, height=26,
                     placeholder_text="e.g. 42,100,203").pack(side="left", padx=(0,4))
        self.track_btn = ctk.CTkButton(id_row, text="Track", width=58, height=26,
                      command=self._set_tracked_id)
        self.track_btn.pack(side="left")
        ctk.CTkButton(parent, text="Clear tracking", width=188, height=24,
                      fg_color="#3a1a1a", hover_color="#5a2a2a",
                      command=self._clear_tracking).pack(padx=12, pady=(2,1))
        self.track_label = ctk.CTkLabel(parent, text="No particle tracked",
                                        text_color="gray60",
                                        font=ctk.CTkFont(size=13))
        self.track_label.pack(padx=12, anchor="w")



    # ── File management ───────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open SDDS file",
            filetypes=[("All files", "*.*"), ("SDDS files", "*.sdds")],
        )
        if not path:
            return
        try:
            pages = read_sdds_file(path)
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc))
            return
        if not pages:
            messagebox.showwarning("Empty file", "No data pages found.")
            return

        # Assign color and short label
        color = FILE_COLORS[len(self._files) % len(FILE_COLORS)]
        name  = Path(path).name
        # Make label unique if same filename loaded twice
        existing = [f["label"] for f in self._files]
        label = name
        suffix = 2
        while label in existing:
            label = name + " (" + str(suffix) + ")"
            suffix += 1

        self._files.append({"label": label, "path": path,
                             "color": color, "pages": pages})
        # Pre-compute stats in background would be nice but sync is fine for now
        self._stats_cache.pop(label, None)
        self._loss_cache.pop(label, None)

        self._update_legend()
        self._update_panel_file_lists(default=label)

        # Slider uses max page count across all files
        n = self._max_pages()
        self.page_slider.configure(to=max(1, n - 1))
        self.page_slider.set(self.current_page)
        self._render_all()

    def _update_legend(self):
        """Rebuild the colored file legend in the toolbar — clickable."""
        for w in self.legend_frame.winfo_children():
            w.destroy()
        # Default selection to first file if none set
        if self._selected_file_label is None and self._files:
            self._selected_file_label = self._files[0]["label"]
        for finfo in self._files:
            is_selected = (finfo["label"] == self._selected_file_label)
            row = ctk.CTkFrame(self.legend_frame,
                               fg_color="#1e1e3a" if is_selected else "transparent",
                               corner_radius=6)
            row.pack(side="left", padx=3, pady=6)
            # Colored dot
            dot = ctk.CTkFrame(row, width=10, height=10,
                               corner_radius=5, fg_color=finfo["color"])
            dot.pack(side="left", padx=(6, 4))
            dot.pack_propagate(False)
            lbl_w = ctk.CTkLabel(row, text=finfo["label"],
                         font=ctk.CTkFont(size=13,
                             weight="bold" if is_selected else "normal"),
                         text_color=finfo["color"],
                         cursor="hand2")
            lbl_w.pack(side="left", padx=(0, 6))
            # Bind click to select this file
            label = finfo["label"]
            for widget in (row, dot, lbl_w):
                widget.bind("<Button-1>",
                    lambda e, l=label: self._select_stats_file(l))

    def _update_panel_file_lists(self, default=None):
        labels = self._file_labels()
        for panel in self._panels:
            panel.update_file_list(labels, default=default)

    # ── Panel management ──────────────────────────────────────────────────────

    def _add_panel(self):
        idx   = len(self._panels)
        panel = PlotPanel(self.grid_frame, self, idx)
        self._panels.append(panel)
        # Give it the current file list
        labels = self._file_labels()
        if labels:
            panel.update_file_list(labels, default=labels[0])
        self._reflow_grid()
        if self._files:
            self._render_all()

    def _remove_panel(self):
        if len(self._panels) <= 1:
            return
        panel = self._panels.pop()
        panel.destroy()
        self._reflow_grid()

    def _reflow_grid(self):
        n = len(self._panels)
        cols = 1 if n == 1 else (2 if n <= 4 else 3)
        rows = (n + cols - 1) // cols

        # Ungrid all panels first
        for panel in self._panels:
            panel.frame.grid_forget()

        # Reset ALL column/row configs to zero weight
        for i in range(10):
            self.grid_frame.columnconfigure(i, weight=0, minsize=0, uniform="")
            self.grid_frame.rowconfigure(i, weight=0, minsize=0, uniform="")

        # Use unique uniform group name per layout to avoid tkinter caching
        grp = "g" + str(n)
        for c in range(cols):
            self.grid_frame.columnconfigure(c, weight=1, uniform=grp)
        for r in range(rows):
            self.grid_frame.rowconfigure(r, weight=1, uniform=grp)

        for i, panel in enumerate(self._panels):
            r, c = divmod(i, cols)
            panel.frame.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)

        # Force geometry update
        self.grid_frame.update_idletasks()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_all(self):
        if not self._files:
            return

        mode      = self.plot_mode_var.get()
        cmap      = self.cmap_var.get()
        pts       = self.pt_size_var.get()
        alph      = self.alpha_var.get()
        hbins     = max(50, int(self.hmap_bins_var.get()))
        log_clr   = self.log_scale_var.get()
        show_hist = self.hist_var.get()
        hist_bins = max(10, int(self.bins_var.get()))

        smooth_n = max(1, int(self.smooth_var.get()))
        for panel in self._panels:
            panel.render(self.current_page, mode, cmap, pts, alph, hbins,
                         log_clr, show_hist, hist_bins, smooth_n)

        # Info bar — show info from first file
        n      = self._max_pages()
        finfo0 = self._files[0]
        pg_idx = min(self.current_page, len(finfo0["pages"]) - 1)
        params = finfo0["pages"][pg_idx]["params"]
        data   = finfo0["pages"][pg_idx]["data"]

        self.page_label.configure(text="Page " + str(self.current_page + 1) + " / " + str(n))
        step  = params.get("Step", "?")
        s_val = params.get("s")
        npart = data.shape[0]
        s_str = ("   s = " + f"{s_val:.4f}" + " m") if isinstance(s_val, float) else ""
        self.param_label.configure(
            text="Step " + str(step) + s_str + "   |   " + f"{npart:,}" + " particles")



    # ── Event handlers ────────────────────────────────────────────────────────

    def _export(self):
        """Save all panels to PNG and PDF."""
        if not self._files:
            messagebox.showwarning("Export", "No data loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export panels",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF file", "*.pdf"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        n   = len(self._panels)
        cols = 1 if n == 1 else (2 if n <= 4 else 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                  figsize=(cols * 5, rows * 4),
                                  facecolor=BG)
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for i, panel in enumerate(self._panels):
            src = panel.fig
            dst = axes_flat[i]
            dst.set_facecolor(AX_BG)
            # Re-render into export figure
            pages = panel.get_pages()
            if not pages:
                dst.text(0.5, 0.5, "No data", transform=dst.transAxes,
                         ha="center", va="center", color="#888888")
                continue
            pg_idx = min(self.current_page, len(pages)-1)
            data   = pages[pg_idx]["data"]
            xn = panel.x_var.get(); yn = panel.y_var.get()
            xi = COLUMNS.index(xn);  yi = COLUMNS.index(yn)
            xd = data[:, xi];        yd = data[:, yi]
            mode  = self.plot_mode_var.get()
            cmap  = self.cmap_var.get()
            pts   = self.pt_size_var.get()
            alph  = self.alpha_var.get()
            hbins = max(20, int(self.hmap_bins_var.get()))
            log_clr = self.log_scale_var.get()
            _draw_on_axes(dst, xd, yd, mode, cmap, pts, alph, hbins,
                          log_clr, panel._current_color, False,
                          smooth_sigma=self.smooth_sigma_var.get() if hasattr(self, 'smooth_sigma_var') else 0.0)
            show_overlay = self.overlay_var.get()
            _draw_overlay(dst, xd, yd, panel._current_color, show_overlay,
                          panel.x_var.get(), panel.y_var.get(),
                          p_central=pg.get('params', {}).get('pCentral', None))
            if show_overlay:
                _draw_twiss_box(dst, xd, yd, xn, yn, panel._current_color)
            xu = COL_UNITS.get(xn, ""); yu = COL_UNITS.get(yn, "")
            dst.set_xlabel((xn+" ["+xu+"]") if xu else xn, color=TEXT_C, fontsize=13)
            dst.set_ylabel((yn+" ["+yu+"]") if yu else yn, color=TEXT_C, fontsize=13)
            dst.tick_params(colors=TEXT_C, labelsize=10)
            for sp in dst.spines.values(): sp.set_edgecolor(SPINE_C)
            dst.grid(True, color=GRID_C, linewidth=0.4)
            dst.set_title(panel.file_var.get(), color=TEXT_C, fontsize=10, pad=3)
        # Hide unused axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.tight_layout(pad=1.0)
        base = path if not path.endswith((".png",".pdf")) else path[:-4]
        png_path = base + ".png"
        pdf_path = base + ".pdf"
        fig.savefig(png_path, dpi=150, facecolor=BG)
        fig.savefig(pdf_path, facecolor=BG)
        plt.close(fig)
        messagebox.showinfo("Export", "Saved:\n" + png_path + "\n" + pdf_path)

    def _set_tracked_id(self):
        """Parse comma-separated IDs and start tracking."""
        raw = self.track_id_var.get().strip()
        if not raw:
            return
        pids = []
        for tok in raw.replace(" ", "").split(","):
            try:
                pids.append(int(tok))
            except ValueError:
                messagebox.showerror("Tracking",
                    "'" + tok + "' is not a valid integer ID.")
                return
        self.tracked_ids = pids
        self._traj_cache.clear()
        tcolors = self._track_colors
        label_parts = []
        for i, pid in enumerate(pids):
            label_parts.append(str(pid))
        self.track_label.configure(
            text="Tracking: " + ", ".join(label_parts),
            text_color="#4f9ef0")
        self.track_btn.configure(text="* Track",
            fg_color="#1a3a5a", hover_color="#224466")
        self._render_all()

    def _clear_tracking(self):
        self.tracked_ids = []
        self._traj_cache.clear()
        self.track_id_var.set("")
        self.track_label.configure(text="No particle tracked", text_color="gray60")
        self.track_btn.configure(text="Track",
            fg_color=("#2b2b2b", "#3b3b3b"), hover_color=("#3b3b3b","#4b4b4b"))
        self._render_all()

    def _open_corr_matrix(self):
        """Open correlation matrix window for the first loaded file."""
        if not self._files:
            messagebox.showwarning("Corr Matrix", "No file loaded.")
            return
        # Toggle active indicator
        self.corr_btn.configure(text="* Corr Matrix", fg_color="#223366",
                                hover_color="#2a3a77")
        # Toggle button active state
        self.corr_btn.configure(text="* Corr Matrix",
                                fg_color="#224488", hover_color="#2a55aa")
        # Use first panel's file if available, else first file
        finfo = None
        if self._panels:
            finfo = self._file_by_label(self._panels[0].file_var.get())
        if finfo is None:
            finfo = self._files[0]
        pages = finfo["pages"]
        pg_idx = min(self.current_page, len(pages) - 1)
        data   = pages[pg_idx]["data"]
        color  = finfo["color"]

        # Columns to show (skip particleID)
        SHOW = ["x", "xp", "y", "yp", "t", "p"]
        nc   = len(SHOW)

        # Destroy old window if still open
        if self._corr_window and self._corr_window.winfo_exists():
            self._corr_window.destroy()

        win = ctk.CTkToplevel(self)
        win.title("Correlation Matrix  —  " + finfo["label"] +
                  "  page " + str(pg_idx + 1))
        win.geometry("900x860")
        win.configure(fg_color=BG)
        self._corr_window = win

        fig, axs = plt.subplots(nc, nc, figsize=(9, 8.5),
                                facecolor=BG)
        fig.subplots_adjust(hspace=0.08, wspace=0.08,
                            left=0.08, right=0.98,
                            top=0.96, bottom=0.06)

        for r in range(nc):
            for c in range(nc):
                ax  = axs[r, c]
                ax.set_facecolor(AX_BG)
                ax.tick_params(colors=TEXT_C, labelsize=7)
                for sp in ax.spines.values():
                    sp.set_edgecolor(SPINE_C)
                xn = SHOW[c]; yn = SHOW[r]
                xi = COLUMNS.index(xn); yi = COLUMNS.index(yn)
                xd = data[:, xi]; yd = data[:, yi]
                if r == c:
                    # Diagonal — 1D histogram
                    ax.hist(xd, bins=60, color=color, alpha=0.8, linewidth=0)
                else:
                    # Off-diagonal — scatter
                    ax.scatter(xd, yd, s=0.3, c=color, alpha=0.3,
                               linewidths=0)
                    # Correlation coefficient
                    r_val = float(np.corrcoef(xd, yd)[0, 1])
                    ax.text(0.97, 0.97, f"{r_val:.2f}",
                            transform=ax.transAxes,
                            ha="right", va="top", fontsize=7,
                            color=color,
                            bbox=dict(facecolor="#0a0a18", edgecolor="none",
                                      alpha=0.6, pad=1))
                # Axis labels on edges only
                if r == nc - 1:
                    ax.set_xlabel(xn, color=TEXT_C, fontsize=8, labelpad=2)
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0:
                    ax.set_ylabel(yn, color=TEXT_C, fontsize=8, labelpad=2)
                else:
                    ax.tick_params(labelleft=False)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        def _close_corr():
            plt.close(fig)
            win.destroy()
            try:
                self.corr_btn.configure(text="Corr Matrix",
                                        fg_color="#1a1a3a", hover_color="#222255")
            except Exception:
                pass
        win.protocol("WM_DELETE_WINDOW", _close_corr)

    def _compute_stats_cache(self, finfo):
        """Compute per-page statistics for all columns. Cached per file."""
        label = finfo["label"]
        if label in self._stats_cache:
            return self._stats_cache[label]
        pages = finfo["pages"]
        STAT_COLS = ["x", "xp", "y", "yp", "t", "p", "dt"]
        CONJ      = [("x","xp"), ("y","yp")]
        n_pages   = len(pages)
        stats = {col: {"mean": [], "sigma": [], "mn": [], "mx": []}
                 for col in STAT_COLS}
        emit  = {pair: [] for pair in CONJ}
        for pg in pages:
            data = pg["data"]
            for col in STAT_COLS:
                ci  = COLUMNS.index(col)
                col_data = data[:, ci]
                stats[col]["mean"].append(float(col_data.mean()))
                stats[col]["sigma"].append(float(col_data.std()))
                stats[col]["mn"].append(float(col_data.min()))
                stats[col]["mx"].append(float(col_data.max()))
            for (cn, cm) in CONJ:
                xd = data[:, COLUMNS.index(cn)] - data[:, COLUMNS.index(cn)].mean()
                yd = data[:, COLUMNS.index(cm)] - data[:, COLUMNS.index(cm)].mean()
                s11 = float(np.mean(xd**2))
                s12 = float(np.mean(xd*yd))
                s22 = float(np.mean(yd**2))
                det = s11*s22 - s12**2
                emit[(cn,cm)].append(float(np.sqrt(max(det, 0))))
        # Convert to arrays
        for col in STAT_COLS:
            for k in stats[col]:
                stats[col][k] = np.array(stats[col][k])
        for pair in CONJ:
            emit[pair] = np.array(emit[pair])
        result = {"stats": stats, "emit": emit, "n_pages": n_pages}
        self._stats_cache[label] = result
        return result

    def _toggle_stats_panel(self):
        self._stats_visible = not self._stats_visible
        if self._stats_visible:
            # Add stats panel to paned window at ~260px
            self._paned.add(self.stats_panel, minsize=180, stretch="never")
            self._paned.paneconfigure(self.stats_panel, height=260)
            self._draw_stats_panel()
            self.stats_btn.configure(
                text="* Stats", fg_color="#5a3a8a", hover_color="#6a4a9a")
        else:
            self._paned.remove(self.stats_panel)
            self.stats_btn.configure(
                text="Stats", fg_color="#2a1a3a", hover_color="#3a2a5a")

    def _draw_stats_panel(self):
        if not self._files or not self._stats_visible:
            return
        self.stats_fig.clear()
        self.stats_fig.patch.set_facecolor(BG)

        STAT_COLS = ["x", "xp", "y", "yp", "t", "p", "dt"]
        CONJ      = [("x","xp"), ("y","yp")]
        # 7 columns + 2 emittance = 9 subplots in 3 rows x 3 cols
        n_plots = len(STAT_COLS) + len(CONJ)
        cols_g  = 3
        rows_g  = (n_plots + cols_g - 1) // cols_g
        gs = self.stats_fig.add_gridspec(rows_g, cols_g,
                                         hspace=0.55, wspace=0.35,
                                         left=0.06, right=0.98,
                                         top=0.93, bottom=0.12)
        pages_x = np.arange(1, self._max_pages() + 1)

        plot_idx = 0
        # Build subplot axes once, then overlay all files
        axs = []
        for i in range(n_plots):
            r, c = divmod(i, cols_g)
            axs.append(self.stats_fig.add_subplot(gs[r, c]))
            axs[-1].set_facecolor(AX_BG)
            axs[-1].tick_params(colors=TEXT_C, labelsize=7)
            for sp in axs[-1].spines.values():
                sp.set_edgecolor(SPINE_C)
            axs[-1].grid(True, color=GRID_C, linewidth=0.3)

        # Set titles
        for i, col in enumerate(STAT_COLS):
            unit = COL_UNITS.get(col, "")
            axs[i].set_title(col + (" ["+unit+"]" if unit else ""),
                             color=TEXT_C, fontsize=8, pad=2)
        for j, pair in enumerate(CONJ):
            axs[len(STAT_COLS)+j].set_title(
                "emit " + pair[0] + "/" + pair[1] + " [m·rad]",
                color=TEXT_C, fontsize=8, pad=2)

        # Show only selected file, or all if none selected
        files_to_show = self._files
        if self._selected_file_label:
            sel = self._file_by_label(self._selected_file_label)
            if sel:
                files_to_show = [sel]
        for finfo in files_to_show:
            sc   = self._compute_stats_cache(finfo)
            clr  = finfo["color"]
            n    = sc["n_pages"]
            px   = np.arange(1, n + 1)

            for i, col in enumerate(STAT_COLS):
                ax = axs[i]
                s  = sc["stats"][col]
                ax.plot(px, s["mean"], color=clr, linewidth=1.0)
                ax.fill_between(px,
                                s["mean"] - s["sigma"],
                                s["mean"] + s["sigma"],
                                color=clr, alpha=0.15)
                ax.plot(px, s["mn"], color=clr, linewidth=0.5,
                        linestyle=":", alpha=0.5)
                ax.plot(px, s["mx"], color=clr, linewidth=0.5,
                        linestyle=":", alpha=0.5)

            for j, pair in enumerate(CONJ):
                ax = axs[len(STAT_COLS)+j]
                ax.plot(px, sc["emit"][pair], color=clr, linewidth=1.0)

        # Current page vline on all subplots
        for ax in axs:
            ax.axvline(self.current_page + 1, color="#ffffff",
                       linewidth=0.7, alpha=0.5, zorder=5)
            ax.set_xlabel("page", color=TEXT_C, fontsize=7, labelpad=1)

        self.stats_canvas.draw_idle()

    def _compute_loss_cache(self, finfo):
        """
        Compute beam loss map for a file.
        Returns {particleID: last_page_index_seen}.
        Only uses the particleID column — lightweight.
        """
        label = finfo["label"]
        if label in self._loss_cache:
            return self._loss_cache[label]
        pages = finfo["pages"]
        pid_idx = COLUMNS.index("particleID")
        # Build set of IDs per page (only particleID column)
        id_sets = []
        for pg in pages:
            ids = set(pg["data"][:, pid_idx].astype(int).tolist())
            id_sets.append(ids)
        # Find last page each particle was seen
        loss_map = {}
        all_ids = id_sets[0]
        for pg_idx in range(len(pages) - 1):
            lost = id_sets[pg_idx] - id_sets[pg_idx + 1]
            for pid in lost:
                loss_map[pid] = pg_idx
        self._loss_cache[label] = loss_map
        return loss_map

    def _toggle_beam_loss(self):
        """Toggle beam loss overlay, with size warning for large datasets."""
        if not self._files:
            messagebox.showwarning("Beam Loss", "No file loaded.")
            return

        if self._show_loss:
            # Turn off
            self._show_loss = False
            self.loss_btn.configure(
                text="Beam Loss", fg_color="#3a1a1a", hover_color="#5a2222")
            self._render_all()
            return

        # Estimate cost and warn if large
        total = sum(
            len(f["pages"]) * f["pages"][0]["data"].shape[0]
            for f in self._files
        )
        if total > 5_000_000:
            answer = messagebox.askyesno(
                "Beam Loss — Large Dataset",
                "Estimated " + f"{total:,}" + " particle-pages to scan.\n"
                "This may take a moment and use significant memory.\n\n"
                "Continue?"
            )
            if not answer:
                return

        # Compute for all loaded files
        for finfo in self._files:
            self._compute_loss_cache(finfo)

        # Report total losses
        total_lost = sum(len(self._loss_cache.get(f["label"], {}))
                         for f in self._files)
        self._show_loss = True
        self.loss_btn.configure(
            text="* Beam Loss", fg_color="#aa2222", hover_color="#cc3333")
        self._render_all()
        messagebox.showinfo(
            "Beam Loss",
            "Loss tracking active.\n"
            "Total lost particles: " + f"{total_lost:,}" + "\n"
            "Lost particles shown as red X on their last seen page."
        )

    def _select_stats_file(self, label):
        """Set the active file for stats panel display."""
        self._selected_file_label = label
        self._update_legend()
        if self._stats_visible:
            self._draw_stats_panel()

    def _open_rf_dialog(self):
        """Open RF bucket configuration dialog, pre-filled if params exist."""
        # Load existing params for pre-fill
        ex = self._rf_params or {}
        ex_cavs = ex.get('cavities', [(1e6, 1, 0.0)])
        ex_mass = ex.get('mass_mev', 0.51099895)
        ex_ac   = ex.get('alphac', 0.0)
        ex_frev = ex.get('f_rev_hz', 1e6) / 1e6
        ex_mode = ex.get('mode', 'Static')
        # Determine species from mass
        ex_species = 'Custom'
        for sp, m in PARTICLE_MASSES.items():
            if abs(m - ex_mass) < 0.001:
                ex_species = sp
                break

        dlg = ctk.CTkToplevel(self)
        dlg.title("RF Bucket Configuration" + (" — Edit" if ex else ""))
        dlg.geometry("480x580")
        dlg.lift()
        dlg.focus_force()
        dlg.update()
        try:
            dlg.grab_set()
        except Exception:
            pass

        pad = dict(padx=12, pady=4)

        ctk.CTkLabel(dlg, text="PARTICLE & LATTICE",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#8ab0d8").pack(anchor="w", **pad)

        row1 = ctk.CTkFrame(dlg, fg_color="transparent")
        row1.pack(fill="x", **pad)
        ctk.CTkLabel(row1, text="Species:", width=80).pack(side="left")
        species_var = ctk.StringVar(value=ex_species)
        ctk.CTkOptionMenu(row1, variable=species_var,
                          values=["Electron", "Proton", "Custom"],
                          width=120).pack(side="left", padx=4)
        ctk.CTkLabel(row1, text="Mass (MeV):", width=90).pack(side="left", padx=(8,0))
        mass_var = ctk.StringVar(value=str(ex_mass))
        ctk.CTkEntry(row1, textvariable=mass_var, width=80).pack(side="left", padx=4)

        def _on_species(*_):
            s = species_var.get()
            if s in PARTICLE_MASSES:
                mass_var.set(str(PARTICLE_MASSES[s]))
        species_var.trace_add("write", _on_species)

        row2 = ctk.CTkFrame(dlg, fg_color="transparent")
        row2.pack(fill="x", **pad)
        ctk.CTkLabel(row2, text="alphac:", width=80).pack(side="left")
        alphac_var = ctk.StringVar(value=str(ex_ac))
        ctk.CTkEntry(row2, textvariable=alphac_var, width=100).pack(side="left", padx=4)
        ctk.CTkLabel(row2, text="f_rev (MHz):", width=90).pack(side="left", padx=(8,0))
        frev_var = ctk.StringVar(value=str(round(ex_frev, 6)))
        ctk.CTkEntry(row2, textvariable=frev_var, width=80).pack(side="left", padx=4)

        ctk.CTkFrame(dlg, height=1, fg_color="#1e1e3a").pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(dlg, text="RF MODE",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#8ab0d8").pack(anchor="w", **pad)
        rf_mode_var = ctk.StringVar(value=ex_mode)
        mode_row = ctk.CTkFrame(dlg, fg_color="transparent")
        mode_row.pack(fill="x", **pad)
        ctk.CTkRadioButton(mode_row, text="Static", variable=rf_mode_var,
                           value="Static").pack(side="left", padx=8)
        ctk.CTkRadioButton(mode_row, text="Ramp (CSV)", variable=rf_mode_var,
                           value="Ramp").pack(side="left", padx=8)

        ctk.CTkFrame(dlg, height=1, fg_color="#1e1e3a").pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(dlg, text="CAVITIES  (V in Volts, phi_s in degrees)",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#8ab0d8").pack(anchor="w", **pad)
        hdr = ctk.CTkFrame(dlg, fg_color="transparent")
        hdr.pack(fill="x", padx=12)
        for txt, w in [("#", 30), ("Voltage (V)", 110), ("Harmonic h", 90), ("phi_s (deg)", 100)]:
            ctk.CTkLabel(hdr, text=txt, width=w,
                         font=ctk.CTkFont(size=13), text_color="gray60").pack(side="left")

        cavity_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        cavity_frame.pack(fill="x", padx=12)
        cavity_rows = []

        def _add_cavity_row(V="1e6", h="1", phi="0"):
            row = ctk.CTkFrame(cavity_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=str(len(cavity_rows)+1), width=30,
                         font=ctk.CTkFont(size=13)).pack(side="left")
            vv = ctk.StringVar(value=V)
            hv = ctk.StringVar(value=h)
            pv = ctk.StringVar(value=phi)
            ctk.CTkEntry(row, textvariable=vv, width=110).pack(side="left", padx=2)
            ctk.CTkEntry(row, textvariable=hv, width=90).pack(side="left", padx=2)
            ctk.CTkEntry(row, textvariable=pv, width=100).pack(side="left", padx=2)
            cavity_rows.append((vv, hv, pv))

        # Pre-fill cavities from existing params
        for V, h, phi in ex_cavs:
            _add_cavity_row(str(V), str(h), str(phi))

        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(btn_row, text="+ Cavity", width=90, height=26,
                      command=_add_cavity_row).pack(side="left")

        ctk.CTkFrame(dlg, height=1, fg_color="#1e1e3a").pack(fill="x", padx=12, pady=6)
        ramp_row = ctk.CTkFrame(dlg, fg_color="transparent")
        ramp_row.pack(fill="x", **pad)
        ramp_path_var = ctk.StringVar(value="No ramp file loaded")
        ctk.CTkLabel(ramp_row, textvariable=ramp_path_var,
                     text_color="gray60",
                     font=ctk.CTkFont(size=13)).pack(side="left", padx=(0,8))

        def _load_ramp():
            path = filedialog.askopenfilename(
                title="Load RF ramp CSV",
                filetypes=[("CSV/text files", "*.csv *.txt *.dat"), ("All files", "*.*")])
            if not path:
                return
            try:
                # Auto-detect delimiter: comma, tab, or whitespace
                with open(path) as f:
                    first_line = f.readline().strip()
                if ',' in first_line:
                    delim = ','
                elif '\t' in first_line:
                    delim = '\t'
                else:
                    delim = None  # whitespace

                data = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if delim:
                            vals = [v.strip() for v in line.split(delim)]
                        else:
                            vals = line.split()
                        data.append([float(v) for v in vals])

                if not data:
                    raise ValueError("No data rows found")
                arr   = np.array(data)
                times = arr[:, 0]
                n_cols = arr.shape[1] - 1
                n_cav  = n_cols // 3
                if n_cav < 1:
                    raise ValueError("Expected at least 4 columns: Time, V, h, phi")
                cav_data = []
                for row in arr:
                    cavs = []
                    for i in range(n_cav):
                        base = 1 + i * 3
                        cavs.append((float(row[base]),
                                     int(row[base+1]),
                                     float(row[base+2])))
                    cav_data.append(cavs)
                self._rf_ramp_data = {"Time": times, "cavities": cav_data}
                ramp_path_var.set(Path(path).name + " (" + str(len(data)) +
                                  " steps, " + str(n_cav) + " cav" +
                                  ("ity)" if n_cav == 1 else "ities)"))
            except Exception as e:
                messagebox.showerror("Ramp load error", str(e))

        ctk.CTkButton(ramp_row, text="Load CSV", width=90, height=26,
                      command=_load_ramp).pack(side="left")

        ctk.CTkFrame(dlg, height=1, fg_color="#1e1e3a").pack(fill="x", padx=12, pady=8)
        bot = ctk.CTkFrame(dlg, fg_color="transparent")
        bot.pack(fill="x", padx=12, pady=4)

        def _apply():
            try:
                mass = float(mass_var.get())
                ac   = float(alphac_var.get())
                frev = float(frev_var.get()) * 1e6
                cavs = [(float(vv.get()), int(float(hv.get())), float(pv.get()))
                        for vv, hv, pv in cavity_rows]
                self._rf_params = {"mass_mev": mass, "alphac": ac,
                                   "f_rev_hz": frev, "cavities": cavs,
                                   "mode": rf_mode_var.get()}

                self._show_rf_bucket = True
                self.rf_btn.configure(text="* RF Bucket",
                                      fg_color="#226633", hover_color="#338844")
                self._render_all()
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("RF config error", str(e))

        def _clear():
            self._rf_params      = None
            self._rf_ramp_data   = None
            self._show_rf_bucket = False
            self.rf_btn.configure(text="RF Bucket",
                                  fg_color="#1a3a2a", hover_color="#224433")
            self._render_all()
            dlg.destroy()

        ctk.CTkButton(bot, text="Apply", width=100, height=30,
                      fg_color="#1a4a2a", hover_color="#226633",
                      command=_apply).pack(side="left", padx=4)
        ctk.CTkButton(bot, text="Clear & Close", width=110, height=30,
                      fg_color="#3a1a1a", hover_color="#552222",
                      command=_clear).pack(side="left", padx=4)
        ctk.CTkButton(bot, text="Cancel", width=80, height=30,
                      command=dlg.destroy).pack(side="left", padx=4)

    def _save_session(self):
        """Save current session to a JSON file."""
        import json
        path = filedialog.asksaveasfilename(
            title="Save Session",
            defaultextension=".json",
            filetypes=[("Session files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            session = {
                "files": [{"path": f["path"], "label": f["label"]}
                          for f in self._files],
                "current_page": self.current_page,
                "panels": [
                    {
                        "file_label": p.file_var.get(),
                        "x": p.x_var.get(),
                        "y": p.y_var.get(),
                        "ax_mode": p.ax_mode_var.get(),
                        "bucket_view": p.bucket_view_var.get(),
                    }
                    for p in self._panels
                ],
                "plot_mode":    self.plot_mode_var.get(),
                "cmap":         self.cmap_var.get(),
                "hmap_bins":    self.hmap_bins_var.get(),
                "smooth_sigma": self.smooth_sigma_var.get(),
                "log_scale":    self.log_scale_var.get(),
                "show_hist":    self.hist_var.get(),
                "hist_bins":    self.bins_var.get(),
                "pt_size":      self.pt_size_var.get(),
                "alpha":        self.alpha_var.get(),
                "smooth_n":     self.smooth_var.get(),
                "sigma":        self.sigma_var.get(),
                "overlay":      self.overlay_var.get(),
                "center_ref":   False,
                "rf_params":    self._rf_params,
                "rf_ramp_path": None,
                "show_rf":      self._show_rf_bucket,
            }
            with open(path, "w") as f:
                json.dump(session, f, indent=2)
            messagebox.showinfo("Session saved",
                                f"Session saved to {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _load_session(self):
        """Restore a previously saved session."""
        import json
        path = filedialog.askopenfilename(
            title="Load Session",
            filetypes=[("Session files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                session = json.load(f)

            # Load files
            for fentry in session.get("files", []):
                fpath = fentry["path"]
                if not Path(fpath).exists():
                    messagebox.showwarning("File missing",
                        f"{fpath} not found — skipping.")
                    continue
                try:
                    pages = read_sdds_file(fpath)
                except Exception as e:
                    messagebox.showerror("Load error",
                        "Could not load " + str(fpath) + ":\n" + str(e))
                    continue
                color = FILE_COLORS[len(self._files) % len(FILE_COLORS)]
                label = Path(fpath).stem[:18]
                self._files.append({
                    "label": label, "path": fpath,
                    "color": color, "pages": pages})
                self._traj_cache[label]  = {}
                self._stats_cache[label] = None

            if not self._files:
                return

            # Restore sidebar settings
            self.plot_mode_var.set(session.get("plot_mode", "Scatter"))
            self.cmap_var.set(session.get("cmap", "turbo"))
            self.hmap_bins_var.set(session.get("hmap_bins", 300))
            self.smooth_sigma_var.set(session.get("smooth_sigma", 2.0))
            self.log_scale_var.set(session.get("log_scale", True))
            self.hist_var.set(session.get("show_hist", False))
            self.bins_var.set(session.get("hist_bins", 60))
            self.pt_size_var.set(session.get("pt_size", 2.0))
            self.alpha_var.set(session.get("alpha", 0.35))
            self.smooth_var.set(session.get("smooth_n", 1))
            self.sigma_var.set(session.get("sigma", 3.0))
            self.overlay_var.set(session.get("overlay", False))

            # Restore RF params
            self._rf_params      = session.get("rf_params", None)
            self._show_rf_bucket = session.get("show_rf", False)
            if self._rf_params:
                self.rf_btn.configure(text="* RF Bucket",
                                      fg_color="#226633", hover_color="#338844")

            # Restore panels
            saved_panels = session.get("panels", [])
            labels = self._file_labels()

            # Add/remove panels to match saved count
            while len(self._panels) < len(saved_panels):
                self._add_panel()
            while len(self._panels) > len(saved_panels) and len(self._panels) > 1:
                p = self._panels.pop()
                p.destroy()

            for i, (panel, pdata) in enumerate(
                    zip(self._panels, saved_panels)):
                panel.update_file_list(labels,
                    default=pdata.get("file_label", labels[0]))
                panel.x_var.set(pdata.get("x", "t"))
                panel.y_var.set(pdata.get("y", "p"))
                panel.ax_mode_var.set(pdata.get("ax_mode", "Roll"))
                panel.bucket_view_var.set(pdata.get("bucket_view", False))
                panel._update_ax_mode_options()

            self._reflow_grid()
            self._update_legend()
            self._update_panel_file_lists()

            # Restore page
            n = self._max_pages()
            self.current_page = min(session.get("current_page", 0), n - 1)
            self.page_slider.configure(to=max(1, n - 1))
            self.page_slider.set(self.current_page)
            self._render_all()

        except Exception as e:
            messagebox.showerror("Load session error", str(e))

    def _step_page(self, delta):
        """Step forward or backward by one page."""
        if not self._files:
            return
        n = self._max_pages()
        self.current_page = max(0, min(self.current_page + delta, n - 1))
        self.page_slider.set(self.current_page)
        self._render_all()

    def _redraw(self, *_):
        """Debounced redraw — waits 120ms after last call before rendering."""
        if hasattr(self, '_redraw_after_id') and self._redraw_after_id:
            try:
                self.after_cancel(self._redraw_after_id)
            except Exception:
                pass
        self._redraw_after_id = self.after(120, self._render_all)

    def _redraw_immediate(self, *_):
        """Immediate redraw — for page navigation where responsiveness matters."""
        self._render_all()

    def _reset_axis_histories(self):
        for panel in self._panels:
            panel._ax_history.clear()

    def _on_slider(self, val):
        if not self._files:
            return
        self.current_page = int(round(float(val)))
        self._redraw_immediate()
        if self._stats_visible:
            self._draw_stats_panel()

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self.play_btn.configure(text="> Play")
        else:
            if not self._files:
                return
            self._playing = True
            self.play_btn.configure(text="|| Pause")
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()

    def _play_loop(self):
        self._frame_rendering = False
        while self._playing:
            fps   = max(1.0, float(self.speed_var.get()))
            delay = 1.0 / fps
            n     = self._max_pages()
            if n == 0:
                break
            # Throttle: skip frame if previous render not done
            if not self._frame_rendering:
                next_page = (self.current_page + 1) % n
                self.current_page = next_page
                self._frame_rendering = True
                self.after(0, self._advance_frame, next_page)
            time.sleep(delay)
        self.after(0, self._on_play_stopped)

    def _advance_frame(self, page):
        """Called on main thread by play loop — safe to touch widgets."""
        if not self._playing:
            self._frame_rendering = False
            return
        try:
            self.page_slider.set(page)
            self._render_all()
            if self._stats_visible:
                self._draw_stats_panel()
        except Exception:
            pass  # widget may be destroyed during shutdown
        finally:
            self._frame_rendering = False

    def _on_play_stopped(self):
        """Called on main thread when play loop exits."""
        try:
            self.play_btn.configure(text="> Play")
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import gc
    app = SDDSViewer()

    def _on_close():
        app._playing = False
        if app._play_thread and app._play_thread.is_alive():
            app._play_thread.join(timeout=1.0)
        for p in app._panels:
            plt.close(p.fig)
        app._files = []
        gc.collect()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", _on_close)
    app.mainloop()
    gc.collect()
