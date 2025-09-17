# app.py (delegates tab builders to /tabs)
# spectro_control_gui.py
# ---------------------------------------------------------------
# A complete tkinter GUI for Avantes spectrometer + lasers + relay
# Live view, automated measurement (with Auto-IT), analysis, export
# ---------------------------------------------------------------

import os
import sys
import time
import json
import queue
import threading
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# matplotlib embed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# serial
import serial
import serial.tools.list_ports

# analysis helpers
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Avantes wrapper (must be present alongside this file)
from avantes_spectrometer import Avantes_Spectrometer


# =========================
# ---- Configuration ------
# =========================

APP_TITLE = "Spectrometer Control & Analysis"
SETTINGS_FILE = "spectro_gui_settings.json"

# Default COM ports (editable in Setup tab)
DEFAULT_COM_PORTS = {
    "OBIS": "COM10",   # 405/445/488 on OBIS
    "RELAY": "COM11",  # 517/532/Hg-Ar on relay board
    "CUBE": "COM1"     # 377 nm cube
}

# OBIS channel map (edit as needed)
OBIS_LASER_MAP = {
    "405": 5,
    "445": 4,
    "488": 3
}

# Default powers (Watts for OBIS/CUBE setpoints, you can interpret as needed)
DEFAULT_LASER_POWERS = {
    "405": 0.005,
    "445": 0.003,
    "488": 0.030,
    "640": 0.030,  # if you use 640 on OBIS group elsewhere
    "377": 0.012,  # example for CUBE current or analog; adapt to your device
    "517": 1.000,  # relays are on/off, but keep a placeholder
    "532": 1.000,
    "Hg_Ar": 1.000
}

# Automated measurement list (you can change from GUI too)
DEFAULT_ALL_LASERS = ["445", "405", "377", "Hg_Ar"]

# Integration time bounds (ms)
IT_MIN = 0.2
IT_MAX = 3000.0
SAT_THRESH = 65400  # ~16-bit ADC ceiling

# Auto-IT target window
TARGET_LOW = 60000
TARGET_HIGH = 65000
TARGET_MID = 62500

# Auto-IT controller steps
IT_STEP_UP = 0.30     # if too low, increase IT
IT_STEP_DOWN = 0.10   # if too high, decrease IT
MAX_IT_ADJUST_ITERS = 1000

# Default starting IT by source
DEFAULT_START_IT = {
    "532": 5.0,
    "517": 80.0,
    "Hg_Ar": 10.0,
    "default": 2.4
}

# Measurement cycles (adjust to taste)
N_SIG = 50
N_DARK = 50
N_SIG_640 = 10
N_DARK_640 = 10

# Known Hg-Ar lines (nm) for dispersion
KNOWN_HG_AR_NM = [289.36, 296.73, 302.15, 313.16, 334.19, 365.01,
                  404.66, 407.78, 435.84, 507.30, 546.08]

# Stray-light IB window half-width (pixels)
IB_REGION_HALF = 2


# ===================================================
# ============== Utility / Data Classes =============
# ===================================================

@dataclass
class HardwareState:
    dll_path: str = ""
    com_ports: Dict[str, str] = field(default_factory=lambda: DEFAULT_COM_PORTS.copy())
    laser_power: Dict[str, float] = field(default_factory=lambda: DEFAULT_LASER_POWERS.copy())

@dataclass
class MeasurementData:
    # Rows like: [Timestamp, Wavelength, IntegrationTime, NumCycles, Pixel_0..Pixel_N-1]
    rows: List[List] = field(default_factory=list)
    npix: int = 2048
    serial_number: str = "Unknown"

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["Timestamp", "Wavelength", "IntegrationTime", "NumCycles"] + [f"Pixel_{i}" for i in range(self.npix)]
        return pd.DataFrame(self.rows, columns=cols)

    def last_vectors_for(self, tag: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (last_signal, last_dark) vectors for a given wavelength tag."""
        if not self.rows:
            return None, None
        df = self.to_dataframe()
        pix_cols = [c for c in df.columns if str(c).startswith("Pixel_")]
        sig_rows = df[df["Wavelength"] == tag]
        dark_rows = df[df["Wavelength"] == f"{tag}_dark"]
        sig = sig_rows.iloc[-1][pix_cols].to_numpy(dtype=float) if not sig_rows.empty else None
        dark = dark_rows.iloc[-1][pix_cols].to_numpy(dtype=float) if not dark_rows.empty else None
        return sig, dark


# ===================================================
# ============== Device Control Helpers =============
# ===================================================

class SerialDevice:
    """Small wrapper around pyserial with safe open/close and commands."""
    def __init__(self, port: str, baud: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.eol = "\r\n"
        self.ser: Optional[serial.Serial] = None
        self.lock = threading.Lock()

    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def open(self) -> bool:
        # Close first to avoid stale handles
        self.close()
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            return True
        except Exception:
            self.ser = None
            return False

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        finally:
            self.ser = None

    def _ensure_open(self) -> None:
        if not self.is_open():
            ok = self.open()
            if not ok:
                raise RuntimeError(f"Could not open serial port '{self.port}'")

    def write_line(self, s: str):
        self._ensure_open()
        with self.lock:
            eol = getattr(self, 'eol', '\r\n')
            self.ser.write((s + eol).encode())
            # small settle delay helps some controllers
            time.sleep(0.2)

    def read_all_lines(self) -> List[str]:
        self._ensure_open()
        with self.lock:
            resp = self.ser.readlines()
        try:
            return [r.decode(errors="ignore").strip() for r in resp]
        except Exception:
            return []
    def read_all_text(self, wait: float = 0.3) -> str:
        """Read all bytes as text after an optional short wait."""
        self._ensure_open()
        with self.lock:
            import time as _t
            if wait and wait > 0:
                _t.sleep(wait)
            try:
                data = self.ser.read_all()
            except Exception:
                data = b""
        try:
            return data.decode(errors="ignore").strip()
        except Exception:
            return ""



class LaserController:
    """Encapsulate OBIS (multi-channel), CUBE (377), and Relay (532/517/Hg-Ar) behavior."""
    def __init__(self):
        self.obis = SerialDevice(DEFAULT_COM_PORTS["OBIS"], 9600, 1)
        self.cube = SerialDevice(DEFAULT_COM_PORTS["CUBE"], 19200, 1)
        self.relay = SerialDevice(DEFAULT_COM_PORTS["RELAY"], 9600, 1)
        self.cube.eol = "\r"

    def all_off(self):
        # Try to open ports and switch everything OFF; ignore errors (ports may not exist yet)
        try:
            try: self.obis._ensure_open()
            except: pass
            for ch in OBIS_LASER_MAP.values():
                try: self.obis.write_line(f"SOUR{ch}:AM:STAT OFF")
                except: pass
        except: pass
        try:
            try: self.cube._ensure_open()
            except: pass
            try: self.cube.write_line("L=0")
            except: pass
        except: pass
        try:
            try: self.relay._ensure_open()
            except: pass
            for ch in (1, 2, 3):  # 532, Hg-Ar, 517 (adjust if your mapping differs)
                try: self.relay.write_line(f"R{ch}R")
                except: pass
        except: pass


    def configure_ports(self, ports: Dict[str, str]):
        self.obis.port = ports.get("OBIS", self.obis.port)
        self.cube.port = ports.get("CUBE", self.cube.port)
        self.relay.port = ports.get("RELAY", self.relay.port)

    def open_all(self) -> Tuple[bool, bool, bool]:
        ok_obis = self.obis.open()
        ok_cube = self.cube.open()
        ok_relay = self.relay.open()
        return ok_obis, ok_cube, ok_relay

    def ensure_open_for_tag(self, tag: str):
        """Open the right serial device for the given source tag."""
        if tag in OBIS_LASER_MAP:
            self.obis._ensure_open()
        elif tag == "377":
            self.cube._ensure_open()
        elif tag in ("517", "532", "Hg_Ar"):
            self.relay._ensure_open()

    # ----- OBIS -----
    def obis_cmd(self, cmd: str) -> List[str]:
        self.obis.write_line(cmd)
        return self.obis.read_all_lines()

    def obis_on(self, channel: int):
        self.obis_cmd(f"SOUR{channel}:AM:STAT ON")

    def obis_off(self, channel: int):
        self.obis_cmd(f"SOUR{channel}:AM:STAT OFF")

    def obis_set_power(self, channel: int, watts: float):
        self.obis_cmd(f"SOUR{channel}:POW:LEV:IMM:AMPL {watts:.4f}")

    # ----- CUBE (example protocol: set current then L=1) -----
    def cube_cmd(self, cmd: str) -> List[str]:
        self.cube.write_line(cmd)
        resp = self.cube.read_all_text(wait=1.0)
        return resp.splitlines() if resp else []

    def cube_on(self, power_mw: float = None, current_mA: float = None):
        """Turn on CUBE (377 nm).
        If power_mw is provided, use EXT=1; CW=1; P=<mW>; L=1.
        Else if current_mA provided, send I=<mA>; L=1 (legacy fallback).
        """
        # Ensure CR line endings for CUBE
        try:
            self.cube.eol = "\r"
        except Exception:
            pass
        if power_mw is not None:
            try: self.cube_cmd("EXT=1")
            except Exception: pass
            try: self.cube_cmd("CW=1")
            except Exception: pass
            try: self.cube_cmd(f"P={int(round(power_mw))}")
            except Exception: pass
            self.cube_cmd("L=1")
        elif current_mA is not None:
            try: self.cube_cmd(f"I={current_mA:.2f}")
            except Exception: pass
            self.cube_cmd("L=1")
        else:
            # Default to 12 mW if nothing given
            try: self.cube_cmd("EXT=1")
            except Exception: pass
            try: self.cube_cmd("CW=1")
            except Exception: pass
            try: self.cube_cmd("P=12")
            except Exception: pass
            self.cube_cmd("L=1")

    def cube_off(self):
        self.cube_cmd("L=0")

    # ----- Relay board: "R{n}S" set, "R{n}R" reset -----
    # ----- Relay board: "R{n}S" set, "R{n}R" reset -----
    def relay_on(self, ch: int):
        self.relay.write_line(f"R{ch}S")

    def relay_off(self, ch: int):
        self.relay.write_line(f"R{ch}R")


# ===================================================
# ================== Analysis Logic =================
# ===================================================

def compute_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Return Full Width Half Max in x-units (linear interpolation)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 3:
        return 0.0
    y_norm = y - np.nanmin(y)
    if np.nanmax(y_norm) <= 0:
        return 0.0
    y_norm /= np.nanmax(y_norm)
    half = 0.5
    above = np.where(y_norm >= half)[0]
    if len(above) < 2:
        return 0.0
    left_idx = above[0]
    right_idx = above[-1]
    # interpolate edges
    def interp(i1, i2):
        if i2 == i1: return x[i1]
        return x[i1] + (x[i2] - x[i1]) * (half - y_norm[i1]) / (y_norm[i2] - y_norm[i1])
    if left_idx == 0:
        x_left = x[0]
    else:
        x_left = interp(left_idx-1, left_idx)
    if right_idx == len(x)-1:
        x_right = x[-1]
    else:
        x_right = interp(right_idx, right_idx+1)
    return float(max(0.0, x_right - x_left))

def best_ordered_linear_match(peaks_pix: np.ndarray, candidate_wls: List[float], min_points: int = 5):
    """
    Find a linear fit wl = a*pix + b that best matches ordered lists.
    Returns (rmse, a, b, pix_sel, wl_sel) or None.
    """
    peaks_pix = np.asarray(peaks_pix, dtype=float)
    P, L = len(peaks_pix), len(candidate_wls)
    if P == 0 or L == 0:
        return None

    def score(pix_sel, wl_sel):
        A = np.vstack([pix_sel, np.ones_like(pix_sel)]).T
        a, b = np.linalg.lstsq(A, wl_sel, rcond=None)[0]
        pred = a * pix_sel + b
        rmse = np.sqrt(np.mean((wl_sel - pred) ** 2))
        return rmse, a, b

    best = None
    if P >= L:
        # slide a window over peaks
        for i in range(P - L + 1):
            pix_sel = peaks_pix[i:i+L]
            wl_sel = np.array(candidate_wls)
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())
    else:
        # slide over known lines
        for j in range(L - P + 1):
            pix_sel = peaks_pix.copy()
            wl_sel = np.array(candidate_wls[j:j+P])
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())
    if best and len(best[3]) >= min_points:
        return best
    return None

def normalized_lsf_from_df(df: pd.DataFrame, tag: str, sat_thresh: float = SAT_THRESH, use_latest: bool = True) -> Optional[np.ndarray]:
    """Build normalized LSF = (signal - dark)/max with guards."""
    if df is None or df.empty:
        return None
    pix_cols = [c for c in df.columns if str(c).startswith("Pixel_")]
    if not pix_cols:
        return None
    sig_rows = df[df["Wavelength"] == tag]
    dark_rows = df[df["Wavelength"] == f"{tag}_dark"]
    if sig_rows.empty or dark_rows.empty:
        return None
    sig_row = sig_rows.iloc[-1] if use_latest else sig_rows.iloc[0]
    dark_row = dark_rows.iloc[-1] if use_latest else dark_rows.iloc[0]
    sig = sig_row[pix_cols].to_numpy(dtype=float)
    dark = dark_row[pix_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(sig)) or not np.all(np.isfinite(dark)):
        return None
    if np.nanmax(sig) >= sat_thresh:
        # saturated signal -> skip
        return None
    lsf = sig - dark
    lsf -= np.nanmin(lsf)
    mx = np.nanmax(lsf)
    if mx <= 0:
        return None
    return lsf / mx

def stray_light_metrics(lsf: np.ndarray, peak_pixel: int, ib_half: int = IB_REGION_HALF) -> Dict[str, float]:
    """Compute basic stray light ratios: OOB/IB etc."""
    n = len(lsf)
    ib_start = max(0, peak_pixel - ib_half)
    ib_end = min(n, peak_pixel + ib_half + 1)
    ib_region = np.arange(ib_start, ib_end)
    ib_sum = float(np.sum(lsf[ib_region]))
    # OOB = everything else
    mask = np.ones(n, dtype=bool)
    mask[ib_region] = False
    oob_sum = float(np.sum(lsf[mask]))
    ratio = (oob_sum / ib_sum) if ib_sum > 0 else np.nan
    return {"IB_sum": ib_sum, "OOB_sum": oob_sum, "OOB_over_IB": ratio}


# ===================================================
# ================== Main Application ===============
# ===================================================

class SpectroApp(tk.Tk):
    def __init__(self):
        # IT coordination
        self._pending_it = None  # type: float | None
        self._it_updating = False

        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1250x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.it_history = []

        # State
        self.hw = HardwareState()
        self.lasers = LaserController()
        self.spec: Optional[Avantes_Spectrometer] = None
        self.sn: str = "Unknown"
        self.npix: int = 2048

        # live view thread control
        self.live_running = threading.Event()
        self.live_thread: Optional[threading.Thread] = None
        self._live_busy = False


        self.spec_lock = threading.Lock()
        self.pending_it_ms = None
        self.current_it_ms = 2.4  

        # measurement control
        self.measure_running = threading.Event()
        self.measure_thread: Optional[threading.Thread] = None

        # in-memory measurements
        self.data = MeasurementData(npix=self.npix, serial_number=self.sn)

        # UI
        self._build_ui()

        # load persisted settings after UI is fully constructed
        self.after_idle(self.load_settings_into_ui)
        self.after(300, self._all_off_on_start)


    # ------------------ UI Construction ------------------
    def _all_off_on_start(self):
        try:
            self._update_ports_from_ui()   # pick up UI COM entries if present
            self.lasers.all_off()
        except:
            pass

    def _live_reset_view(self):
        self.live_limits_locked = False
        try:
            self.live_ax.relim()
            self.live_ax.autoscale()
            self.live_fig.canvas.draw_idle()
        except:
            pass
def _live_loop(self):
    if not self.live_running.is_set():
        return
    if getattr(self, "_live_busy", False):
        self.after(30, self._live_loop)
        return

    self._live_busy = True
    try:
        y = np.array([], dtype=float)
        try:
            with self.spec_lock:
                self.spec.measure(ncy=1)
                self.spec.wait_for_measurement()
                y = np.array(getattr(self.spec, "rcm", []), dtype=float)

                if self.pending_it_ms is not None:
                    try:
                        self.spec.set_it(self.pending_it_ms)
                        self._post_info(f"Applied queued IT={self.pending_it_ms:.3f} ms.")
                        self.current_it_ms = self.pending_it_ms
                    finally:
                        self.pending_it_ms = None

        except Exception as e:
            self._post_error("Live frame", e)
            y = np.array([], dtype=float)

        x = np.arange(len(y))
        self._update_live_spectrum(x, y)

    finally:
        self._live_busy = False
        self.after(30, self._live_loop)



    def _attempt_reconnect(self):
        try:
            if hasattr(self, "spec") and hasattr(self.spec, "disconnect"):
                try: self.spec.disconnect()
                except Exception: pass
            # Reconnect using your existing connect logic
            self.connect_spectrometer()  # assuming this sets self.spec and self.sn
            # Restore current IT if you keep it in state
            if hasattr(self, "current_it_ms"):
                try: self.spec.set_it(self.current_it_ms)
                except Exception: pass
            self._post_info("Reconnected to spectrometer.")
        except Exception as e:
            self._post_error("Reconnect failed", e)


    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.live_tab = ttk.Frame(nb)
        self.measure_tab = ttk.Frame(nb)
        self.analysis_tab = ttk.Frame(nb)
        self.setup_tab = ttk.Frame(nb)

        nb.add(self.live_tab, text="Live View")
        nb.add(self.measure_tab, text="Measurements")
        nb.add(self.analysis_tab, text="Analysis")
        nb.add(self.setup_tab, text="Setup")

        self._build_live_tab()
        self._build_measure_tab()
        self._build_analysis_tab()
        self._build_setup_tab()

    # ------------------ Live View Tab --------------------
    def _build_live_tab(self):
         from tabs.live_view_tab import build as _build
         _build(self)
def apply_it(self):
    """Set integration time (ms) safely. Queues during live/measure; applies when idle."""
    try:
        raw = getattr(self, 'it_entry', None).get().strip() if getattr(self, 'it_entry', None) else str(getattr(self, 'current_it_ms', 2.4))
        it_ms = float(raw)
    except Exception:
        self._post_error("Invalid IT", "Could not parse integration time entry.")
        return

    IT_MIN = getattr(self, "IT_MIN", 0.02)        # ms
    IT_MAX = getattr(self, "IT_MAX", 10000.0)     # ms
    it_ms = max(IT_MIN, min(IT_MAX, it_ms))

    self.current_it_ms = it_ms

    if getattr(self, "live_running", None) and self.live_running.is_set():
        self.pending_it_ms = it_ms
        self._post_info(f"Queued IT={it_ms:.3f} ms (will apply next frame).")
        return
    if getattr(self, "measure_running", None) and self.measure_running.is_set():
        self.pending_it_ms = it_ms
        self._post_info(f"Queued IT={it_ms:.3f} ms (will apply between steps).")
        return

    try:
        with self.spec_lock:
            self.spec.set_it(it_ms)
        self._post_info(f"Set IT={it_ms:.3f} ms.")
    except Exception as e:
        self._post_error("Set IT failed", e)



    def start_live(self):
        if not self.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        if self.live_running.is_set():
            return
        self.live_running.set()
        self.live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self.live_thread.start()

    def stop_live(self):
        self.live_running.clear()

    def _live_loop(self):
        while self.live_running.is_set():
            try:
                # Start one frame
                self.spec.measure(ncy=1)
                # Wait for frame to complete
                self.spec.wait_for_measurement()

                # Apply any deferred IT safely after the completed frame
                if self._pending_it is not None:
                    try:
                        it_to_apply = self._pending_it
                        self._pending_it = None
                        self._it_updating = True
                        self.spec.set_it(it_to_apply)
                        try:
                            self.title(f"Applied IT={it_to_apply:.3f} ms")
                        except Exception:
                            pass
                    except Exception as e:
                        self._post_error("Apply IT (deferred)", e)
                    finally:
                        self._it_updating = False
                        try:
                            self.apply_it_btn.state(["!disabled"])  # if exists
                        except Exception:
                            pass

                # After IT changes (or none), fetch data and draw
                y = np.array(self.spec.rcm, dtype=float)
                x = np.arange(len(y))
                self.npix = len(y)
                self.data.npix = self.npix
                self._update_live_spectrum(x, y)

            except Exception as e:
                self._post_error("Live error", e)
                break
def _update_live_spectrum(self, x, y):
    try:
        y = np.asarray(y, dtype=float) if y is not None else np.array([], dtype=float)
        x = np.asarray(x, dtype=float) if x is not None else np.arange(len(y))

        if y.size == 0:
            if hasattr(self, 'live_line'):
                self.live_line.set_data([], [])
            if hasattr(self, 'live_ax'):
                self.live_ax.set_xlim(0, 10)
                self.live_ax.set_ylim(0, 1000.0)
            if hasattr(self, 'live_canvas'):
                self.live_canvas.draw_idle()
            return

        if hasattr(self, 'live_line'):
            self.live_line.set_data(x, y)
        if hasattr(self, 'live_ax'):
            self.live_ax.set_xlim(0, max(10, len(y) - 1))
            self.live_ax.set_ylim(0, self._safe_ymax(y))
        if hasattr(self, 'live_canvas'):
            self.live_canvas.draw_idle()
    except Exception as e:
        self._post_error("Live plot update", e)

def _update_live_plot(self, y, it: float, peak: float, tag: str):
    xs = np.arange(len(y))
    if hasattr(self, 'meas_sig_line'):
        self.meas_sig_line.set_data(xs, y)
    xmax = max(10, len(y) - 1)
    ymax = self._safe_ymax(y)
    if hasattr(self, 'meas_ax'):
        self.meas_ax.set_xlim(0, xmax)
        self.meas_ax.set_ylim(0, ymax)
        self.meas_ax.set_title(f"Spectrometer= {getattr(self, 'sn', 'Unknown')}: Live Measurement for {tag} nm | IT={it:.1f} ms | peak={peak:.0f}")
    if hasattr(self, 'meas_canvas'):
        self.meas_canvas.draw_idle()

def _auto_adjust_it(self, start_it: float, tag: str) -> tuple[float, float]:
    IT_MIN = getattr(self, "IT_MIN", 0.02)
    IT_MAX = getattr(self, "IT_MAX", 10000.0)
    SAT_THRESH = getattr(self, "SAT_THRESH", 0.98 * 65535)
    TARGET_LOW = getattr(self, "TARGET_LOW", 0.45 * 65535)
    TARGET_HIGH = getattr(self, "TARGET_HIGH", 0.65 * 65535)
    IT_STEP_UP = getattr(self, "IT_STEP_UP", 0.2)
    IT_STEP_DOWN = getattr(self, "IT_STEP_DOWN", 0.2)
    MAX_IT_ADJUST_ITERS = getattr(self, "MAX_IT_ADJUST_ITERS", 20)

    it_ms = max(IT_MIN, min(IT_MAX, start_it))
    peak = float('nan')
    iters = 0
    self.it_history = []
    fails = 0
    MAX_FAILS = 3

    def keep_running():
        mr = getattr(self, "measure_running", None)
        return (mr is None) or mr.is_set()

    while iters <= MAX_IT_ADJUST_ITERS and keep_running():
        try:
            with self.spec_lock:
                self.spec.set_it(it_ms)
                self.spec.measure(ncy=1)
                self.spec.wait_for_measurement()
                y = np.array(getattr(self.spec, "rcm", []), dtype=float)
            fails = 0
        except Exception as e:
            fails += 1
            if fails >= MAX_FAILS:
                self._post_error("Auto-IT frame", e)
                return it_ms, peak
            import time; time.sleep(0.02)
            continue

        if y.size == 0 or not np.isfinite(y).any():
            iters += 1
            continue

        try:
            peak = float(np.nanmax(y))
        except ValueError:
            iters += 1
            continue

        self.it_history.append((it_ms, peak))
        self.after(0, lambda arr=y.copy(), it_val=it_ms, pk=peak, tg=tag:
                   self._update_live_plot(arr, it_val, pk, tg))

        if peak >= SAT_THRESH:
            it_ms = max(IT_MIN, it_ms * 0.7)
        elif TARGET_LOW <= peak <= TARGET_HIGH:
            return it_ms, peak
        elif peak < TARGET_LOW:
            it_ms = min(IT_MAX, it_ms + IT_STEP_UP)
        else:
            it_ms = max(IT_MIN, it_ms - IT_STEP_DOWN)

        iters += 1

    return it_ms, peak
if __name__ == "__main__":
    app = SpectroApp()
    app.mainloop()
