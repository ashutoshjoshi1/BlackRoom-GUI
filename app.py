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
        if self._live_busy:
            # Schedule next tick & bail; avoids overlapping DLL calls
            self.after(30, self._live_loop)
            return

        self._live_busy = True
        try:
            # One-shot acquisition
            try:
                # Your existing Avantes calls here; this is typical:
                self.spec.measure(ncy=1)
                self.spec.wait_for_measurement()
                y = np.array(self.spec.rcm, dtype=float) if hasattr(self.spec, "rcm") else np.array([], dtype=float)
            except Exception as e:
                # Handle transient Avantes errors (-5 pending, -16 comm) by skipping this frame
                # You can also log e here
                y = np.array([], dtype=float)

            x = np.arange(len(y))
            # Update plot safely (will handle empty y)
            self._update_live_spectrum(x, y)

        finally:
            self._live_busy = False
            # Schedule next loop tick
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
        if not self.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        # Parse & clamp
        try:
            it = float(self.it_entry.get())
        except Exception as e:
            messagebox.showerror("Apply IT", f"Invalid IT value: {e}")
            return
        it = max(IT_MIN, min(IT_MAX, it))

        # If live is running, defer until between frames
        if getattr(self, 'live_running', None) and self.live_running.is_set():
            self._pending_it = it
            try:
                self.apply_it_btn.state(["disabled"])  # if button exists
            except Exception:
                pass
            # non-blocking toast via title/status
            try:
                self.title(f"Queued IT={it:.3f} ms (will apply after current frame)")
            except Exception:
                pass
            return

        # If a measurement is in-flight, wait briefly
        try:
            if getattr(self.spec, 'measuring', False):
                t0 = time.time()
                while getattr(self.spec, 'measuring', False) and time.time() - t0 < 3.0:
                    try:
                        self.spec.wait_for_measurement()
                        break
                    except Exception:
                        time.sleep(0.05)
        except Exception:
            pass

        # Apply now
        try:
            self._it_updating = True
            self.spec.set_it(it)
            messagebox.showinfo("Integration", f"Applied IT = {it:.3f} ms")
        except Exception as e:
            messagebox.showerror("Apply IT", str(e))
        finally:
            self._it_updating = False
            try:
                self.apply_it_btn.state(["!disabled"])  # re-enable
            except Exception:
                pass


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
        """Update the Live View plot safely, even if y is empty."""
        try:
            import numpy as np
            y = np.asarray(y, dtype=float) if y is not None else np.array([], dtype=float)
            x = np.asarray(x, dtype=float) if x is not None else np.arange(len(y))

            # If empty frame, show nothing but keep sane axes; just draw and return.
            if y.size == 0:
                self.live_line.set_data([], [])
                self.live_ax.set_xlim(0, 10)
                self.live_ax.set_ylim(0, 1000.0)
                self.live_canvas.draw_idle()
                return

            # Normal update
            self.live_line.set_data(x, y)
            xmax = max(10, len(y) - 1)
            ymax = self._safe_ymax(y)   # << safe upper bound, no nanmax on empty
            self.live_ax.set_xlim(0, xmax)
            self.live_ax.set_ylim(0, ymax)
            self.live_canvas.draw_idle()

        except Exception as e:
            self._post_error("Live plot update", e)


    def _update_live_plot(self, y: np.ndarray, it: float, peak: float, tag: str):
        """Update the measurement plot with current data (runs on the main UI thread)."""
        xs = np.arange(len(y))
        self.meas_sig_line.set_data(xs, y)
        # Adjust axes limits for new data
        xmax = max(10, len(y) - 1)
        ymax = max(1000.0, float(np.nanmax(y)) * 1.1)
        self.meas_ax.set_xlim(0, xmax)
        self.meas_ax.set_ylim(0, ymax)
        # Update title with current IT and peak info
        self.meas_ax.set_title(f"Spectrometer= {self.sn}: Live Measurement for {tag} nm | IT={it:.1f} ms | peak={peak:.0f}")
        self.meas_canvas.draw_idle()


    def toggle_laser(self, tag: str, turn_on: bool):
        try:
            # make sure we use the latest COM port entries
            self._update_ports_from_ui()
            # open the right serial port lazily
            self.lasers.ensure_open_for_tag(tag)

            if tag in OBIS_LASER_MAP:
                ch = OBIS_LASER_MAP[tag]
                if turn_on:
                    watts = float(self._get_power(tag))
                    self.lasers.obis_set_power(ch, watts)
                    self.lasers.obis_on(ch)
                else:
                    self.lasers.obis_off(ch)

            elif tag == "377":
                if turn_on:
                    val = float(self._get_power(tag))
                    mw = val * 1000.0 if val <= 0.3 else val
                    self.lasers.cube_on(power_mw=mw)
                else:
                    self.lasers.cube_off()

            elif tag == "517":
                if turn_on:
                    self.lasers.relay_on(3)
                else:
                    self.lasers.relay_off(3)

            elif tag == "532":
                if turn_on:
                    self.lasers.relay_on(1)
                else:
                    self.lasers.relay_off(1)

            elif tag == "Hg_Ar":
                if turn_on:
                    self.lasers.relay_on(2)
                else:
                    self.lasers.relay_off(2)

        except Exception as e:
            self._post_error(f"Laser {tag}", e)

    # ------------------ Measurements Tab -----------------
    def _build_measure_tab(self):
         from tabs.measurements_tab import build as _build
         _build(self)

    def run_all_selected(self):
        if not self.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        if self.measure_running.is_set():
            return
        tags = [t for t, v in self.measure_vars.items() if v.get()]
        if not tags:
            messagebox.showwarning("Run", "No lasers selected.")
            return
        start_it_override = None
        try:
            txt = self.auto_it_entry.get().strip()
            if txt:
                start_it_override = float(txt)
        except:
            start_it_override = None
        self.measure_running.set()
        self.measure_thread = threading.Thread(
            target=self._measure_sequence_thread, args=(tags, start_it_override), daemon=True)
        self.measure_thread.start()

    def stop_measure(self):
        self.measure_running.clear()

    def _measure_sequence_thread(self, laser_tags: List[str], start_it_override: Optional[float]):
        # Make sure ports reflect UI and are open for the run
        try:
            self._update_ports_from_ui()
            self.lasers.open_all()
        except Exception as e:
            self._post_error("Ports", e)
            self.measure_running.clear()
            return

        # Ensure everything OFF initially (auto-opens as needed)
        try:
            for ch in OBIS_LASER_MAP.values():
                try: self.lasers.obis_off(ch)
                except: pass
            self.lasers.cube_off()
            self.lasers.relay_off(1)  # 532
            self.lasers.relay_off(2)  # Hg-Ar
            self.lasers.relay_off(3)  # 517
        except Exception:
            pass

        for tag in laser_tags:
            if not self.measure_running.is_set():
                break
            try:
                self._run_single_measurement(tag, start_it_override)
            except Exception as e:
                self._post_error(f"Measurement {tag}", e)

        # Turn all off at the end
        try:
            for ch in OBIS_LASER_MAP.values():
                try: self.lasers.obis_off(ch)
                except: pass
            self.lasers.cube_off()
            self.lasers.relay_off(1)
            self.lasers.relay_off(2)
            self.lasers.relay_off(3)
        except Exception:
            pass

        self.measure_running.clear()

    def _safe_ymax(self, y, floor: float = 1000.0) -> float:
        """Return a sane Y upper bound even if y is empty or all-NaN."""
        import numpy as np
        try:
            y = np.asarray(y, dtype=float)
        except Exception:
            return floor
        if y.size == 0 or not np.isfinite(y).any():
            return floor
        try:
            m = np.nanmax(y)
        except ValueError:
            return floor
        if not np.isfinite(m) or m <= 0:
            return floor
        return max(floor, float(m) * 1.1)

    def _auto_adjust_it(self, start_it: float, tag: str) -> Tuple[float, float]:
        it_ms = max(IT_MIN, min(IT_MAX, start_it))
        peak = float('nan')
        iters = 0
        self.it_history = []  # track (it, peak) for history if needed
        while iters <= MAX_IT_ADJUST_ITERS:
            self.spec.set_it(it_ms)
            self.spec.measure(ncy=1)
            self.spec.wait_for_measurement()
            y = np.array(self.spec.rcm, dtype=float)
            if y.size == 0:
                iters += 1
                continue
            peak = float(np.nanmax(y))
            # Record step and update live plot
            self.it_history.append((it_ms, peak))
            # Schedule main-thread update of the plot for this step
            self.after(0, lambda arr=y.copy(), it_val=it_ms, pk=peak, tg=tag: self._update_live_plot(arr, it_val, pk, tg))
            # Auto-IT logic: adjust integration time
            if peak >= SAT_THRESH:
                it_ms = max(IT_MIN, it_ms * 0.7)
                iters += 1
                continue
            if TARGET_LOW <= peak <= TARGET_HIGH:
                return it_ms, peak  # Found good integration time
            if peak < TARGET_LOW:
                it_ms = min(IT_MAX, it_ms + IT_STEP_UP)
            else:
                it_ms = max(IT_MIN, it_ms - IT_STEP_DOWN)
            iters += 1
        return it_ms, peak  # Return last values if loop ends


    def _ensure_source_state(self, tag: str, turn_on: bool):
        """Turn on/off source described by tag with port auto-open."""
        # ensure correct device port is open
        self.lasers.ensure_open_for_tag(tag)

        if tag in OBIS_LASER_MAP:
            ch = OBIS_LASER_MAP[tag]
            if turn_on:
                pwr = float(self._get_power(tag))
                self.lasers.obis_set_power(ch, pwr)
                self.lasers.obis_on(ch)
            else:
                self.lasers.obis_off(ch)

        elif tag == "377":
            if turn_on:
                val = float(self._get_power(tag))
                mw = val * 1000.0 if val <= 0.3 else val
                self.lasers.cube_on(power_mw=mw)
            else:
                self.lasers.cube_off()

        elif tag == "517":
            if turn_on: self.lasers.relay_on(3)
            else:       self.lasers.relay_off(3)

        elif tag == "532":
            if turn_on: self.lasers.relay_on(1)
            else:       self.lasers.relay_off(1)

        elif tag == "Hg_Ar":
            if turn_on:
                self._countdown_modal(45, "Fiber Switch", "Switch the fiber to Hg-Ar and press Enter to skip.")
                self.lasers.relay_on(2)
            else:
                self.lasers.relay_off(2)


    def _run_single_measurement(self, tag: str, start_it_override: Optional[float]):
        # Turn on only the requested tag; others off
        for k in ["377", "517", "532", "Hg_Ar"]:
            if k != tag:
                try:
                    self._ensure_source_state(k, False)
                except:
                    pass
        for k, ch in OBIS_LASER_MAP.items():
            if k != tag:
                try: self.lasers.obis_off(ch)
                except: pass

        self._ensure_source_state(tag, True)
        time.sleep(1.0)  # allow source to stabilize

        # pick start IT
        start_it = start_it_override if start_it_override is not None else DEFAULT_START_IT.get(tag, DEFAULT_START_IT["default"])
        # Auto-IT
        it_ms, peak = self._auto_adjust_it(start_it)

        if TARGET_LOW <= peak <= TARGET_HIGH:
            # Signal
            self.spec.set_it(it_ms)
            self.spec.measure(ncy=N_SIG)
            self.spec.wait_for_measurement()
            y_signal = np.array(self.spec.rcm, dtype=float)

            # Turn OFF tag
            self._ensure_source_state(tag, False)

            # Dark
            time.sleep(0.3)
            self.spec.set_it(it_ms)
            self.spec.measure(ncy=N_DARK)
            self.spec.wait_for_measurement()
            y_dark = np.array(self.spec.rcm, dtype=float)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.data.rows.append([now, tag, it_ms, N_SIG] + y_signal.tolist())
            self.data.rows.append([now, f"{tag}_dark", it_ms, N_DARK] + y_dark.tolist())

            self._update_last_plots(tag)
        else:
            # could not reach target -> just turn off
            self._ensure_source_state(tag, False)

    def _countdown_modal(self, seconds: int, title: str, message: str):
        """Blocking modal with countdown; Enter key to skip."""
        top = tk.Toplevel(self)
        top.title(title)
        top.geometry("500x180+200+200")
        ttk.Label(top, text=message, wraplength=460).pack(pady=8)
        lbl = ttk.Label(top, text="", font=("Segoe UI", 14))
        lbl.pack(pady=10)

        skip = {"flag": False}
        def on_key(ev):
            skip["flag"] = True
        top.bind("<Return>", on_key)

        for s in range(seconds, -1, -1):
            if skip["flag"]:
                break
            lbl.config(text=f"{s} sec")
            top.update()
            time.sleep(1.0)
        top.destroy()

    def _update_last_plots(self, tag: str):
        sig, dark = self.data.last_vectors_for(tag)
        def update():
            # Update main plot lines
            xmax = 10
            ymax = 1000.0
            if sig is not None:
                xs = np.arange(len(sig))
                self.meas_sig_line.set_data(xs, sig)
                xmax = max(xmax, len(sig) - 1)
                ymax = max(ymax, float(np.nanmax(sig)) * 1.1)
            if dark is not None:
                xd = np.arange(len(dark))
                self.meas_dark_line.set_data(xd, dark)
                xmax = max(xmax, len(dark) - 1)
                ymax = max(ymax, float(np.nanmax(dark)) * 1.1)
            self.meas_ax.set_xlim(0, xmax)
            self.meas_ax.set_ylim(0, ymax)
            self.meas_canvas.draw_idle()
        self.after(0, update)


    def save_csv(self):
        if not self.data.rows:
            messagebox.showwarning("Save CSV", "No data collected yet.")
            return
        df = self.data.to_dataframe()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"Measurements_{self.data.serial_number}_{ts}.csv"
        path = filedialog.asksaveasfilename(
            title="Save CSV", defaultextension=".csv",
            initialfile=default, filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            messagebox.showinfo("Save CSV", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save CSV", str(e))

    # ------------------ Analysis Tab --------------------
    def _build_analysis_tab(self):
         from tabs.analysis_tab import build as _build
         _build(self)
    def start_analysis_from_measure(self):
        """Start a measurement run for the lasers selected in the Measurement tab
        (analysis selection if available), and paint the resulting LSF/other plots in the Analysis tab.
        """
        if not self.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        tags = [t for t, v in getattr(self, 'analysis_vars', {}).items() if v.get()] or \
               [t for t, v in self.measure_vars.items() if v.get()]
        if not tags:
            messagebox.showwarning("Analysis", "No lasers selected to analyze.")
            return
        try:
            self.ana_ax1.clear(); self.ana_ax1.grid(True)
            self.ana_ax2.clear(); self.ana_ax2.grid(True)
            self.analysis_text.delete('1.0', 'end')
        except Exception:
            pass
        self.measure_running.set()
        def _runner():
            try:
                self._measure_sequence_thread(tags, None)
            finally:
                try:
                    self.nb.select(self.analysis_tab)
                except Exception:
                    pass
        self.measure_thread = threading.Thread(target=_runner, daemon=True)
        self.measure_thread.start()
    def run_analysis(self):
        if not self.data.rows:
            messagebox.showwarning("Analysis", "No measurement data available.")
            self.measure_running.clear()
            return
        # Prepare output directory
        df = self.data.to_dataframe()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.getcwd(), 'data')
        sn_dir = os.path.join(base_dir, str(self.sn))
        out_dir = os.path.join(sn_dir, ts)
        os.makedirs(out_dir, exist_ok=True)
        # Save measurements to CSV
        csv_path = os.path.join(out_dir, f"Measurements_{self.sn}_{ts}.csv")
        df.to_csv(csv_path, index=False)
        # Compute normalized LSFs for each laser (excluding Hg_Ar lamp)
        tags = [str(w) for w in df["Wavelength"].unique() if not str(w).endswith("_dark")]
        lsf_list = []; lsf_tags = []
        for tag in tags:
            if tag == "Hg_Ar":
                continue
            lsf = normalized_lsf_from_df(df, tag)
            if lsf is not None:
                lsf_list.append(lsf); lsf_tags.append(tag)
        # Plot all normalized LSFs (log scale)
        if lsf_list:
            plt.ioff()
            fig_norm = plt.figure(figsize=(12,6))
            plt.yscale('log')
            for lsf, tg in zip(lsf_list, lsf_tags):
                label = f"{tg} nm" if tg.isdigit() else tg
                plt.plot(lsf, label=label)
            plt.title(f"Spectrometer= {self.sn}: Normalized LSFs")
            plt.xlabel("Pixel Index"); plt.ylabel("Normalized Intensity")
            plt.grid(True); plt.legend()
            fig_norm.savefig(os.path.join(out_dir, f"Normalized_LSFs_{self.sn}_{ts}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_norm)
        # If Hg_Ar was measured, analyze dispersion (find peaks and match known lines)
        if "Hg_Ar" in tags:
            sig = df[df["Wavelength"]=="Hg_Ar"].iloc[-1, 4:].astype(float).values
            dark = df[df["Wavelength"]=="Hg_Ar_dark"].iloc[-1, 4:].astype(float).values
            signal_corr = np.clip(sig - dark, 1e-5, None)
            peaks, _ = find_peaks(signal_corr, prominence=0.014*np.nanmax(signal_corr), distance=20)
            peaks = np.sort(peaks)
            sol = best_ordered_linear_match(peaks, KNOWN_HG_AR_NM, min_points=5)
            if not sol:
                sol = best_ordered_linear_match(peaks, KNOWN_HG_AR_NM[:-1], min_points=5)
            matched_pixels, matched_wavelengths = ([], [])
            if sol:
                rmse, a_lin, b_lin, matched_pixels, matched_wavelengths = sol
            # Plot Hg-Ar spectrum with peaks labeled
            fig_hg = plt.figure(figsize=(14,6)); plt.yscale('log')
            pixels = np.arange(len(signal_corr))
            plt.plot(pixels, signal_corr, label="Dark-Corrected Hg-Ar Signal", color='blue')
            plt.plot(peaks, signal_corr[peaks], 'ro', label='Detected Peaks')
            for pix, wl in zip(matched_pixels, matched_wavelengths):
                y_val = signal_corr[int(pix)]
                plt.text(pix, y_val + 0.1*y_val, f"{wl:.1f} nm", color='brown', fontsize=10, ha='center')
            plt.xlabel("Pixel"); plt.ylabel("Signal (Counts)")
            plt.title(f"Spectrometer= {self.sn}: Hg-Ar Spectrum with Detected Peaks")
            plt.legend(); plt.grid(True)
            fig_hg.savefig(os.path.join(out_dir, f"HgAr_Spectrum_{self.sn}_{ts}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_hg)
        # Compute SDF matrix and plots if at least one laser LSF is available
        if lsf_list:
            IB_Half = IB_REGION_HALF  # half-width of in-band region for stray light calc (e.g., 10)
            total_pix = len(lsf_list[0])
            SDF_matrix = np.zeros((total_pix, total_pix))
            def normalize_lsf_stray(lsf, pix):
                ib_start = max(0, pix - IB_Half); ib_end = min(len(lsf), pix + IB_Half + 1)
                ib_sum = float(np.sum(lsf[ib_start:ib_end]))
                lsf_out = lsf.copy(); lsf_out[ib_start:ib_end] = 0
                return np.zeros_like(lsf) if ib_sum <= 0 else lsf_out / ib_sum
            pixel_locations = [int(np.nanargmax(lsf)) for lsf in lsf_list]
            # Place each normalized stray-corrected LSF into SDF matrix column
            for lsf, pix in zip(lsf_list, pixel_locations):
                SDF_matrix[:len(lsf), pix] = normalize_lsf_stray(lsf, pix)
            # Shift stray light distributions between measured lines
            for j in range(len(pixel_locations)-1, 0, -1):
                curr_pix = pixel_locations[j]; prev_pix = pixel_locations[j-1]
                for col in range(curr_pix-1, prev_pix, -1):
                    shift = curr_pix - col
                    SDF_matrix[:-shift, col] = SDF_matrix[shift:, curr_pix]
                    SDF_matrix[-shift:, col] = 0
            # Extend first LSF to pixel 0
            first_pix = pixel_locations[0]
            for col in range(first_pix-1, -1, -1):
                shift = first_pix - col
                SDF_matrix[:-shift, col] = SDF_matrix[shift:, first_pix]
                SDF_matrix[-shift:, col] = 0
            # Shift rightmost LSF downward to fill right side
            last_pix = pixel_locations[-1]
            for col in range(last_pix+1, total_pix):
                shift = col - last_pix
                SDF_matrix[shift:, col] = SDF_matrix[:-shift, last_pix]
                SDF_matrix[:shift, col] = 0
            # Fill bottom zeros with last row of shifting LSF
            for j in range(len(pixel_locations)-1, -1, -1):
                curr_pix = pixel_locations[j]
                stop_col = pixel_locations[j-1] + 1 if j > 0 else 0
                last_val = SDF_matrix[-1, curr_pix]
                for col in range(curr_pix-1, stop_col-1, -1):
                    ib_start = max(0, col - IB_Half); ib_end = min(total_pix, col + IB_Half + 1)
                    for row in range(ib_end, total_pix):
                        if SDF_matrix[row, col] == 0:
                            SDF_matrix[row, col] = last_val
            # Fill top zeros with first row of rightmost LSF
            last_pix2 = pixel_locations[-1]; first_val = SDF_matrix[0, last_pix2]
            for col in range(last_pix2+1, total_pix):
                ib_start = max(0, col - IB_Half); ib_end = min(total_pix, col + IB_Half + 1)
                for row in range(0, ib_start):
                    if SDF_matrix[row, col] == 0:
                        SDF_matrix[row, col] = first_val
            # Plot SDF line distribution for each measured line
            fig_sdf = plt.figure(figsize=(12,6))
            for pix in pixel_locations:
                plt.plot(SDF_matrix[:, pix], label=f'{pix} pixel')
            plt.xlabel('Pixels'); plt.ylabel('SDF Value')
            plt.title(f"Spectrometer= {self.sn}: Spectral Distribution Function (SDF)")
            plt.legend(); plt.grid(True)
            fig_sdf.savefig(os.path.join(out_dir, f"SDF_Plot_{self.sn}_{ts}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_sdf)
            # Plot SDF matrix heatmap
            fig_heat, ax_heat = plt.subplots(figsize=(10,6))
            im = ax_heat.imshow(SDF_matrix, aspect='auto', origin='lower', cmap='coolwarm')
            plt.colorbar(im, label='SDF Value')
            ax_heat.set_xlabel('Pixels'); ax_heat.set_ylabel('Spectral Pixel Index')
            ax_heat.set_title(f"Spectrometer= {self.sn}: SDF Matrix Heatmap")
            fig_heat.savefig(os.path.join(out_dir, f"SDF_Heatmap_{self.sn}_{ts}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_heat)
        # Notify user and show Analysis tab
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("end", f"Analysis complete. Results saved in {out_dir}\n")
        try:
            nb = self.live_tab.nametowidget(self.live_tab.winfo_parent())  # get Notebook widget
            nb.select(self.analysis_tab)  # switch to Analysis tab
        except Exception:
            pass
        self.measure_running.clear()

    def export_analysis_plots(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = filedialog.askdirectory(title="Select folder to save analysis plots")
        if not base:
            return
        try:
            p1 = os.path.join(base, f"analysis_plot1_{ts}.png")
            p2 = os.path.join(base, f"analysis_plot2_{ts}.png")
            self.ana_fig1.savefig(p1, dpi=150, bbox_inches="tight")
            self.ana_fig2.savefig(p2, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Export Plots", f"Saved:\n{p1}\n{p2}")
        except Exception as e:
            messagebox.showerror("Export Plots", str(e))

    def export_analysis_summary(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"analysis_summary_{ts}.txt"
        path = filedialog.asksaveasfilename(
            title="Save Summary", defaultextension=".txt",
            initialfile=default, filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        try:
            txt = self.analysis_text.get("1.0", "end").strip()
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt + "\n")
            messagebox.showinfo("Export Summary", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export Summary", str(e))

    # ------------------ Setup Tab -----------------------
    def _build_setup_tab(self):
         from tabs.setup_tab import build as _build
         _build(self)

    def refresh_ports(self):
        ports = list(serial.tools.list_ports.comports())
        names = [p.device for p in ports]
        if names:
            self.obis_entry.delete(0, "end"); self.obis_entry.insert(0, names[0])
            if len(names) > 1:
                self.cube_entry.delete(0, "end"); self.cube_entry.insert(0, names[1])
            if len(names) > 2:
                self.relay_entry.delete(0, "end"); self.relay_entry.insert(0, names[2])
            messagebox.showinfo("Ports", "Populated with first detected ports.\nAdjust as needed.")
        else:
            messagebox.showwarning("Ports", "No serial ports detected.")

    def test_com_connect(self):
        self._update_ports_from_ui()
        ok_obis = self.lasers.obis.open()
        self.obis_status.config(foreground=("green" if ok_obis else "red"))
        ok_cube = self.lasers.cube.open()
        self.cube_status.config(foreground=("green" if ok_cube else "red"))
        ok_relay = self.lasers.relay.open()
        self.relay_status.config(foreground=("green" if ok_relay else "red"))
        # Close after test to free ports (or keep open if you prefer)
        time.sleep(0.2)
        if ok_obis: self.lasers.obis.close()
        if ok_cube: self.lasers.cube.close()
        if ok_relay: self.lasers.relay.close()

    def browse_dll(self):
        path = filedialog.askopenfilename(
            title="Select avaspecx64.dll", filetypes=[("DLL", "*.dll"), ("All files", "*.*")])
        if path:
            self.dll_entry.delete(0, "end")
            self.dll_entry.insert(0, path)

    def connect_spectrometer(self):
        try:
            dll = self.dll_entry.get().strip()
            if not dll or not os.path.isfile(dll):
                raise RuntimeError("Please select a valid avaspecx64.dll")
            # init wrapper
            ava = Avantes_Spectrometer()
            ava.dll_path = dll
            ava.alias = "Ava1"
            ava.npix_active = 2048
            ava.debug_mode = 1
            ava.initialize_spec_logger()

            res = ava.load_spec_dll()
            if res != "OK":
                raise RuntimeError(f"load_spec_dll returned: {res}")
            res = ava.initialize_dll()
            # enumerate
            res, ndev = ava.get_number_of_devices()
            if res != "OK" or ndev <= 0:
                raise RuntimeError("No Avantes devices detected.")
            res, infos = ava.get_all_devices_info(ndev)
            sers = []
            for idx in range(ndev):
                ident = getattr(infos, f"a{idx}")
                sn = ident.SerialNumber
                if isinstance(sn, (bytes, bytearray)):
                    sn = sn.decode('utf-8', errors='ignore')
                sers.append(sn)
            if ndev > 1:
                choices = "\n".join(f"{k+1}: {s}" for k, s in enumerate(sers))
                selection = simpledialog.askinteger(
                    "Select Spectrometer",
                    f"Multiple spectrometers detected:\n{choices}\nEnter device number:",
                    parent=self
                )
                if selection is None:
                    messagebox.showinfo("Spectrometer", "Spectrometer selection cancelled.")
                    return
                if 1 <= selection <= len(sers):
                    ava.sn = sers[selection-1]
                else:
                    messagebox.showwarning("Spectrometer", "Invalid selection. Connecting to first device.")
                    ava.sn = sers[0]
            elif sers:
                ava.sn = sers[0]
            ava.connect()
            self.spec = ava
            self.sn = getattr(ava, "sn", "Unknown")
            self.spec_status.config(text=f"Connected: {self.sn}", foreground="green")
            messagebox.showinfo("Spectrometer", f"Connected to spectrometer {self.sn}")
        except Exception as e:
            self._post_error("Connect Spectrometer", e)
            dll = self.dll_entry.get().strip()
            if not dll or not os.path.isfile(dll):
                raise RuntimeError("Please select a valid avaspecx64.dll")
            # init wrapper
            ava = Avantes_Spectrometer()
            ava.dll_path = dll
            ava.alias = "Ava1"
            ava.npix_active = 2048
            ava.debug_mode = 1
            ava.initialize_spec_logger()

            res = ava.load_spec_dll()
            if res != "OK":
                raise RuntimeError(f"load_spec_dll returned: {res}")
            res = ava.initialize_dll()
            # enumerate
            res, ndev = ava.get_number_of_devices()
            if res != "OK" or ndev <= 0:
                raise RuntimeError("No Avantes devices detected.")
            res, infos = ava.get_all_devices_info(ndev)
            sers = []
            for idx in range(ndev):
                ident = getattr(infos, f"a{idx}")
                sn = ident.SerialNumber
                if isinstance(sn, (bytes, bytearray)):
                    sn = sn.decode('utf-8', errors='ignore')
                sers.append(sn)
            if ndev > 1:
                choices = "\n".join(f"{k+1}: {s}" for k, s in enumerate(sers))
                selection = simpledialog.askinteger(
                    "Select Spectrometer",
                    f"Multiple spectrometers detected:\n{choices}\nEnter device number:",
                    parent=self
                )
                if selection is None:
                    messagebox.showinfo("Spectrometer", "Spectrometer selection cancelled.")
                    return
                if 1 <= selection <= len(sers):
                    ava.sn = sers[selection-1]
                else:
                    messagebox.showwarning("Spectrometer", "Invalid selection. Connecting to first device.")
                    ava.sn = sers[0]
            elif sers:
                ava.sn = sers[0]
            ava.connect()
            self.spec = ava
            self.sn = getattr(ava, "sn", "Unknown")
            self.spec_status.config(text=f"Connected: {self.sn}", foreground="green")
            messagebox.showinfo("Spectrometer", f"Connected to spectrometer {self.sn}")
        

    def disconnect_spectrometer(self):
        try:
            self.stop_live()
            if self.spec:
                try:
                    self.spec.disconnect()
                except Exception:
                    pass
            self.spec = None
            self.spec_status.config(text="Disconnected", foreground="red")
        except Exception as e:
            self._post_error("Spectrometer Disconnect", e)

    def _update_ports_from_ui(self):
        self.hw.com_ports["OBIS"] = self.obis_entry.get().strip() or DEFAULT_COM_PORTS["OBIS"]
        self.hw.com_ports["CUBE"] = self.cube_entry.get().strip() or DEFAULT_COM_PORTS["CUBE"]
        self.hw.com_ports["RELAY"] = self.relay_entry.get().strip() or DEFAULT_COM_PORTS["RELAY"]
        self.lasers.configure_ports(self.hw.com_ports)

    def _get_power(self, tag: str) -> float:
        try:
            e = self.power_entries.get(tag)
            if e is None:
                return DEFAULT_LASER_POWERS.get(tag, 0.01)
            return float(e.get().strip())
        except:
            return DEFAULT_LASER_POWERS.get(tag, 0.01)

    def save_settings(self):
        self._update_ports_from_ui()
        self.hw.dll_path = self.dll_entry.get().strip()
        for tag, e in self.power_entries.items():
            try:
                self.hw.laser_power[tag] = float(e.get().strip())
            except:
                pass
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "dll_path": self.hw.dll_path,
                    "com_ports": self.hw.com_ports,
                    "laser_power": self.hw.laser_power
                }, f, indent=2)
            messagebox.showinfo("Settings", f"Saved to {SETTINGS_FILE}")
        except Exception as e:
            messagebox.showerror("Settings", str(e))

    def load_settings_into_ui(self):
        # If setup widgets are not yet available, try again shortly
        required = ["dll_entry", "obis_entry", "cube_entry", "relay_entry", "power_entries"]
        if not all(hasattr(self, n) for n in required):
            try:
                self.after(50, self.load_settings_into_ui)
            except Exception:
                pass
            return

        if not os.path.isfile(SETTINGS_FILE):
            # init defaults
            self.dll_entry.delete(0, "end")
            self.dll_entry.insert(0, self.hw.dll_path)
            self.obis_entry.delete(0, "end"); self.obis_entry.insert(0, DEFAULT_COM_PORTS["OBIS"])
            self.cube_entry.delete(0, "end"); self.cube_entry.insert(0, DEFAULT_COM_PORTS["CUBE"])
            self.relay_entry.delete(0, "end"); self.relay_entry.insert(0, DEFAULT_COM_PORTS["RELAY"])
            for tag, e in self.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(DEFAULT_LASER_POWERS.get(tag, 0.01)))
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.dll_entry.delete(0, "end")
            self.dll_entry.insert(0, obj.get("dll_path", ""))
            cp = obj.get("com_ports", DEFAULT_COM_PORTS)
            self.obis_entry.delete(0, "end"); self.obis_entry.insert(0, cp.get("OBIS", DEFAULT_COM_PORTS["OBIS"]))
            self.cube_entry.delete(0, "end"); self.cube_entry.insert(0, cp.get("CUBE", DEFAULT_COM_PORTS["CUBE"]))
            self.relay_entry.delete(0, "end"); self.relay_entry.insert(0, cp.get("RELAY", DEFAULT_COM_PORTS["RELAY"]))
            lp = obj.get("laser_power", DEFAULT_LASER_POWERS)
            for tag, e in self.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(lp.get(tag, DEFAULT_LASER_POWERS.get(tag, 0.01))))
        except Exception as e:
            messagebox.showerror("Load Settings", str(e))

    # ------------------ General helpers ------------------

    def _post_error(self, title: str, ex: Exception):
        tb = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f"[{title}] {ex}\n{tb}", file=sys.stderr)
        self.after(0, lambda: messagebox.showerror(title, str(ex)))

    def on_close(self):
        try:
            self.stop_live()
            self.stop_measure()
            if self.spec:
                try: self.spec.disconnect()
                except: pass
            for dev in [self.lasers.obis, self.lasers.cube, self.lasers.relay]:
                try: dev.close()
                except: pass
        finally:
            self.destroy()


if __name__ == "__main__":
    app = SpectroApp()
    app.mainloop()
