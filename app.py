# app.py (tab builders delegated to tabs/* modules)
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
from tkinter import ttk, filedialog, messagebox

# matplotlib embed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# serial
import serial
import serial.tools.list_ports

# analysis helpers
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Avantes wrapper (must be present alongside this file)
from avantes_spectrometer import Avantes_Spectrometer

# Characterization analysis helpers
from characterization_analysis import (
    AnalysisArtifact,
    CharacterizationConfig,
    CharacterizationResult,
    perform_characterization,
)


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
    "488": 3,
    "640": 2,
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
DEFAULT_ALL_LASERS = ["532", "445", "405", "377", "Hg_Ar", "640"]

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

        # measurement control
        self.measure_running = threading.Event()
        self.measure_thread: Optional[threading.Thread] = None

        # in-memory measurements
        self.data = MeasurementData(npix=self.npix, serial_number=self.sn)

        # Characterization analysis state
        self.char_config = CharacterizationConfig()
        self.analysis_result: Optional[CharacterizationResult] = None
        self.analysis_artifacts: List[AnalysisArtifact] = []
        self.analysis_summary_lines: List[str] = []
        self.results_folder: Optional[str] = None
        self.last_results_timestamp: Optional[str] = None
        self.latest_csv_path: Optional[str] = None

        # expose configuration constants for helper modules
        self.IT_MIN = IT_MIN
        self.IT_MAX = IT_MAX
        self.SAT_THRESH = SAT_THRESH
        self.TARGET_LOW = TARGET_LOW
        self.TARGET_HIGH = TARGET_HIGH
        self.TARGET_MID = TARGET_MID
        self.IT_STEP_UP = IT_STEP_UP
        self.IT_STEP_DOWN = IT_STEP_DOWN
        self.MAX_IT_ADJUST_ITERS = MAX_IT_ADJUST_ITERS
        self.DEFAULT_START_IT = DEFAULT_START_IT
        self.N_SIG = N_SIG
        self.N_DARK = N_DARK
        self.N_SIG_640 = N_SIG_640
        self.N_DARK_640 = N_DARK_640
        self.DEFAULT_COM_PORTS = DEFAULT_COM_PORTS
        self.DEFAULT_LASER_POWERS = DEFAULT_LASER_POWERS
        self.SETTINGS_FILE = SETTINGS_FILE

        # UI
        self._build_ui()

        # load persisted settings if any
        self.load_settings_into_ui()
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
    def _build_measure_tab(self):
        from tabs.measurements_tab import build as _build
        _build(self)
    def _build_analysis_tab(self):
        from tabs.analysis_tab import build as _build
        _build(self)
    def _build_setup_tab(self):
        from tabs.setup_tab import build as _build
        _build(self)

    # ------------------ Characterization results helpers ------------------
    def _prepare_results_folder(self) -> Tuple[str, str]:
        sn = self.sn or "Unknown"
        base = os.path.join(os.getcwd(), "data", sn)
        os.makedirs(base, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base, timestamp

    def _safe_ymax(self, arr: np.ndarray) -> float:
        try:
            if arr is None or len(arr) == 0 or not np.isfinite(arr).any():
                return 1000.0
            return max(1000.0, float(np.nanmax(arr)) * 1.1)
        except Exception:
            return 1000.0

    def _update_auto_it_plot(self, tag: str, spectrum: np.ndarray, it_ms: float, peak: float) -> None:
        if not hasattr(self, "meas_sig_line"):
            return

        x = np.arange(len(spectrum))
        self.meas_sig_line.set_data(x, spectrum)
        if hasattr(self, "meas_dark_line"):
            self.meas_dark_line.set_data([], [])

        xmax = max(10, len(spectrum) - 1)
        ymax = self._safe_ymax(spectrum)
        try:
            self.meas_ax.set_xlim(0, xmax)
            self.meas_ax.set_ylim(0, ymax)
            self.meas_ax.set_title(
                f"Spectrometer= {self.sn or 'Unknown'}: Live Measurement for {tag} nm | IT={it_ms:.1f} ms | peak={peak:.0f}"
            )
        except Exception:
            pass
        try:
            self.meas_canvas.draw_idle()
        except Exception:
            pass

    def _clear_analysis_notebook(self) -> None:
        if getattr(self, "analysis_canvases", None):
            for canvas in self.analysis_canvases:
                try:
                    canvas.get_tk_widget().destroy()
                except Exception:
                    pass
        if getattr(self, "analysis_notebook", None):
            for tab_id in self.analysis_notebook.tabs():
                self.analysis_notebook.forget(tab_id)
        self.analysis_canvases = []

    def _update_analysis_ui(self, csv_path: Optional[str] = None) -> None:
        if not hasattr(self, "analysis_notebook"):
            return

        self._clear_analysis_notebook()

        if not self.analysis_artifacts:
            self.analysis_status_var.set("Run measurements to generate characterization charts.")
            self.analysis_text.configure(state="normal")
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("1.0", "No analysis has been generated yet.")
            self.analysis_text.configure(state="disabled")
            self.export_plots_btn.state(["disabled"])
            self.open_folder_btn.state(["disabled"])
            return

        if csv_path is None:
            csv_path = self.latest_csv_path or ""

        status_file = os.path.basename(csv_path) if csv_path else "saved measurements"
        status = f"Analysis generated from {status_file}"
        if self.results_folder:
            status += f" in {self.results_folder}"
        self.analysis_status_var.set(status)

        for artifact in self.analysis_artifacts:
            frame = ttk.Frame(self.analysis_notebook)
            self.analysis_notebook.add(frame, text=artifact.name)
            canvas = FigureCanvasTkAgg(artifact.figure, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(canvas, frame)
            self.analysis_canvases.append(canvas)

        summary_text = "\n".join(self.analysis_summary_lines) if self.analysis_summary_lines else ""
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", summary_text or "Characterization completed.")
        self.analysis_text.configure(state="disabled")

        self.export_plots_btn.state(["!disabled"])
        self.open_folder_btn.state(["!disabled"])

    def refresh_analysis_view(self):
        self._update_analysis_ui(self.latest_csv_path)

    def _finalize_measurement_run(self) -> None:
        if not self.data.rows:
            return
        try:
            folder, timestamp = self._prepare_results_folder()
            df = self.data.to_dataframe()
            csv_path = os.path.join(folder, f"All_Lasers_Measurements_{self.sn or 'Unknown'}_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            result = perform_characterization(df, self.sn or "Unknown", folder, timestamp, self.char_config)
            self.analysis_result = result
            self.analysis_artifacts = result.artifacts
            self.analysis_summary_lines = result.summary_lines
            self.analysis_summary_lines.insert(0, f"✅ Saved measurements to {csv_path}")
            self.analysis_summary_lines.append(f"Plots saved to {folder}")
            self.results_folder = folder
            self.last_results_timestamp = timestamp
            self.latest_csv_path = csv_path
            self.after(0, lambda: self._update_analysis_ui(csv_path))
        except Exception as exc:
            try:
                self._post_error("Characterization", exc)
            except Exception:
                raise

    def export_analysis_plots(self):
        if not self.analysis_artifacts:
            return
        folder = filedialog.askdirectory(title="Select folder for exported plots")
        if not folder:
            return
        try:
            for artifact in self.analysis_artifacts:
                name = artifact.name.replace(" ", "_")
                out_path = os.path.join(folder, f"{name}_{self.last_results_timestamp or ''}.png")
                artifact.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        except Exception as exc:
            self._post_error("Export Plots", exc)

    def open_results_folder(self):
        if not self.results_folder:
            messagebox.showinfo("Results", "No results folder available yet.")
            return
        folder = self.results_folder
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f"open '{folder}'")
            else:
                os.system(f"xdg-open '{folder}' >/dev/null 2>&1 &")
        except Exception as exc:
            self._post_error("Open Folder", exc)
