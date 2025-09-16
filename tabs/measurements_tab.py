# Auto-generated from gui.py splitter
from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os, time

def build(app):
        top = ttk.Frame(app.measure_tab)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Automated Measurements").pack(side="left")

        app.run_all_btn = ttk.Button(top, text="Run Selected", command=app.run_all_selected)
        app.stop_all_btn = ttk.Button(top, text="Stop", command=app.stop_measure)
        app.save_csv_btn = ttk.Button(top, text="Save CSV", command=app.save_csv)

        app.run_all_btn.pack(side="right", padx=5)
        app.stop_all_btn.pack(side="right", padx=5)
        app.save_csv_btn.pack(side="right", padx=5)

    
        app.start_analysis_btn = ttk.Button(top, text="Start Analysis", command=app.start_analysis_from_measure)
        app.start_analysis_btn.pack(side="right", padx=5)
    # Laser selection + Auto-IT options
        mid = ttk.Frame(app.measure_tab)
        mid.pack(fill="x", padx=8, pady=8)

        ttk.Label(mid, text="Select lasers to run:").grid(row=0, column=0, sticky="w", padx=4, pady=4)

        app.measure_vars = {}
        tags = ["405", "445", "488", "517", "532", "377", "Hg_Ar"]
        for i, tag in enumerate(tags):
       v = tk.BooleanVar(value=(tag in DEFAULT_ALL_LASERS))
       chk = ttk.Checkbutton(mid, text=tag + " nm", variable=v)
       chk.grid(row=1 + i // 6, column=(i % 6), padx=4, pady=4, sticky="w")
       app.measure_vars[tag] = v

        ttk.Label(mid, text="Auto-IT start (ms, default if blank):").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        app.auto_it_entry = ttk.Entry(mid, width=10)
        app.auto_it_entry.insert(0, "")
        app.auto_it_entry.grid(row=3, column=1, sticky="w", padx=4, pady=4)

        # Result plots (signal & dark)
        bot = ttk.Frame(app.measure_tab)
        bot.pack(fill="both", expand=True, padx=8, pady=8)

        app.meas_fig = Figure(figsize=(12, 6), dpi=100)
        app.meas_ax  = app.meas_fig.add_subplot(111)
        app.meas_ax.set_title("Last Measurement: Signal & Dark")
        app.meas_ax.set_xlabel("Pixel")
        app.meas_ax.set_ylabel("Counts")
        app.meas_ax.grid(True)

        (app.meas_sig_line,)  = app.meas_ax.plot([], [], lw=1.2, label="Signal (ON)")
        (app.meas_dark_line,) = app.meas_ax.plot([], [], lw=1.2, linestyle="--", label="Dark (OFF)")
        app.meas_ax.legend(loc="upper left")

        # Inset for Auto-IT steps (peaks and IT vs step index)
        app.meas_inset = app.meas_ax.inset_axes([0.58, 0.52, 0.38, 0.42])  # x, y, w, h (relative)
        app.meas_inset.set_title("Auto-IT steps")
        app.meas_inset.set_xlabel("step")
        app.meas_inset.set_ylabel("peak")

        app.meas_inset2 = app.meas_inset.twinx()
        app.meas_inset2.set_ylabel("IT (ms)")

        (app.inset_peak_line,) = app.meas_inset.plot([], [], marker="o", lw=1, label="Peak")
        (app.inset_it_line,)   = app.meas_inset2.plot([], [], marker="s", lw=1, linestyle="--", label="IT (ms)")

        app.meas_canvas = FigureCanvasTkAgg(app.meas_fig, bot)
        app.meas_canvas.draw()
        app.meas_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(app.meas_canvas, bot)
