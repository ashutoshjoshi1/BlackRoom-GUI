# Auto-generated from gui.py splitter
from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os, time

def build(app):
        top = ttk.Frame(app.analysis_tab)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Analysis Type:").pack(side="left")
        app.analysis_type = tk.StringVar(value="LSF")
        combo = ttk.Combobox(top, textvariable=app.analysis_type, width=18,
                        values=["LSF", "Dispersion", "Stray Light", "Resolution"])
        combo.pack(side="left", padx=8)

        ttk.Label(top, text="Wavelength tag (e.g., 405, 445, Hg_Ar):").pack(side="left", padx=(16, 4))
        app.analysis_tag_entry = ttk.Entry(top, width=15)
        app.analysis_tag_entry.insert(0, "Hg_Ar")
        app.analysis_tag_entry.pack(side="left", padx=4)

        ttk.Button(top, text="Run Analysis", command=app.run_analysis).pack(side="left", padx=12)

        ttk.Button(top, text="Export Plots", command=app.export_analysis_plots).pack(side="right")
        ttk.Button(top, text="Export Summary", command=app.export_analysis_summary).pack(side="right", padx=6)

        mid = ttk.Frame(app.analysis_tab)
        mid.pack(fill="both", expand=True, padx=8, pady=8)

        app.ana_fig1 = Figure(figsize=(7, 4), dpi=100)
        app.ana_ax1 = app.ana_fig1.add_subplot(111)
        app.ana_ax1.set_title("Analysis Plot 1")
        app.ana_ax1.grid(True)
        app.ana_canvas1 = FigureCanvasTkAgg(app.ana_fig1, mid)
        app.ana_canvas1.draw()
        app.ana_canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)
        NavigationToolbar2Tk(app.ana_canvas1, mid)

        app.ana_fig2 = Figure(figsize=(7, 4), dpi=100)
        app.ana_ax2 = app.ana_fig2.add_subplot(111)
        app.ana_ax2.set_title("Analysis Plot 2")
        app.ana_ax2.grid(True)
        app.ana_canvas2 = FigureCanvasTkAgg(app.ana_fig2, mid)
        app.ana_canvas2.draw()
        app.ana_canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)
        NavigationToolbar2Tk(app.ana_canvas2, mid)

        bottom = ttk.Frame(app.analysis_tab)
        bottom.pack(fill="x", padx=8, pady=8)
        ttk.Label(bottom, text="Analysis Summary:").pack(anchor="w")
        app.analysis_text = tk.Text(bottom, height=8)
        app.analysis_text.pack(fill="x", expand=True)


