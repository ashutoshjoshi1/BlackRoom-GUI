
# Analysis tab
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def build(app):
    # Top controls
    top = ttk.Frame(app.analysis_tab)
    top.pack(fill="x", padx=8, pady=8)

    ttk.Label(top, text="Analysis Type:").pack(side="left")
    app.analysis_type = tk.StringVar(value="LSF")
    ttk.Combobox(top, textvariable=app.analysis_type, values=["LSF", "Dispersion", "Stray Light"], width=16, state="readonly").pack(side="left", padx=6)

    ttk.Label(top, text="Tag:").pack(side="left", padx=(12, 2))
    app.analysis_tag_entry = ttk.Entry(top, width=10)
    app.analysis_tag_entry.insert(0, "Hg_Ar")
    app.analysis_tag_entry.pack(side="left")

    ttk.Button(top, text="Run Analysis", command=app.run_analysis).pack(side="left", padx=10)

    # Bottom area: two plots + text
    bot = ttk.Frame(app.analysis_tab)
    bot.pack(fill="both", expand=True, padx=8, pady=8)

    left = ttk.Frame(bot)
    left.pack(side="left", fill="both", expand=True)
    right = ttk.Frame(bot)
    right.pack(side="right", fill="y")

    # Two figures
    app.ana_fig1 = Figure(figsize=(5, 3), dpi=100)
    app.ana_ax1 = app.ana_fig1.add_subplot(111)
    app.ana_ax1.grid(True)

    app.ana_fig2 = Figure(figsize=(5, 3), dpi=100)
    app.ana_ax2 = app.ana_fig2.add_subplot(111)
    app.ana_ax2.grid(True)

    app.ana_canvas1 = FigureCanvasTkAgg(app.ana_fig1, left)
    app.ana_canvas1.draw()
    app.ana_canvas1.get_tk_widget().pack(fill="both", expand=True, pady=(0, 6))
    NavigationToolbar2Tk(app.ana_canvas1, left)

    app.ana_canvas2 = FigureCanvasTkAgg(app.ana_fig2, left)
    app.ana_canvas2.draw()
    app.ana_canvas2.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
    NavigationToolbar2Tk(app.ana_canvas2, left)

    # Text output + export buttons
    ttk.Label(right, text="Summary").pack(anchor="w")
    app.analysis_text = tk.Text(right, width=40, height=25)
    app.analysis_text.pack(fill="y", expand=False)

    ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
    ttk.Button(right, text="Export Plots", command=app.export_analysis_plots).pack(fill="x", pady=2)
    ttk.Button(right, text="Export Summary", command=app.export_analysis_summary).pack(fill="x", pady=2)
