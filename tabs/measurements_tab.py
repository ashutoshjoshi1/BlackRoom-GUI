
# Measurements tab
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def build(app):
    # Top header
    top = ttk.Frame(app.measure_tab)
    top.pack(fill="x", padx=8, pady=8)
    ttk.Label(top, text="Automated Measurements").pack(side="left")

    # Middle controls
    mid = ttk.Frame(app.measure_tab)
    mid.pack(fill="x", padx=8, pady=4)

    # Laser selection checkboxes
    lasers_frame = ttk.LabelFrame(mid, text="Select Lasers")
    lasers_frame.grid(row=0, column=0, padx=6, pady=6, sticky="w")

    app.measure_vars = {}
    for i, tag in enumerate(["405", "445", "488", "377", "517", "532", "Hg_Ar"]):
        v = tk.BooleanVar(value=False)
        text = f"{tag} nm" if tag.isdigit() else tag
        ttk.Checkbutton(lasers_frame, text=text, variable=v).grid(row=i//2, column=i%2, sticky="w", padx=4, pady=2)
        app.measure_vars[tag] = v

    # Start IT override
    ttk.Label(mid, text="Start IT (ms):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    app.auto_it_entry = ttk.Entry(mid, width=10)
    app.auto_it_entry.insert(0, "")
    app.auto_it_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)

    # Run/Stop buttons
    btns = ttk.Frame(mid)
    btns.grid(row=1, column=2, sticky="w", padx=8)
    ttk.Button(btns, text="Run Selected", command=app.run_all_selected).pack(side="left", padx=4)
    ttk.Button(btns, text="Stop", command=app.stop_measure).pack(side="left", padx=4)

    # Bottom plot section
    bot = ttk.Frame(app.measure_tab)
    bot.pack(fill="both", expand=True, padx=8, pady=8)

    app.meas_fig = Figure(figsize=(7, 4), dpi=100)
    app.meas_ax = app.meas_fig.add_subplot(111)
    app.meas_ax.set_title("Measurement Result")
    app.meas_ax.set_xlabel("Pixel")
    app.meas_ax.set_ylabel("Counts")

    # inset for Auto-IT (peaks and IT)
    app.meas_inset = app.meas_ax.inset_axes([0.58, 0.52, 0.38, 0.42])
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
