# Auto-generated from gui.py splitter
from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os, time

def build(app):
        left = ttk.Frame(app.live_tab)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        right = ttk.Frame(app.live_tab)
        right.pack(side="right", fill="y", padx=8, pady=8)

        # Matplotlib figure
        app.live_fig = Figure(figsize=(8, 5), dpi=100)
        app.live_ax = app.live_fig.add_subplot(111)
        app.live_ax.set_title("Live Spectrum")
        app.live_ax.set_xlabel("Pixel")
        app.live_ax.set_ylabel("Counts")
        app.live_line, = app.live_ax.plot([], [], lw=1, label="Signal")
        app.live_ax.grid(True)
        app.live_ax.legend(loc="upper right")

        app.live_canvas = FigureCanvasTkAgg(app.live_fig, master=left)
        app.live_canvas.draw()
        app.live_canvas.get_tk_widget().pack(fill="both", expand=True)

        app.live_toolbar = NavigationToolbar2Tk(app.live_canvas, left)

        # track zoom/pan interactions
        app.live_limits_locked = False
        app._live_mouse_down = False

        def _on_press(event):
            app._live_mouse_down = True

        def _on_release(event):
       # When user releases after an interaction on axes, lock current limits
            if event.inaxes is not None:
                app.live_limits_locked = True
app._live_mouse_down = False

app.live_canvas.mpl_connect("button_press_event", _on_press)
app.live_canvas.mpl_connect("button_release_event", _on_release)

        # Controls
ttk.Label(right, text="Integration Time (ms):").pack(anchor="w")
ttk.Button(right, text="Reset Zoom", command=app._live_reset_view).pack(anchor="w", pady=(6, 0))
app.it_entry = ttk.Entry(right, width=12)
app.it_entry.insert(0, "2.4")
app.it_entry.pack(anchor="w", pady=(0, 10))
app.apply_it_btn = ttk.Button(right, text="Apply IT", command=app.apply_it)
app.apply_it_btn.pack(anchor="w", pady=(0, 10))

ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)

ttk.Label(right, text="Laser Controls").pack(anchor="w")
app.laser_vars = {}
for tag in ["405", "445", "488", "377", "517", "532", "Hg_Ar"]:
       var = tk.BooleanVar(value=False)
       btn = ttk.Checkbutton(
           right, text=f"{tag} nm", variable=var,
           command=lambda t=tag, v=var: app.toggle_laser(t, v.get()))
       btn.pack(anchor="w")
       app.laser_vars[tag] = var

       ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)

       app.live_start_btn = ttk.Button(right, text="Start Live", command=app.start_live)
       app.live_stop_btn = ttk.Button(right, text="Stop Live", command=app.stop_live)
       app.live_start_btn.pack(anchor="w", pady=2)
       app.live_stop_btn.pack(anchor="w", pady=2)
