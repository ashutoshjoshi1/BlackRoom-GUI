# Live View tab
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def build(app):
    # Layout frames
    left = ttk.Frame(app.live_tab)
    left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

    right = ttk.Frame(app.live_tab)
    right.pack(side="right", fill="y", padx=8, pady=8)

    # Figure & axes
    app.live_fig = Figure(figsize=(6, 4), dpi=100)
    app.live_ax = app.live_fig.add_subplot(111)
    app.live_ax.set_title("Live Spectrum")
    app.live_ax.set_xlabel("Pixel")
    app.live_ax.set_ylabel("Counts")
    (app.live_line,) = app.live_ax.plot([], [], lw=1)

    # Canvas + toolbar
    app.live_canvas = FigureCanvasTkAgg(app.live_fig, master=left)
    app.live_canvas.draw()
    app.live_canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(app.live_canvas, left)

    # Limit locking flag (user zoom/pan should hold the limits)
    app.live_limits_locked = False
    def _lock_on_user_action(event):
        try:
            if event.inaxes == app.live_ax:
                app.live_limits_locked = True
        except Exception:
            pass
    # Lock when user clicks or scroll-zooms
    app.live_canvas.mpl_connect("button_press_event", _lock_on_user_action)
    app.live_canvas.mpl_connect("scroll_event", _lock_on_user_action)

    # Controls
    ttk.Button(right, text="Reset View", command=app._live_reset_view).pack(anchor="w", pady=(0, 10))

    # Laser toggles
    app.laser_vars = {}
    for tag in ["405", "445", "488", "377", "517", "532", "Hg_Ar"]:
        var = tk.BooleanVar(value=False)
        text = f"{tag} nm" if tag.isdigit() else tag
        ttk.Checkbutton(
            right, text=text, variable=var,
            command=lambda t=tag, v=var: app.toggle_laser(t, v.get())
        ).pack(anchor="w")
        app.laser_vars[tag] = var

    ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)

    # Start/Stop buttons
    app.live_start_btn = ttk.Button(right, text="Start Live", command=app.start_live)
    app.live_stop_btn  = ttk.Button(right, text="Stop Live",  command=app.stop_live)
    app.live_start_btn.pack(anchor="w", pady=2)
    app.live_stop_btn.pack(anchor="w", pady=2)

    # Integration time control
    int_frame = ttk.Frame(right)
    int_frame.pack(anchor="w", pady=8)
    ttk.Label(int_frame, text="Integration (ms):").pack(side="left")
    app.it_entry = ttk.Entry(int_frame, width=8)
    app.it_entry.insert(0, "2.4")        # default 2.4 ms
    app.it_entry.pack(side="left", padx=4)
    app.apply_it_btn = ttk.Button(int_frame, text="Apply", command=app.apply_it)
    app.apply_it_btn.pack(side="left")
