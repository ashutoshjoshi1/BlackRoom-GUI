# Auto-generated from gui.py splitter
from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os, time

def build(app):
        frame = ttk.Frame(app.setup_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Spectrometer block
        spec_group = ttk.LabelFrame(frame, text="Spectrometer")
        spec_group.pack(fill="x", padx=6, pady=6)

        ttk.Label(spec_group, text="DLL Path (avaspecx64.dll):").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        app.dll_entry = ttk.Entry(spec_group, width=60)
        app.dll_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(spec_group, text="Browse", command=app.browse_dll).grid(row=0, column=2, padx=4, pady=4)

        app.spec_status = ttk.Label(spec_group, text="Disconnected", foreground="red")
        app.spec_status.grid(row=0, column=3, padx=10)

        ttk.Button(spec_group, text="Connect", command=app.connect_spectrometer).grid(row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(spec_group, text="Disconnect", command=app.disconnect_spectrometer).grid(row=1, column=2, padx=4, pady=4, sticky="w")

        # COM ports
        ports_group = ttk.LabelFrame(frame, text="COM Port Configuration")
        ports_group.pack(fill="x", padx=6, pady=6)

        ttk.Label(ports_group, text="OBIS:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(ports_group, text="CUBE:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(ports_group, text="RELAY:").grid(row=2, column=0, sticky="e", padx=4, pady=4)

        app.obis_entry = ttk.Entry(ports_group, width=12)
        app.cube_entry = ttk.Entry(ports_group, width=12)
        app.relay_entry = ttk.Entry(ports_group, width=12)
        app.obis_entry.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        app.cube_entry.grid(row=1, column=1, padx=4, pady=4, sticky="w")
        app.relay_entry.grid(row=2, column=1, padx=4, pady=4, sticky="w")

        ttk.Button(ports_group, text="Refresh Ports", command=app.refresh_ports).grid(row=0, column=2, padx=6)
        ttk.Button(ports_group, text="Test Connect", command=app.test_com_connect).grid(row=1, column=2, padx=6)

        app.obis_status = ttk.Label(ports_group, text="●", foreground="red")
        app.cube_status = ttk.Label(ports_group, text="●", foreground="red")
        app.relay_status = ttk.Label(ports_group, text="●", foreground="red")
        app.obis_status.grid(row=0, column=3, padx=4)
        app.cube_status.grid(row=1, column=3, padx=4)
        app.relay_status.grid(row=2, column=3, padx=4)

        # Laser power config
        power_group = ttk.LabelFrame(frame, text="Laser Power Configuration")
        power_group.pack(fill="x", padx=6, pady=6)

        app.power_entries: Dict[str, ttk.Entry] = {}
        row = 0
        for tag in ["405", "445", "488", "640", "377", "517", "532", "Hg_Ar"]:
            ttk.Label(power_group, text=f"{tag} nm power:").grid(row=row, column=0, sticky="e", padx=4, pady=2)
e = ttk.Entry(power_group, width=12)
e.insert(0, str(DEFAULT_LASER_POWERS.get(tag, 0.01)))
e.grid(row=row, column=1, sticky="w", padx=4, pady=2)
app.power_entries[tag] = e
row += 1

        # Save/Load
save_group = ttk.Frame(frame)
save_group.pack(fill="x", padx=6, pady=8)
ttk.Button(save_group, text="Save Settings", command=app.save_settings).pack(side="left")
ttk.Button(save_group, text="Load Settings", command=app.load_settings_into_ui).pack(side="left", padx=6)
