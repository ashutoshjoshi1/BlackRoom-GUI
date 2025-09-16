
# Setup tab
import tkinter as tk
from tkinter import ttk

def build(app):
    frame = ttk.Frame(app.setup_tab)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Spectrometer
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

    # Ports
    ports = ttk.LabelFrame(frame, text="Serial Ports")
    ports.pack(fill="x", padx=6, pady=6)

    ttk.Label(ports, text="OBIS (multi-channel) COM:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
    app.obis_entry = ttk.Entry(ports, width=18)
    app.obis_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)

    ttk.Label(ports, text="CUBE 377 nm COM:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
    app.cube_entry = ttk.Entry(ports, width=18)
    app.cube_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)

    ttk.Label(ports, text="Relay COM:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
    app.relay_entry = ttk.Entry(ports, width=18)
    app.relay_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)

    ttk.Button(ports, text="Refresh Ports", command=app.refresh_ports).grid(row=0, column=2, padx=6)
    ttk.Button(ports, text="All Lasers OFF", command=app._all_off_on_start).grid(row=1, column=2, padx=6)

    # Power presets
    power_group = ttk.LabelFrame(frame, text="Laser Power Presets (Watts for OBIS, mW for 377nm)")
    power_group.pack(fill="x", padx=6, pady=6)

    app.power_entries = {}
    row = 0
    for tag in ["405", "445", "488", "640", "377", "517", "532", "Hg_Ar"]:
        ttk.Label(power_group, text=f"{tag} power:").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        e = ttk.Entry(power_group, width=12)
        try:
            default_val = str(DEFAULT_LASER_POWERS.get(tag, 0.01))
        except Exception:
            default_val = "0.01"
        e.insert(0, default_val)
        e.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        app.power_entries[tag] = e
        row += 1

    # Save/Load
    save_group = ttk.Frame(frame)
    save_group.pack(fill="x", padx=6, pady=8)
    ttk.Button(save_group, text="Save Settings", command=app.save_settings).pack(side="left")
    ttk.Button(save_group, text="Load Settings", command=app.load_settings_into_ui).pack(side="left", padx=6)
