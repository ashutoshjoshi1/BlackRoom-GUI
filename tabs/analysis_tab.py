# Auto-generated from gui.py by splitter
from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os, time

def build(app):
    def _build_analysis_tab(self):
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


    def start_analysis_from_measure(self):
        """Start a measurement run for the lasers selected in the Measurement tab
        (analysis selection if available), and paint the resulting LSF/other plots in the Analysis tab.
        """
        if not app.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        tags = [t for t, v in getattr(self, 'analysis_vars', {}).items() if v.get()] or \
               [t for t, v in app.measure_vars.items() if v.get()]
        if not tags:
            messagebox.showwarning("Analysis", "No lasers selected to analyze.")
            return
        try:
            app.ana_ax1.clear(); app.ana_ax1.grid(True)
            app.ana_ax2.clear(); app.ana_ax2.grid(True)
            app.analysis_text.delete('1.0', 'end')
        except Exception:
            pass
        app.measure_running.set()
        def _runner():
            try:
                app._measure_sequence_thread(tags, None)
            finally:
                try:
                    app.nb.select(app.analysis_tab)
                except Exception:
                    pass
        app.measure_thread = threading.Thread(target=_runner, daemon=True)
        app.measure_thread.start()
    def run_analysis(self):
        if not app.data.rows:
            messagebox.showwarning("Analysis", "No measurement data available.")
            return
        df = app.data.to_dataframe()
        atype = app.analysis_type.get()
        tag = app.analysis_tag_entry.get().strip() or "Hg_Ar"
        app.analysis_text.delete("1.0", "end")
        app.ana_ax1.clear(); app.ana_ax1.grid(True)
        app.ana_ax2.clear(); app.ana_ax2.grid(True)

        try:
            if atype == "LSF":
                lsf = normalized_lsf_from_df(df, tag)
                if lsf is None:
                    raise RuntimeError("Could not compute LSF (missing rows or saturated).")
                x = np.arange(len(lsf))
                app.ana_ax1.set_title(f"LSF (normalized) - {tag}")
                app.ana_ax1.plot(x, lsf)
                # center/peak
                peak_pix = int(np.nanargmax(lsf))
                app.ana_ax2.set_title("Zoom near peak")
                lo = max(0, peak_pix - 50); hi = min(len(lsf), peak_pix + 50)
                app.ana_ax2.plot(np.arange(lo, hi), lsf[lo:hi])
                app.analysis_text.insert("end", f"LSF computed for {tag}\nPeak Pixel: {peak_pix}\n")

            elif atype == "Dispersion":
                # Use Hg_Ar (or tag) to find peaks, then match to known lines
                sig, _ = app.data.last_vectors_for(tag)
                if sig is None:
                    raise RuntimeError(f"No '{tag}' signal found.")
                # find prominent peaks
                peaks, _ = find_peaks(sig, height=np.nanmax(sig)*0.2, distance=5)
                peaks = np.sort(peaks)
                # attempt match with known lines
                sol = best_ordered_linear_match(peaks, KNOWN_HG_AR_NM, min_points=4)
                if not sol:
                    raise RuntimeError("Could not fit linear dispersion to known lines.")
                rmse, a, b, pix_sel, wl_sel = sol
                wl_pred = a * np.arange(len(sig)) + b
                app.ana_ax1.set_title("Dispersion Mapping (nm vs pixel)")
                app.ana_ax1.plot(np.arange(len(sig)), wl_pred, lw=1)
                app.ana_ax2.set_title("Peak Match")
                app.ana_ax2.plot(pix_sel, wl_sel, "o")
                app.analysis_text.insert("end", f"Dispersion fit: wl = {a:.6f}*pix + {b:.3f}\nRMSE: {rmse:.3f} nm\n")
                app.analysis_text.insert("end", f"Used peaks: {pix_sel.tolist()}\nMapped to: {wl_sel.tolist()}\n")

            elif atype == "Stray Light":
                lsf = normalized_lsf_from_df(df, tag)
                if lsf is None:
                    raise RuntimeError("Could not compute LSF for stray light.")
                peak_pix = int(np.nanargmax(lsf))
                metrics = stray_light_metrics(lsf, peak_pix, ib_half=IB_REGION_HALF)
                app.ana_ax1.set_title("LSF (normalized)")
                app.ana_ax1.plot(np.arange(len(lsf)), lsf)
                app.ana_ax2.set_title("Bands")
                lo = max(0, peak_pix-50); hi = min(len(lsf), peak_pix+50)
                app.ana_ax2.plot(np.arange(lo, hi), lsf[lo:hi])
                app.analysis_text.insert("end", "Stray Light Metrics:\n")
                for k, v in metrics.items():
                    app.analysis_text.insert("end", f"  {k}: {v:.6g}\n")

            elif atype == "Resolution":
                # FWHM near a peak; if dispersion fit known, you can convert to nm
                tag_use = tag
                sig, dark = app.data.last_vectors_for(tag_use)
                if sig is None or dark is None:
                    raise RuntimeError("Missing signal/dark for resolution.")
                y = (sig - dark).astype(float)
                y -= np.nanmin(y)
                if np.nanmax(y) <= 0:
                    raise RuntimeError("Flat/invalid spectrum for resolution.")
                y /= np.nanmax(y)
                xpix = np.arange(len(y))
                fwhm_pix = compute_fwhm(xpix, y)
                # try to estimate nm per pixel via simple two-line fit if available
                peaks, _ = find_peaks(y, height=0.2, distance=5)
                nm_per_pix = np.nan
                if len(peaks) >= 4:
                    sol = best_ordered_linear_match(peaks, KNOWN_HG_AR_NM, min_points=4)
                    if sol:
                        _, a, b, _, _ = sol
                        nm_per_pix = a
                app.ana_ax1.set_title("Signal - Dark (normalized)")
                app.ana_ax1.plot(xpix, y)
                app.ana_ax2.set_title("Peak Zoom")
                if len(peaks) > 0:
                    p0 = int(peaks[np.argmax(y[peaks])])
                    lo = max(0, p0-50); hi = min(len(y), p0+50)
                    app.ana_ax2.plot(np.arange(lo, hi), y[lo:hi])
                txt = f"FWHM ≈ {fwhm_pix:.3f} pixels"
                if np.isfinite(nm_per_pix):
                    txt += f"  (~{fwhm_pix*nm_per_pix:.3f} nm with slope {nm_per_pix:.6f} nm/pixel)"
                app.analysis_text.insert("end", txt + "\n")

            else:
                raise RuntimeError(f"Unknown analysis type '{atype}'")

        except Exception as e:
            app._post_error("Analysis Error", e)

        app.ana_canvas1.draw()
        app.ana_canvas2.draw()

    def export_analysis_plots(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = filedialog.askdirectory(title="Select folder to save analysis plots")
        if not base:
            return
        try:
            p1 = os.path.join(base, f"analysis_plot1_{ts}.png")
            p2 = os.path.join(base, f"analysis_plot2_{ts}.png")
            app.ana_fig1.savefig(p1, dpi=150, bbox_inches="tight")
            app.ana_fig2.savefig(p2, dpi=150, bbox_inches="tight")
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
            txt = app.analysis_text.get("1.0", "end").strip()
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt + "\n")
            messagebox.showinfo("Export Summary", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export Summary", str(e))

    # ------------------ Setup Tab -----------------------

    def _build_setup_tab(self):
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

    def refresh_ports(self):
        ports = list(serial.tools.list_ports.comports())
        names = [p.device for p in ports]
        if names:
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, names[0])
            if len(names) > 1:
                app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, names[1])
            if len(names) > 2:
                app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, names[2])
            messagebox.showinfo("Ports", "Populated with first detected ports.\nAdjust as needed.")
        else:
            messagebox.showwarning("Ports", "No serial ports detected.")

    def test_com_connect(self):
        app._update_ports_from_ui()
        ok_obis = app.lasers.obis.open()
        app.obis_status.config(foreground=("green" if ok_obis else "red"))
        ok_cube = app.lasers.cube.open()
        app.cube_status.config(foreground=("green" if ok_cube else "red"))
        ok_relay = app.lasers.relay.open()
        app.relay_status.config(foreground=("green" if ok_relay else "red"))
        # Close after test to free ports (or keep open if you prefer)
        time.sleep(0.2)
        if ok_obis: app.lasers.obis.close()
        if ok_cube: app.lasers.cube.close()
        if ok_relay: app.lasers.relay.close()

    def browse_dll(self):
        path = filedialog.askopenfilename(
            title="Select avaspecx64.dll", filetypes=[("DLL", "*.dll"), ("All files", "*.*")])
        if path:
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, path)

    def connect_spectrometer(self):
        try:
            dll = app.dll_entry.get().strip()
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
            # pick first SN if available, else rely on wrapper default
            try:
                sers = []
                for i in range(ndev):
                    ident = getattr(infos, f"a{i}")
                    sn = ident.SerialNumber
                    if isinstance(sn, (bytes, bytearray)):
                        sn = sn.decode("utf-8", errors="ignore")
                    sers.append(sn)
                if sers:
                    ava.sn = sers[0]
            except Exception:
                pass

            ava.connect()
            app.spec = ava
            app.sn = getattr(ava, "sn", "Unknown")
            app.data.serial_number = app.sn
            app.npix = getattr(ava, "npix_active", app.npix)
            app.data.npix = app.npix

            app.spec_status.config(text=f"Connected: {app.sn}", foreground="green")
            messagebox.showinfo("Spectrometer", f"Connected to SN={app.sn}")
        except Exception as e:
            app.spec = None
            app.spec_status.config(text="Disconnected", foreground="red")
            app._post_error("Spectrometer Connect", e)

    def disconnect_spectrometer(self):
        try:
            app.stop_live()
            if app.spec:
                try:
                    app.spec.disconnect()
                except Exception:
                    pass
            app.spec = None
            app.spec_status.config(text="Disconnected", foreground="red")
        except Exception as e:
            app._post_error("Spectrometer Disconnect", e)

    def _update_ports_from_ui(self):
        app.hw.com_ports["OBIS"] = app.obis_entry.get().strip() or DEFAULT_COM_PORTS["OBIS"]
        app.hw.com_ports["CUBE"] = app.cube_entry.get().strip() or DEFAULT_COM_PORTS["CUBE"]
        app.hw.com_ports["RELAY"] = app.relay_entry.get().strip() or DEFAULT_COM_PORTS["RELAY"]
        app.lasers.configure_ports(app.hw.com_ports)

    def _get_power(self, tag: str) -> float:
        try:
            e = app.power_entries.get(tag)
            if e is None:
                return DEFAULT_LASER_POWERS.get(tag, 0.01)
            return float(e.get().strip())
        except:
            return DEFAULT_LASER_POWERS.get(tag, 0.01)

    def save_settings(self):
        app._update_ports_from_ui()
        app.hw.dll_path = app.dll_entry.get().strip()
        for tag, e in app.power_entries.items():
            try:
                app.hw.laser_power[tag] = float(e.get().strip())
            except:
                pass
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "dll_path": app.hw.dll_path,
                    "com_ports": app.hw.com_ports,
                    "laser_power": app.hw.laser_power
                }, f, indent=2)
            messagebox.showinfo("Settings", f"Saved to {SETTINGS_FILE}")
        except Exception as e:
            messagebox.showerror("Settings", str(e))

    def load_settings_into_ui(self):
        if not os.path.isfile(SETTINGS_FILE):
            # init defaults
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, app.hw.dll_path)
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, DEFAULT_COM_PORTS["OBIS"])
            app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, DEFAULT_COM_PORTS["CUBE"])
            app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, DEFAULT_COM_PORTS["RELAY"])
            for tag, e in app.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(DEFAULT_LASER_POWERS.get(tag, 0.01)))
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, obj.get("dll_path", ""))
            cp = obj.get("com_ports", DEFAULT_COM_PORTS)
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, cp.get("OBIS", DEFAULT_COM_PORTS["OBIS"]))
            app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, cp.get("CUBE", DEFAULT_COM_PORTS["CUBE"]))
            app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, cp.get("RELAY", DEFAULT_COM_PORTS["RELAY"]))
            lp = obj.get("laser_power", DEFAULT_LASER_POWERS)
            for tag, e in app.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(lp.get(tag, DEFAULT_LASER_POWERS.get(tag, 0.01))))
        except Exception as e:
            messagebox.showerror("Load Settings", str(e))

    # ------------------ General helpers ------------------

    def _post_error(self, title: str, ex: Exception):
        tb = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f"[{title}] {ex}\n{tb}", file=sys.stderr)
        app.after(0, lambda: messagebox.showerror(title, str(ex)))

    def on_close(self):
        try:
            app.stop_live()
            app.stop_measure()
            if app.spec:
                try: app.spec.disconnect()
                except: pass
            for dev in [app.lasers.obis, app.lasers.cube, app.lasers.relay]:
                try: dev.close()
                except: pass
        finally:
            app.destroy()


    if __name__ == "__main__":
    app = SpectroApp()
    app.mainloop()
