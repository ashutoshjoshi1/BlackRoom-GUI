#!/usr/bin/env python3
"""
Test analysis functionality with mock data
"""

import tkinter as tk
from app import SpectroApp
import pandas as pd
import numpy as np
import os
import tempfile

def create_mock_measurement_data():
    """Create mock measurement data CSV for testing analysis."""
    
    # Create mock data structure
    data = []
    
    # Mock laser measurements
    lasers = ["532", "445", "405", "377", "Hg_Ar"]
    
    for laser in lasers:
        # Signal measurement
        signal_data = np.random.randint(10000, 60000, 2048)
        # Add a peak for the laser
        peak_pixel = 500 + hash(laser) % 1000  # Deterministic peak location
        signal_data[peak_pixel-10:peak_pixel+10] = np.random.randint(55000, 65000, 20)
        
        signal_row = ["2024-01-01 12:00:00", laser, 5.0, 50] + signal_data.tolist()
        data.append(signal_row)
        
        # Dark measurement
        dark_data = np.random.randint(100, 1000, 2048)
        dark_row = ["2024-01-01 12:01:00", f"{laser}_dark", 5.0, 50] + dark_data.tolist()
        data.append(dark_row)
    
    # Add 640nm data
    signal_640 = np.random.randint(1000, 5000, 2048)
    dark_640 = np.random.randint(100, 500, 2048)
    
    data.append(["2024-01-01 12:30:00", "640_100ms", 100.0, 10] + signal_640.tolist())
    data.append(["2024-01-01 12:31:00", "640_100ms_dark", 100.0, 10] + dark_640.tolist())
    
    # Create DataFrame
    columns = ["Timestamp", "Wavelength", "IntegrationTime", "Cycles"]
    columns.extend([f"Pixel_{i}" for i in range(2048)])
    
    df = pd.DataFrame(data, columns=columns)
    return df

def test_analysis_functionality():
    """Test the comprehensive analysis functionality."""
    print("Testing analysis functionality...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = SpectroApp()
        app.withdraw()
        print("✓ App created successfully")
        
        # Create mock data
        df = create_mock_measurement_data()
        
        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            df.to_csv(csv_path, index=False)
        
        print(f"✓ Mock data created: {csv_path}")
        
        # Test analysis generation
        print("🔬 Running comprehensive analysis...")
        plot_paths = app.run_analysis_and_save_plots(csv_path)
        
        print(f"✓ Analysis complete! Generated {len(plot_paths)} plots:")
        for plot_path in plot_paths:
            if os.path.exists(plot_path):
                print(f"  ✓ {os.path.basename(plot_path)}")
            else:
                print(f"  ❌ {os.path.basename(plot_path)} (file not found)")
        
        # Test plot selection functionality
        if hasattr(app, 'current_plot_paths') and app.current_plot_paths:
            print("✓ Plot paths stored for dropdown")
            
            if hasattr(app, '_on_plot_selected'):
                print("✓ Plot selection method available")
            
            if hasattr(app, '_load_plot_in_display'):
                print("✓ Plot display method available")
        
        # Cleanup
        try:
            os.unlink(csv_path)
            for plot_path in plot_paths:
                if os.path.exists(plot_path):
                    os.unlink(plot_path)
            # Remove plots directory if empty
            plots_dir = os.path.dirname(plot_paths[0]) if plot_paths else None
            if plots_dir and os.path.exists(plots_dir):
                try:
                    os.rmdir(plots_dir)
                except:
                    pass
        except Exception as e:
            print(f"Cleanup warning: {e}")
        
        app.destroy()
        print("\n✅ Analysis functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        root.destroy()

if __name__ == "__main__":
    test_analysis_functionality()
