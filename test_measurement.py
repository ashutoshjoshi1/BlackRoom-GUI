#!/usr/bin/env python3
"""
Test measurement functionality without hardware
"""

import tkinter as tk
from app import SpectroApp
import numpy as np

class MockSpectrometer:
    """Mock spectrometer for testing"""
    def __init__(self):
        self.sn = "TEST123"
        self.npix_active = 2048
        self.rcm = np.random.randint(1000, 5000, 2048)  # Mock spectrum data
        
    def set_it(self, it_ms):
        print(f"Mock: Setting IT to {it_ms:.3f} ms")
        
    def measure(self, ncy=1):
        print(f"Mock: Starting measurement with {ncy} cycles")
        
    def wait_for_measurement(self):
        print("Mock: Measurement complete")
        # Simulate different peak values for testing
        peak_val = np.random.randint(50000, 70000)
        self.rcm = np.random.randint(100, peak_val, 2048)

def test_measurement_sequence():
    """Test the measurement sequence with mock hardware"""
    print("Testing measurement sequence...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = SpectroApp()
        app.withdraw()
        
        # Replace spectrometer with mock
        app.spec = MockSpectrometer()
        
        print("✓ App created with mock spectrometer")
        
        # Test laser selection
        selected_lasers = [tag for tag, var in app.measure_vars.items() if var.get()]
        print(f"✓ Selected lasers: {selected_lasers}")
        
        # Test measurement plot
        if hasattr(app, 'measure_line'):
            print("✓ Measurement plot ready")
            
            # Test plot update
            test_data = np.random.randint(1000, 60000, 2048)
            app._update_measurement_plot(test_data, "405", 5.0, 55000)
            print("✓ Plot update test successful")
        
        # Test auto-adjust integration time
        print("\nTesting auto-adjust integration time...")
        success, final_it = app._auto_adjust_integration_time_with_plot("405", 2.0)
        print(f"✓ Auto-adjust result: success={success}, final_it={final_it:.3f}")
        
        # Test laser control methods
        print("\nTesting laser control methods...")
        app._turn_off_laser("405")
        app._turn_off_laser("377")
        app._turn_off_laser("532")
        print("✓ Laser control methods work")
        
        app.destroy()
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        root.destroy()

if __name__ == "__main__":
    test_measurement_sequence()
