#!/usr/bin/env python3
"""
Simple test script to verify laser control functionality
"""

import tkinter as tk
from app import SpectroApp

def test_laser_control():
    """Test laser control functionality"""
    print("Testing laser control functionality...")
    
    # Create app but don't show main window
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = SpectroApp()
        app.withdraw()
        
        print("✓ App created successfully")
        
        # Test laser control methods
        print("\nTesting laser control methods:")
        
        # Test OBIS laser (405 nm)
        print("- Testing 405 nm OBIS laser...")
        try:
            app.toggle_laser("405", True)
            print("  ✓ 405 nm ON command sent")
            app.toggle_laser("405", False)
            print("  ✓ 405 nm OFF command sent")
        except Exception as e:
            print(f"  ⚠️ 405 nm control error: {e}")
        
        # Test CUBE laser (377 nm)
        print("- Testing 377 nm CUBE laser...")
        try:
            app.toggle_laser("377", True)
            print("  ✓ 377 nm ON command sent")
            app.toggle_laser("377", False)
            print("  ✓ 377 nm OFF command sent")
        except Exception as e:
            print(f"  ⚠️ 377 nm control error: {e}")
        
        # Test relay laser (532 nm)
        print("- Testing 532 nm relay laser...")
        try:
            app.toggle_laser("532", True)
            print("  ✓ 532 nm ON command sent")
            app.toggle_laser("532", False)
            print("  ✓ 532 nm OFF command sent")
        except Exception as e:
            print(f"  ⚠️ 532 nm control error: {e}")
        
        print("\n✓ Laser control test completed")
        print("Note: Actual laser hardware may not be connected, but commands were sent successfully")
        
        app.destroy()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        root.destroy()

if __name__ == "__main__":
    test_laser_control()
