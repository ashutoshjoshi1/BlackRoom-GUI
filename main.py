# main.py - entrypoint
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import SpectroApp

if __name__ == "__main__":
    app = SpectroApp()
    app.mainloop()
