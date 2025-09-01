#!/usr/bin/env python3
"""
FPL Analytics App - Refactored Entry Point
"""

from core.app_controller import FPLAppController

def main():
    """Main entry point for the FPL Analytics App"""
    app = FPLAppController()
    app.run()

if __name__ == "__main__":
    main()
