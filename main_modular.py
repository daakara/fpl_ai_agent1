"""
Refactored FPL Analytics Application - Main Entry Point
Uses modular architecture with separate components for better maintainability
"""
import streamlit as st
import pandas as pd
from core.app_controller import FPLAppController


def main():
    """Main application entry point"""
    try:
        # Initialize and run the application controller
        app = FPLAppController()
        app.run()
        
    except Exception as e:
        st.error(f"Application startup error: {str(e)}")
        st.write("Please refresh the page or check the console for details.")


if __name__ == "__main__":
    main()