#!/usr/bin/env python3
"""
Deployment script for FPL Analytics App
"""
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def run_tests():
    """Run test suite"""
    print("🧪 Running tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    return result.returncode == 0


def start_app():
    """Start the application"""
    print("🚀 Starting FPL Analytics App...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main_refactored.py"])


def main():
    """Main deployment process"""
    print("🚀 FPL Analytics App - Deployment")
    print("=" * 40)
    
    # Install dependencies
    install_dependencies()
    
    # Run tests
    if run_tests():
        print("✅ All tests passed!")
        
        # Start application
        start_app()
    else:
        print("❌ Tests failed! Please fix issues before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
