#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPL Analytics App - Health Check Script
Comprehensive health monitoring and validation
"""

import os
import sys
import subprocess
import importlib
import importlib.util
from pathlib import Path
import requests
from datetime import datetime

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Fix for Windows Unicode issues
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass
    
    # Force UTF-8 encoding for output
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

def safe_print(text):
    """Print text with fallback for Unicode issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove emojis and special characters for Windows compatibility
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)

class HealthChecker:
    """Comprehensive health checker for FPL Analytics App"""
    
    def __init__(self):
        self.workspace_dir = Path.cwd()
        self.errors = []
        self.warnings = []
        
    def run_health_check(self):
        """Execute comprehensive health check"""
        safe_print("FPL Analytics App - Health Check")
        safe_print("=" * 50)
        safe_print(f"Check started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        safe_print(f"Workspace: {self.workspace_dir}")
        safe_print("=" * 50)
        
        # Infrastructure checks
        self.check_directory_structure()
        self.check_required_files()
        self.check_python_dependencies()
        
        # Functionality checks
        self.check_imports()
        self.check_fpl_api_access()
        self.check_streamlit_compatibility()
        
        # Generate report
        self.generate_health_report()
        
        return len(self.errors) == 0
    
    def check_directory_structure(self):
        """Check if required directories exist"""
        safe_print("Checking directory structure...")
        
        required_dirs = [
            "config", "services", "pages", "utils", 
            "tests", "scripts", "logs"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.workspace_dir / dir_name
            if not dir_path.exists():
                self.errors.append(f"Missing directory: {dir_name}")
            else:
                safe_print(f"  Directory OK: {dir_name}")
    
    def check_required_files(self):
        """Check if required files exist"""
        safe_print("Checking required files...")
        
        required_files = [
            "config/enhanced_app_config.py",
            "services/fpl_data_service.py",
            "pages/player_analysis_page.py",
            "pages/fixture_analysis_page.py", 
            "pages/my_team_page.py",
            "utils/enhanced_error_handling.py",
            "utils/caching.py",
            "utils/logging.py",
            "requirements_production.txt"
        ]
        
        for file_path in required_files:
            full_path = self.workspace_dir / file_path
            if not full_path.exists():
                self.errors.append(f"Missing file: {file_path}")
            else:
                safe_print(f"  File OK: {file_path}")
    
    def check_python_dependencies(self):
        """Check Python dependencies"""
        safe_print("Checking Python dependencies...")
        
        required_packages = [
            "streamlit", "pandas", "numpy", "requests", 
            "plotly", "seaborn", "matplotlib"
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                safe_print(f"  Package OK: {package}")
            except ImportError:
                self.errors.append(f"Missing package: {package}")
    
    def check_imports(self):
        """Check if core modules can be imported"""
        safe_print("Checking module imports...")
        
        # Simple file existence check instead of complex imports
        modules_to_check = [
            "services/fpl_data_service.py",
            "pages/player_analysis_page.py",
            "pages/fixture_analysis_page.py",
            "pages/my_team_page.py",
            "config/enhanced_app_config.py"
        ]
        
        for module_file in modules_to_check:
            file_path = self.workspace_dir / module_file
            if file_path.exists():
                # Basic syntax check by trying to compile
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(file_path), 'exec')
                    safe_print(f"  Module OK: {module_file}")
                except SyntaxError as e:
                    self.errors.append(f"Syntax error in {module_file}: {str(e)}")
                except Exception as e:
                    self.warnings.append(f"Could not validate {module_file}: {str(e)}")
            else:
                self.errors.append(f"Module file not found: {module_file}")
    
    def check_fpl_api_access(self):
        """Check FPL API accessibility"""
        safe_print("Checking FPL API access...")
        
        try:
            # Skip SSL verification for corporate networks
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            response = requests.get(
                "https://fantasy.premierleague.com/api/bootstrap-static/",
                timeout=10,
                verify=False
            )
            if response.status_code == 200:
                safe_print("  FPL API access OK")
            else:
                self.warnings.append(f"FPL API returned status {response.status_code}")
        except Exception as e:
            self.warnings.append(f"FPL API access failed: {str(e)}")
    
    def check_streamlit_compatibility(self):
        """Check Streamlit compatibility"""
        safe_print("Checking Streamlit compatibility...")
        
        try:
            import streamlit as st
            safe_print(f"  Streamlit version: {st.__version__}")
            safe_print("  Streamlit compatibility OK")
        except Exception as e:
            self.errors.append(f"Streamlit compatibility issue: {str(e)}")
    
    def generate_health_report(self):
        """Generate health check report"""
        safe_print("\n" + "=" * 50)
        safe_print("HEALTH CHECK REPORT")
        safe_print("=" * 50)
        
        if not self.errors and not self.warnings:
            safe_print("STATUS: ALL SYSTEMS HEALTHY")
            safe_print("The FPL Analytics App is ready to run!")
        else:
            if self.errors:
                safe_print(f"ERRORS FOUND: {len(self.errors)}")
                for error in self.errors:
                    safe_print(f"  ERROR: {error}")
            
            if self.warnings:
                safe_print(f"WARNINGS: {len(self.warnings)}")
                for warning in self.warnings:
                    safe_print(f"  WARNING: {warning}")
            
            if self.errors:
                safe_print("\nSTATUS: SYSTEM HAS ISSUES")
                safe_print("Please resolve errors before running the app.")
            else:
                safe_print("\nSTATUS: SYSTEM OPERATIONAL WITH WARNINGS")
                safe_print("App should work but may have reduced functionality.")
        
        safe_print(f"\nCheck completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        safe_print("=" * 50)


def main():
    """Main entry point for health check"""
    checker = HealthChecker()
    success = checker.run_health_check()
    
    if success:
        safe_print("\nHealth check passed!")
        sys.exit(0)
    else:
        safe_print("\nHealth check failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
