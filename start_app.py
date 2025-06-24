#!/usr/bin/env python3
"""
Start the Indonesian Hate Speech Detection Streamlit App

This script ensures the app starts from the correct directory and with proper paths.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Streamlit app"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    app_file = project_root / "app" / "streamlit_demo.py"
    
    # Check if the app file exists
    if not app_file.exists():
        print(f"❌ Error: {app_file} not found!")
        sys.exit(1)
    
    print("🇮🇩 Starting Indonesian Hate Speech Detection App...")
    print(f"📁 Project root: {project_root}")
    print(f"📄 App file: {app_file}")
    print("🌐 The app will open in your default browser...")
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Run streamlit from the project root
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        print(f"🔧 Running: {' '.join(cmd)}")
        print("🚀 Starting Streamlit server...")
        print("📱 Access the app at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 