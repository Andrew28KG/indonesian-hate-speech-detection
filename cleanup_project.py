#!/usr/bin/env python3
"""
Project Cleanup Script
Removes unneeded files from the Indonesian Hate Speech Detection project
"""

import os
import shutil
from pathlib import Path

def main():
    print("🧹 Indonesian Hate Speech Detection - Project Cleanup")
    print("=" * 60)
    
    # Files to remove
    files_to_remove = [
        "run_app.py",           # Redundant launcher
        "run_demo.py",          # Redundant launcher  
        "test_app.py",          # Development test script
        "test_setup.py",        # Setup test script
        "validate_project.py",  # Project validation script
        "process_data.py",      # Data processing script (data already processed)
        "fix_unicode_error.py", # Unicode fix utility (no longer needed)
        "rebuild_models.py",    # Model rebuild script (models already built)
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "outputs/models",       # Empty directory
        "notebooks/__pycache__", # Python cache
    ]
    
    print("\n📋 Files to be removed:")
    for file in files_to_remove:
        print(f"  🗑️ {file}")
    
    print("\n📋 Directories to be removed:")
    for directory in dirs_to_remove:
        print(f"  🗑️ {directory}/")
    
    # Ask for confirmation
    print("\n⚠️  WARNING: This will permanently delete the above files!")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    print("\n🧹 Starting cleanup...")
    
    # Remove files
    removed_files = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  ✅ Removed {file_path}")
                removed_files += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"  ⚠️ {file_path} not found")
    
    # Remove directories
    removed_dirs = 0
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"  ✅ Removed {dir_path}/")
                removed_dirs += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {dir_path}: {e}")
        else:
            print(f"  ⚠️ {dir_path} not found")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   📁 Removed {removed_files} files")
    print(f"   📁 Removed {removed_dirs} directories")
    
    print("\n🎯 Remaining project structure:")
    essential_files = [
        "README.md",
        "requirements.txt", 
        "LICENSE",
        "setup.py",
        "start_app.py",          # Main launcher
        "app/streamlit_demo.py", # Main Streamlit app
        "notebooks/gui.py",      # GUI application
        "models/",               # Trained models
        "data/",                 # Data files
        "utils/",                # Utility modules
        "IndonesianAbusiveWords/" # Dictionary
    ]
    
    for item in essential_files:
        if os.path.exists(item):
            print(f"  ✅ {item}")
        else:
            print(f"  ❌ {item} - MISSING")
    
    print("\n🚀 Your project is now clean and ready for distribution!")
    print("\n📝 To use the project:")
    print("   🌐 Streamlit app: streamlit run app/streamlit_demo.py")
    print("   🖥️ GUI app: cd notebooks && python gui.py")
    print("   📊 Notebooks: jupyter notebook")

if __name__ == "__main__":
    main() 