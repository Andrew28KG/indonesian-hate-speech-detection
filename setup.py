#!/usr/bin/env python3
"""
Indonesian Hate Speech Detection - Setup Script
This script helps set up the environment and download required dependencies.
"""

import os
import sys
import subprocess
import importlib.util

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def setup_nltk():
    """Download required NLTK data"""
    try:
        import nltk
        print("📦 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/processed',
        'data/interim',
        'models',
        'logs',
        'outputs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def install_requirements():
    """Install requirements from requirements.txt"""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    key_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'streamlit',
        'Sastrawi',
        'nltk'
    ]
    
    print("🔍 Verifying installation...")
    missing_packages = []
    
    for package in key_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"✅ {package}")
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        return False
    
    print("✅ All key packages installed successfully!")
    return True

def setup_git_hooks():
    """Set up Git hooks (optional)"""
    if not os.path.exists('.git'):
        print("ℹ️  Not a Git repository, skipping Git hooks setup")
        return True
    
    # This is optional and can be extended
    print("ℹ️  Git hooks setup can be added here if needed")
    return True

def main():
    """Main setup function"""
    print("🇮🇩 Indonesian Hate Speech Detection - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("⚠️  Failed to install requirements. Please install manually using:")
        print("   pip install -r requirements.txt")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup NLTK
    print("\n📚 Setting up NLTK...")
    setup_nltk()
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    if not verify_installation():
        print("⚠️  Some packages may not be installed correctly.")
        print("Please check the installation and try running:")
        print("   pip install -r requirements.txt")
    
    # Setup Git hooks (optional)
    setup_git_hooks()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Run 'python run_demo.py' to start the demo")
    print("2. Open notebooks/ folder to explore the analysis")
    print("3. Check README.md for detailed documentation")
    print("\n🚀 Happy analyzing!")

if __name__ == "__main__":
    main()
