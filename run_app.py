#!/usr/bin/env python3
"""
Simple Python launcher for the DConfusion Streamlit app.
This ensures we use the correct Python interpreter with all dependencies.
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    packages = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'scipy']

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing

def install_dependencies(packages):
    """Install missing packages."""
    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--user'
        ] + packages)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("=" * 60)
    print("DConfusion Streamlit App Launcher")
    print("=" * 60)
    print(f"Using Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print()

    # Check dependencies
    missing = check_dependencies()

    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ")

        if response.lower() == 'y':
            if install_dependencies(missing):
                print("✅ Dependencies installed!")
            else:
                print("❌ Failed to install dependencies")
                print("Try manually: pip3 install " + ' '.join(missing))
                return 1
        else:
            print("Cannot run without dependencies.")
            print("Install manually: pip3 install " + ' '.join(missing))
            return 1

    print("✅ All dependencies found")
    print()
    print("🚀 Starting Streamlit app...")
    print("   Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'app', 'streamlit_app.py')

    # Run streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path
        ])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
