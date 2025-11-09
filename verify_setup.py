#!/usr/bin/env python3
"""
Verification script for DConfusion setup.
Checks that all dependencies are installed correctly.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - Install with: pip3 install {package_name}")
        return False

def main():
    print("=" * 60)
    print("DConfusion Setup Verification")
    print("=" * 60)
    print(f"\nPython interpreter: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print("\nChecking dependencies...\n")

    # Core dependencies
    all_good = True
    all_good &= check_import("numpy")
    all_good &= check_import("matplotlib")
    all_good &= check_import("scipy")

    # App dependencies
    all_good &= check_import("pandas")
    all_good &= check_import("streamlit")

    # Package itself
    print("\nChecking DConfusion package...\n")
    try:
        from dconfusion import DConfusion
        print("✓ dconfusion package")

        # Quick functionality test
        cm = DConfusion(10, 2, 3, 15)
        acc = cm.get_accuracy()
        print(f"✓ DConfusion works (test accuracy: {acc:.4f})")
        all_good &= True
    except Exception as e:
        print(f"✗ dconfusion - Error: {e}")
        all_good = False

    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed! You're ready to go.")
        print("\nTo run the web app:")
        print(f"  {sys.executable} -m streamlit run app/streamlit_app.py")
        print("\nOr use the convenience script:")
        print("  ./run_app.sh")
    else:
        print("❌ Some dependencies are missing.")
        print("\nTo install all dependencies:")
        print(f"  {sys.executable} -m pip install -r requirements-app.txt")
        print("\nFor more help, see TROUBLESHOOTING.md")
    print("=" * 60)

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
