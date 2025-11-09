#!/usr/bin/env python3
"""
Simple test to see which packages are available.
Run this to diagnose import issues.
"""

import sys

print("=" * 60)
print("Python Environment Test")
print("=" * 60)
print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"\nPython path:")
for p in sys.path:
    print(f"  - {p}")

print("\n" + "=" * 60)
print("Testing Package Imports")
print("=" * 60 + "\n")

packages = [
    ('streamlit', 'streamlit'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('scipy', 'scipy'),
    ('dconfusion', 'dconfusion'),
]

all_ok = True

for module, display_name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown version')
        print(f"✅ {display_name:15s} - {version}")
    except ImportError as e:
        print(f"❌ {display_name:15s} - NOT FOUND")
        print(f"   Error: {e}")
        all_ok = False
    except Exception as e:
        print(f"⚠️  {display_name:15s} - ERROR: {e}")
        all_ok = False

print("\n" + "=" * 60)

if all_ok:
    print("✅ ALL PACKAGES AVAILABLE!")
    print("\nYou can run the app with:")
    print(f"  {sys.executable} -m streamlit run app/streamlit_app.py")
else:
    print("❌ SOME PACKAGES MISSING")
    print("\nTo install missing packages:")
    print(f"  {sys.executable} -m pip install --user streamlit pandas numpy matplotlib scipy")

print("=" * 60)
