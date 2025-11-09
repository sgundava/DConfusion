# Troubleshooting Guide

## Common Issues and Solutions

### Issue: `ModuleNotFoundError: No module named 'matplotlib'`

**Cause:** The required dependencies aren't installed or you're using a different Python interpreter.

**Solution:**

```bash
# Install all required dependencies
pip3 install -r requirements-app.txt

# Or install individually
pip3 install streamlit pandas matplotlib scipy numpy
```

If the error persists:

```bash
# Check which Python you're using
which python3
python3 --version

# Make sure you're installing to the same Python
python3 -m pip install -r requirements-app.txt
```

### Issue: `streamlit: command not found`

**Cause:** Streamlit was installed but not added to your PATH.

**Solution:**

Use the Python module runner instead:

```bash
python3 -m streamlit run app/streamlit_app.py
```

Or add to PATH:
```bash
# Add this to your ~/.bashrc or ~/.zshrc
export PATH="$HOME/Library/Python/3.13/bin:$PATH"
```

### Issue: Can't import `dconfusion` in the app

**Cause:** Python can't find the dconfusion package.

**Solution:**

Run the app from the project root directory:

```bash
cd /Users/suryagundavarapu/Developer/DConfusion
python3 -m streamlit run app/streamlit_app.py
```

Or install the package in development mode:

```bash
pip3 install -e .
```

### Issue: Different behavior in terminal vs. app

**Cause:** You might have multiple Python installations or virtual environments.

**Solution:**

Always use the same Python interpreter:

```bash
# Check your Python
python3 -c "import sys; print(sys.executable)"

# Use that specific Python to run streamlit
/path/to/your/python3 -m streamlit run app/streamlit_app.py
```

### Issue: Port 8501 already in use

**Cause:** Another Streamlit app is running.

**Solution:**

Kill the existing process:

```bash
# Find the process
lsof -ti:8501

# Kill it
kill -9 $(lsof -ti:8501)

# Or use a different port
streamlit run app/streamlit_app.py --server.port 8502
```

## Verifying Your Setup

Run this to check everything is installed correctly:

```bash
python3 << 'EOF'
import sys
print(f"Python: {sys.executable}")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError:
    print("✗ matplotlib - Install with: pip3 install matplotlib")

try:
    import scipy
    print("✓ scipy")
except ImportError:
    print("✗ scipy - Install with: pip3 install scipy")

try:
    import streamlit
    print("✓ streamlit")
except ImportError:
    print("✗ streamlit - Install with: pip3 install streamlit")

try:
    import numpy
    print("✓ numpy")
except ImportError:
    print("✗ numpy - Install with: pip3 install numpy")

try:
    import pandas
    print("✓ pandas")
except ImportError:
    print("✗ pandas - Install with: pip3 install pandas")

try:
    from dconfusion import DConfusion
    print("✓ dconfusion")
except ImportError as e:
    print(f"✗ dconfusion - {e}")

print("\nIf all checks pass, run: python3 -m streamlit run app/streamlit_app.py")
EOF
```

## Still Having Issues?

1. **Create a clean virtual environment:**

```bash
cd /Users/suryagundavarapu/Developer/DConfusion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

2. **Check for conflicting packages:**

```bash
pip3 list | grep -E "matplotlib|scipy|streamlit|numpy|pandas"
```

3. **Reinstall everything:**

```bash
pip3 uninstall matplotlib scipy streamlit numpy pandas -y
pip3 install -r requirements-app.txt
```

## Getting Help

If none of these solutions work, please provide:
- Python version: `python3 --version`
- OS: `uname -a`
- Pip packages: `pip3 list`
- Full error message

File an issue at: https://github.com/sgundava/dconfusion/issues
