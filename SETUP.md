# Complete Setup Guide for DConfusion

## The Issue: "No module named streamlit" or "No module named matplotlib"

This happens when Python can't find the installed packages. Here's how to fix it permanently.

## Solution 1: Use the Python Launcher (Easiest)

Simply run:

```bash
cd /Users/suryagundavarapu/Developer/DConfusion
python3 run_app.py
```

This script:
- ✅ Uses the correct Python interpreter
- ✅ Checks for missing dependencies
- ✅ Offers to install them automatically
- ✅ Runs the app

## Solution 2: Install Dependencies Properly

### Step 1: Verify Your Python

```bash
which python3
python3 --version
```

You should see Python 3.8 or higher.

### Step 2: Install Dependencies to User Directory

```bash
python3 -m pip install --user streamlit pandas matplotlib scipy numpy
```

The `--user` flag installs to your user directory, avoiding permission issues.

### Step 3: Verify Installation

```bash
python3 verify_setup.py
```

This will check everything is installed correctly.

### Step 4: Run the App

```bash
python3 -m streamlit run app/streamlit_app.py
```

Note: Use `python3 -m streamlit` NOT just `streamlit`

## Solution 3: Use a Virtual Environment (Best Practice)

Create an isolated environment with all dependencies:

```bash
# Navigate to project
cd /Users/suryagundavarapu/Developer/DConfusion

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements-app.txt

# Run the app
streamlit run app/streamlit_app.py

# When done, deactivate
deactivate
```

## Why Does This Happen?

### Multiple Python Installations

You might have:
- System Python: `/usr/bin/python`
- Python.org Python: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`
- Homebrew Python: `/opt/homebrew/bin/python3`

Each has its own set of installed packages!

### Solution: Always Use the Same Python

Find which Python has your packages:

```bash
python3 -m pip show streamlit
```

If this shows streamlit, then always use:
- `python3` (not `python`)
- `python3 -m streamlit` (not `streamlit`)
- `python3 run_app.py` (not `python run_app.py`)

## Quick Reference Commands

| What You Want | Command |
|---------------|---------|
| Verify setup | `python3 verify_setup.py` |
| Install deps | `python3 -m pip install --user -r requirements-app.txt` |
| Run app (easiest) | `python3 run_app.py` |
| Run app (direct) | `python3 -m streamlit run app/streamlit_app.py` |
| Check Python | `which python3 && python3 --version` |
| Check streamlit | `python3 -m pip show streamlit` |

## Still Not Working?

### Check if streamlit is really installed:

```bash
python3 -c "import streamlit; print('OK')"
```

If this prints "OK", streamlit is installed for python3.

### If it says "No module named streamlit":

```bash
python3 -m pip install --user streamlit
```

### If you get permission errors:

```bash
# Use --user flag
python3 -m pip install --user streamlit pandas matplotlib scipy numpy
```

Or use a virtual environment (Solution 3 above).

## Pro Tip: Add an Alias

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
alias dconfusion='cd /Users/suryagundavarapu/Developer/DConfusion && python3 run_app.py'
```

Then you can just type `dconfusion` from anywhere to run the app!

## Testing Without Running the Full App

Test that everything imports:

```bash
python3 << 'EOF'
print("Testing imports...")
import streamlit
print("✓ streamlit")
import pandas
print("✓ pandas")
import matplotlib
print("✓ matplotlib")
import scipy
print("✓ scipy")
from dconfusion import DConfusion
print("✓ dconfusion")
print("\n✅ All packages import successfully!")
print("Run: python3 run_app.py")
EOF
```

## Need More Help?

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting steps.
