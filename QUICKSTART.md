# DConfusion Quick Start Guide

## Project Structure

Your project now has a clean, modular structure:

```
DConfusion/
├── dconfusion/              # Core Python package
│   ├── __init__.py
│   ├── DConfusion.py        # Main facade class (37 lines)
│   ├── core.py             # Core initialization & display
│   ├── metrics.py          # All metric calculations
│   ├── visualization.py    # Plotting functions
│   ├── io.py              # Import/export (CSV, JSON, predictions)
│   └── validation.py      # Input validation utilities
│
├── app/                    # Streamlit web UI
│   ├── streamlit_app.py   # Main web application
│   └── README.md          # App-specific documentation
│
├── requirements-app.txt    # Web UI dependencies
├── pyproject.toml         # Package configuration
└── README.md
```

## Running the Streamlit App

### Step 0: Verify Setup (Recommended)

Before running the app, verify all dependencies are installed:

```bash
python3 verify_setup.py
```

This will check your setup and tell you exactly what's missing (if anything).

### Step 1: Install Dependencies

```bash
# Install the app dependencies
pip3 install -r requirements-app.txt
```

**Troubleshooting?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

### Step 2: Run the App

```bash
# From the project root directory
streamlit run app/streamlit_app.py
```

If streamlit is not in your PATH, use:
```bash
python3 -m streamlit run app/streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Using the Web Interface

### Adding Confusion Matrices

The sidebar offers three input methods:

**1. Binary Classification (TP/FN/FP/TN)**
- Enter the four confusion matrix values
- Perfect for simple binary classifiers
- Example: TP=85, FN=15, FP=10, TN=90

**2. Multi-class Matrix**
- Specify number of classes (2-10)
- Fill in the NxN matrix values
- Optional: Add custom labels (comma-separated)

**3. From Predictions**
- Paste true labels: `1,0,1,1,0,0,1,0`
- Paste predicted labels: `1,0,0,1,0,1,1,0`
- Automatically generates confusion matrix

### Four Main Views

**📊 Visualizations Tab**
- View all matrices side-by-side
- **NEW:** Status badges (🔴🟡🟢) showing data quality
- Toggle normalized view (percentages)
- Show metrics panels (binary only)
- Remove individual matrices

**📈 Metrics Comparison Tab**
- Compare all models at once
- See best performers for each metric
- Download metrics as CSV
- Includes: Accuracy, Precision, Recall, F1, etc.

**⚠️ Warnings & Quality Tab** (NEW!)
- **Dashboard**: Overview of matrix quality across all models
- **Individual Analysis**: Detailed warnings with recommendations
- **Comparison Warnings**: Statistical validity of model comparisons
- **Research Info**: Background on warning categories
- Detects: small samples, class imbalance, misleading metrics, etc.

**📋 Detailed View Tab**
- Full matrix details
- Frequency distributions
- Matrix properties
- Export individual matrices (JSON/CSV)

## Example Workflow

Let's compare two models:

1. **Add Model A**
   - Click sidebar: Binary input
   - Name: "Random Forest"
   - TP=85, FN=15, FP=10, TN=90
   - Click "Add Matrix"

2. **Add Model B**
   - Name: "SVM"
   - TP=80, FN=20, FP=5, TN=95
   - Click "Add Matrix"

3. **Compare**
   - Go to "Metrics Comparison" tab
   - See which model has better Precision, Recall, etc.
   - Check "Best Performers" section

4. **Export**
   - Click "Download Metrics CSV"
   - Or export individual matrices from "Detailed View"

## Tips & Tricks

### Comparing Multiple Models
- Add 3+ models to see comprehensive comparisons
- Use descriptive names like "Model v1.0", "Model v2.0"
- Normalized views help when sample sizes differ

### Best Practices
- **Precision-focused**: Look for lowest FP rate
- **Recall-focused**: Look for lowest FN rate
- **Balanced**: Use F1 Score or G-Mean
- **Medical/Critical**: Prioritize Recall (minimize false negatives)
- **Spam detection**: Prioritize Precision (minimize false positives)

### Keyboard Shortcuts
- `Ctrl/Cmd + R` - Rerun the app
- `Ctrl/Cmd + Shift + R` - Clear cache and rerun
- Settings (⋮) → Toggle wide mode for better viewing

## Using DConfusion in Python Code

```python
from dconfusion import DConfusion

# Binary classification
cm = DConfusion(tp=85, fn=15, fp=10, tn=90)
print(f"Accuracy: {cm.accuracy}")
print(f"F1 Score: {cm.f1_score}")

# Multi-class
cm2 = DConfusion(
    confusion_matrix=[[50, 5, 2], [3, 45, 4], [1, 2, 48]],
    labels=['Cat', 'Dog', 'Bird']
)
print(cm2)

# From predictions
y_true = [1, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1]
cm3 = DConfusion.from_predictions(y_true, y_pred)
print(cm3.get_all_metrics())

# Export/Import
cm.to_csv('my_matrix.csv')
cm_loaded = DConfusion.from_csv('my_matrix.csv')

# Plotting
fig = cm.plot(normalize=True, show_metrics=True)
fig.savefig('confusion_matrix.png')
```

## Next Steps

### For Researchers
- Add multiple models to compare experiments
- Export metrics table for your papers
- Save visualizations for presentations

### For Developers
- Integrate with your ML pipeline
- Use the modular structure (metrics.py, io.py) in your code
- Build custom UIs using the core modules

### Going to Production
- Deploy to Streamlit Cloud (free hosting)
- Or use Docker for self-hosting
- Add authentication if needed

## Need Help?

- Check `app/README.md` for app-specific docs
- Review the package README for API documentation
- File issues on GitHub

---

**Built with ❤️ using DConfusion and Streamlit**
