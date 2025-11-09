# DConfusion Streamlit App

A web-based interface for comparing multiple confusion matrices and analyzing their performance metrics.

## Features

- 📊 **Multiple Input Methods**
  - Binary classification (TP/FN/FP/TN)
  - Multi-class matrices
  - From prediction lists

- 📈 **Visualizations**
  - Side-by-side confusion matrix plots
  - Normalized views (percentages)
  - Optional metrics panels

- 🔍 **Metrics Comparison**
  - Compare all models at once
  - Highlight best performers
  - Export metrics as CSV

- 📋 **Detailed Views**
  - Matrix properties
  - Frequency distributions
  - Export as JSON/CSV

## Installation

1. Install dependencies:
```bash
pip install -r requirements-app.txt
```

2. Run the app:
```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Add Matrices**: Use the sidebar to add confusion matrices
2. **Compare**: View all matrices side-by-side in the Visualizations tab
3. **Analyze**: Check the Metrics Comparison tab to see which model performs best
4. **Export**: Download metrics or individual matrices from the Detailed View tab

## Example Workflow

1. Add "Model A" using binary input: TP=85, FN=15, FP=10, TN=90
2. Add "Model B" using binary input: TP=80, FN=20, FP=5, TN=95
3. Compare them in the Metrics Comparison tab
4. Download the comparison as CSV

## Tips

- Use normalized views to compare matrices with different sample sizes
- The "Best Performers" section automatically highlights which model excels at each metric
- You can add as many matrices as you want for comparison
