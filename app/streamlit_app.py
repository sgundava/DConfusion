"""
DConfusion Streamlit App

A web interface for comparing multiple confusion matrices and their metrics.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import dconfusion
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, APP_TITLE, APP_DESCRIPTION, FOOTER_HTML
from utils.session import init_session_state, get_matrices
from components import (
    render_sidebar,
    render_visualizations_tab,
    render_metrics_tab,
    render_warnings_tab,
    render_statistical_testing_tab,
    render_cost_analysis_tab,
    render_detailed_view_tab,
    render_metric_completion_tab,
    render_consistency_testing_tab,
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Title and description
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Initialize session state
init_session_state()

# Render sidebar
render_sidebar()

# Get matrices from session state
matrices = get_matrices()

# Main content area - three top-level tabs
tab_comparison, tab_metric_completion, tab_consistency = st.tabs([
    "📊 Model Comparison",
    "🔍 Metric Completion",
    "🔬 Consistency Testing"
])

with tab_comparison:
    if not matrices:
        st.info("👈 Add a confusion matrix using the sidebar to get started!")
    else:
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Visualizations",
            "📈 Metrics Comparison",
            "⚠️ Warnings & Quality",
            "📊 Statistical Testing",
            "💰 Cost-Sensitive Analysis",
            "📋 Detailed View"
        ])

        with tab1:
            render_visualizations_tab(matrices)

        with tab2:
            render_metrics_tab(matrices)

        with tab3:
            render_warnings_tab(matrices)

        with tab4:
            render_statistical_testing_tab(matrices)

        with tab5:
            render_cost_analysis_tab(matrices)

        with tab6:
            render_detailed_view_tab(matrices)

# Metric Completion Tab (always available, independent of models)
with tab_metric_completion:
    render_metric_completion_tab()

# Consistency Testing Tab (always available - has manual mode)
with tab_consistency:
    render_consistency_testing_tab(matrices)

# Footer
st.markdown("---")
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
