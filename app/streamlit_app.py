"""
DConfusion Streamlit App

A web interface for comparing multiple confusion matrices and their metrics.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import dconfusion
sys.path.insert(0, str(Path(__file__).parent.parent))

from dconfusion import DConfusion, WarningSeverity

# Page configuration
st.set_page_config(
    page_title="DConfusion - Confusion Matrix Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 DConfusion - Confusion Matrix Comparison Tool")
st.markdown("""
Compare multiple confusion matrices side-by-side and analyze their performance metrics.
Perfect for evaluating different models or configurations.
""")

# Initialize session state for storing confusion matrices
if 'matrices' not in st.session_state:
    st.session_state.matrices = {}

# Sidebar for adding confusion matrices
st.sidebar.header("Add Confusion Matrix")

input_method = st.sidebar.radio(
    "Input Method",
    ["Binary (TP/FN/FP/TN)", "Multi-class Matrix", "From Predictions"]
)

matrix_name = st.sidebar.text_input("Matrix Name", value=f"Model {len(st.session_state.matrices) + 1}")

if input_method == "Binary (TP/FN/FP/TN)":
    st.sidebar.markdown("### Enter Values")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        tp = st.number_input("True Positive", min_value=0, value=85, step=1)
        fn = st.number_input("False Negative", min_value=0, value=15, step=1)

    with col2:
        fp = st.number_input("False Positive", min_value=0, value=10, step=1)
        tn = st.number_input("True Negative", min_value=0, value=90, step=1)

    if st.sidebar.button("Add Matrix", type="primary"):
        try:
            cm = DConfusion(tp, fn, fp, tn)
            st.session_state.matrices[matrix_name] = cm

            # Check for warnings
            warnings = cm.check_warnings(include_info=False)
            critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
            warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

            if critical:
                st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(critical)} CRITICAL warning(s)")
            elif warning_level:
                st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(warning_level)} warning(s)")
            else:
                st.sidebar.success(f"✅ Added {matrix_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

elif input_method == "Multi-class Matrix":
    st.sidebar.markdown("### Multi-class Configuration")
    n_classes = st.sidebar.number_input("Number of Classes", min_value=2, max_value=10, value=3, step=1)

    st.sidebar.markdown("### Enter Matrix Values")
    matrix_values = []

    for i in range(n_classes):
        row = []
        cols = st.sidebar.columns(n_classes)
        for j in range(n_classes):
            with cols[j]:
                val = st.number_input(f"[{i},{j}]", min_value=0, value=10 if i==j else 2, step=1, key=f"matrix_{i}_{j}")
                row.append(val)
        matrix_values.append(row)

    labels_input = st.sidebar.text_input("Labels (comma-separated, optional)", value="")
    labels = [l.strip() for l in labels_input.split(",")] if labels_input else None

    if st.sidebar.button("Add Matrix", type="primary"):
        try:
            cm = DConfusion(confusion_matrix=matrix_values, labels=labels)
            st.session_state.matrices[matrix_name] = cm

            # Check for warnings
            warnings = cm.check_warnings(include_info=False)
            critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
            warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

            if critical:
                st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(critical)} CRITICAL warning(s)")
            elif warning_level:
                st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(warning_level)} warning(s)")
            else:
                st.sidebar.success(f"✅ Added {matrix_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

else:  # From Predictions
    st.sidebar.markdown("### Enter Predictions")
    y_true_input = st.sidebar.text_area("True Labels (comma-separated)", value="1,0,1,1,0,0,1,0")
    y_pred_input = st.sidebar.text_area("Predicted Labels (comma-separated)", value="1,0,0,1,0,1,1,0")

    if st.sidebar.button("Add Matrix", type="primary"):
        try:
            y_true = [int(x.strip()) for x in y_true_input.split(",")]
            y_pred = [int(x.strip()) for x in y_pred_input.split(",")]
            cm = DConfusion.from_predictions(y_true, y_pred)
            st.session_state.matrices[matrix_name] = cm

            # Check for warnings
            warnings = cm.check_warnings(include_info=False)
            critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
            warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

            if critical:
                st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(critical)} CRITICAL warning(s)")
            elif warning_level:
                st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(warning_level)} warning(s)")
            else:
                st.sidebar.success(f"✅ Added {matrix_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# Clear all button
if st.sidebar.button("🗑️ Clear All Matrices"):
    st.session_state.matrices = {}
    st.rerun()

# Main content area
if not st.session_state.matrices:
    st.info("👈 Add a confusion matrix using the sidebar to get started!")
else:
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📈 Metrics Comparison", "⚠️ Warnings & Quality", "📋 Detailed View"])

    with tab1:
        st.header("Confusion Matrix Visualizations")

        # Options for visualization
        col1, col2 = st.columns(2)
        with col1:
            normalize = st.checkbox("Normalize (show percentages)", value=False)
        with col2:
            show_metrics = st.checkbox("Show metrics panel (binary only)", value=True)

        # Display matrices in columns
        n_matrices = len(st.session_state.matrices)
        cols_per_row = min(3, n_matrices)

        matrix_items = list(st.session_state.matrices.items())
        for i in range(0, n_matrices, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (name, cm) in enumerate(matrix_items[i:i+cols_per_row]):
                with cols[j]:
                    # Check for warnings and add status badge
                    warnings = cm.check_warnings(include_info=False)
                    critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
                    warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

                    if critical:
                        st.subheader(f"🔴 {name}")
                        st.caption(f"{len(critical)} critical issue(s)")
                    elif warning_level:
                        st.subheader(f"🟡 {name}")
                        st.caption(f"{len(warning_level)} warning(s)")
                    else:
                        st.subheader(f"🟢 {name}")

                    try:
                        fig = cm.plot(
                            normalize=normalize,
                            show_metrics=show_metrics and cm.n_classes == 2,
                            figsize=(6, 5)
                        )
                        st.pyplot(fig)

                        # Remove button
                        if st.button(f"Remove", key=f"remove_{name}"):
                            del st.session_state.matrices[name]
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error plotting: {str(e)}")

    with tab2:
        st.header("Metrics Comparison")

        # Collect all metrics
        metrics_data = {}
        for name, cm in st.session_state.matrices.items():
            try:
                all_metrics = cm.get_all_metrics()

                # For binary matrices
                if cm.n_classes == 2:
                    metrics_data[name] = {
                        'Accuracy': all_metrics.get('accuracy', 0),
                        'Precision': all_metrics.get('precision', 0),
                        'Recall': all_metrics.get('recall', 0),
                        'Specificity': all_metrics.get('specificity', 0),
                        'F1 Score': all_metrics.get('f1_score', 0),
                        'False Positive Rate': all_metrics.get('false_positive_rate', 0),
                        'False Negative Rate': all_metrics.get('false_negative_rate', 0),
                    }

                    # Add optional metrics if available
                    if 'g_mean' in all_metrics:
                        metrics_data[name]['G-Mean'] = all_metrics['g_mean']
                    if 'matthews_correlation_coefficient' in all_metrics:
                        metrics_data[name]['MCC'] = all_metrics['matthews_correlation_coefficient']
                    if 'cohens_kappa' in all_metrics:
                        metrics_data[name]['Cohen\'s Kappa'] = all_metrics['cohens_kappa']

                # For multi-class matrices
                else:
                    metrics_data[name] = {
                        'Accuracy': all_metrics.get('accuracy', 0),
                        'Macro Precision': all_metrics.get('macro_precision', 0),
                        'Macro Recall': all_metrics.get('macro_recall', 0),
                        'Macro F1 Score': all_metrics.get('macro_f1_score', 0),
                        'Weighted Precision': all_metrics.get('weighted_precision', 0),
                        'Weighted Recall': all_metrics.get('weighted_recall', 0),
                        'Weighted F1 Score': all_metrics.get('weighted_f1_score', 0),
                    }
            except Exception as e:
                st.error(f"Error calculating metrics for {name}: {str(e)}")

        if metrics_data:
            # Create comparison dataframe
            df = pd.DataFrame(metrics_data).T

            # Format as percentages
            df_display = df.copy()
            for col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

            st.dataframe(df_display, width='stretch')

            # Highlight best values
            st.markdown("### 🏆 Best Performers")
            best_metrics = {}
            for metric in df.columns:
                if metric not in ['False Positive Rate', 'False Negative Rate']:
                    best_model = df[metric].idxmax()
                    best_value = df[metric].max()
                    best_metrics[metric] = f"{best_model} ({best_value:.4f})"
                else:
                    # For error rates, lower is better
                    best_model = df[metric].idxmin()
                    best_value = df[metric].min()
                    best_metrics[metric] = f"{best_model} ({best_value:.4f})"

            best_df = pd.DataFrame([best_metrics]).T
            best_df.columns = ['Best Model (Value)']
            st.dataframe(best_df, width='stretch')

            # Download metrics as CSV
            csv = df.to_csv()
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv,
                file_name="confusion_matrix_metrics.csv",
                mime="text/csv"
            )

    with tab3:
        st.header("⚠️ Data Quality Warnings")
        st.markdown("""
        This section checks for common pitfalls in confusion matrix analysis based on peer-reviewed research.
        Warnings help identify issues with sample size, class imbalance, metric reliability, and more.
        """)

        # Summary statistics
        total_matrices = len(st.session_state.matrices)
        matrices_with_critical = 0
        matrices_with_warnings = 0
        matrices_clean = 0

        for name, cm in st.session_state.matrices.items():
            warnings = cm.check_warnings(include_info=False)
            critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
            warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

            if critical:
                matrices_with_critical += 1
            elif warning_level:
                matrices_with_warnings += 1
            else:
                matrices_clean += 1

        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matrices", total_matrices)
        with col2:
            st.metric("🔴 Critical Issues", matrices_with_critical)
        with col3:
            st.metric("🟡 Warnings", matrices_with_warnings)
        with col4:
            st.metric("🟢 Clean", matrices_clean)

        st.markdown("---")

        # Individual matrix warnings
        for name, cm in st.session_state.matrices.items():
            warnings = cm.check_warnings(include_info=False)
            critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
            warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

            # Determine status icon
            if critical:
                status = "🔴"
                status_text = "CRITICAL ISSUES"
                color = "red"
            elif warning_level:
                status = "🟡"
                status_text = "WARNINGS"
                color = "orange"
            else:
                status = "🟢"
                status_text = "NO ISSUES"
                color = "green"

            with st.expander(f"{status} **{name}** - {status_text}", expanded=bool(critical or warning_level)):
                if not warnings:
                    st.success("✅ This confusion matrix has no detected issues. Good quality data!")
                else:
                    # Display critical warnings first
                    if critical:
                        st.error(f"**🔴 {len(critical)} CRITICAL WARNING(S)**")
                        for w in critical:
                            st.markdown(f"**{w.category}**")
                            st.markdown(f"*{w.message}*")
                            if w.recommendation:
                                st.info(f"💡 **Recommendation:** {w.recommendation}")
                            st.markdown("---")

                    # Then regular warnings
                    if warning_level:
                        st.warning(f"**🟡 {len(warning_level)} WARNING(S)**")
                        for w in warning_level:
                            st.markdown(f"**{w.category}**")
                            st.markdown(f"*{w.message}*")
                            if w.recommendation:
                                st.info(f"💡 **Recommendation:** {w.recommendation}")
                            st.markdown("---")

                # Matrix summary
                st.markdown("**Matrix Summary:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- Total Samples: {cm.total}")
                    st.write(f"- Classes: {cm.n_classes}")
                with col2:
                    st.write(f"- Accuracy: {cm.get_accuracy():.4f}")
                    if cm.n_classes == 2:
                        pos_samples = cm.true_positive + cm.false_negative
                        neg_samples = cm.true_negative + cm.false_positive
                        st.write(f"- Positive/Negative: {pos_samples}/{neg_samples}")

        # Model comparison warnings
        if len(st.session_state.matrices) >= 2:
            st.markdown("---")
            st.subheader("🔄 Model Comparison Warnings")
            st.markdown("Check if comparing these models is statistically meaningful.")

            matrix_list = list(st.session_state.matrices.items())
            for i in range(len(matrix_list)):
                for j in range(i + 1, len(matrix_list)):
                    name1, cm1 = matrix_list[i]
                    name2, cm2 = matrix_list[j]

                    # Only compare if both are same type (binary or multi-class)
                    if cm1.n_classes == cm2.n_classes == 2:
                        from dconfusion import check_comparison_validity
                        comp_warnings = check_comparison_validity(cm1, cm2)

                        if comp_warnings:
                            with st.expander(f"⚠️ **{name1}** vs **{name2}**", expanded=False):
                                st.warning(f"Found {len(comp_warnings)} comparison issue(s)")
                                for w in comp_warnings:
                                    st.markdown(f"**{w.category}**")
                                    st.markdown(f"*{w.message}*")
                                    if w.recommendation:
                                        st.info(f"💡 **Recommendation:** {w.recommendation}")
                                    st.markdown("---")

                                # Show actual comparison
                                result = cm1.compare_with(cm2, show_warnings=False)
                                st.markdown("**Metric Comparison:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(name1, f"{result['value1']:.4f}")
                                with col2:
                                    st.metric(name2, f"{result['value2']:.4f}")
                                with col3:
                                    st.metric("Difference", f"{result['difference']:.4f}",
                                            delta=f"{result['relative_difference']*100:.2f}%")

        # Information about warnings
        with st.expander("ℹ️ About These Warnings", expanded=False):
            st.markdown("""
            ### Research Foundation

            These warnings are based on peer-reviewed research on binary classification metrics:

            1. **Chicco et al.** - Studies on MCC advantages and metric limitations
            2. **Lovell et al.** - Research showing uncertainty scales as 1/√N
            3. **Fazekas & Kovács** - Work on numerical consistency in ML evaluation

            ### Warning Categories

            - **Sample Size**: Total samples < 100 leads to high uncertainty
            - **Class Imbalance**: Minority class with < 30 samples
            - **Empty Cells**: Zero values in TP/TN/FP/FN
            - **Perfect Classification**: May indicate data leakage
            - **Poor Basic Rates**: Low TPR, TNR, PPV, or NPV
            - **Misleading Accuracy**: High accuracy hiding poor performance
            - **Comparison Issues**: Problems when comparing models

            ### Key Insights

            - **Uncertainty scales as 1/√N** - Need 4x more data to halve uncertainty
            - **Absolute sample count per class matters more than balance ratio**
            - **High accuracy or ROC AUC can hide poor performance in specific rates**
            - **Always report all 4 basic rates** (TPR, TNR, PPV, NPV)
            - **Perfect results often indicate methodological issues**

            For more details, see the package documentation.
            """)

    with tab4:
        st.header("Detailed Matrix View")

        for name, cm in st.session_state.matrices.items():
            with st.expander(f"📋 {name}", expanded=True):
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Confusion Matrix**")
                    st.text(str(cm))

                    st.markdown("**Frequency Distribution**")
                    st.text(cm.frequency())

                with col2:
                    st.markdown("**Matrix Properties**")
                    st.write(f"- Classes: {cm.n_classes}")
                    st.write(f"- Labels: {cm.labels}")
                    st.write(f"- Total Samples: {cm.total}")
                    st.write(f"- Correct Predictions: {cm.get_sum_of_corrects()}")
                    st.write(f"- Incorrect Predictions: {cm.get_sum_of_errors()}")

                    if cm.n_classes == 2:
                        st.markdown("**Binary Values**")
                        binary_vals = cm.get_binary_values()
                        for key, val in binary_vals.items():
                            st.write(f"- {key}: {val}")

                # Export options
                st.markdown("**Export Options**")
                col1, col2 = st.columns(2)

                with col1:
                    dict_export = cm.to_dict()
                    st.download_button(
                        label="📥 Export as JSON",
                        data=str(dict_export),
                        file_name=f"{name.replace(' ', '_')}.json",
                        mime="application/json",
                        key=f"json_{name}"
                    )

                with col2:
                    import tempfile
                    import os
                    tmpfile = tempfile.mktemp(suffix='.csv')
                    cm.to_csv(tmpfile)
                    with open(tmpfile, 'r') as f:
                        csv_data = f.read()
                    os.remove(tmpfile)

                    st.download_button(
                        label="📥 Export as CSV",
                        data=csv_data,
                        file_name=f"{name.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"csv_{name}"
                    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with DConfusion 📊 | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
