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

# Helper function to display warnings
def display_warnings(warnings):
    """Display warnings in a formatted way."""
    for w in warnings:
        st.markdown(f"**{w.category}**")
        st.markdown(f"*{w.message}*")
        if w.recommendation:
            st.info(f"💡 **Recommendation:** {w.recommendation}")
        st.markdown("---")

# Helper function to add matrix and show feedback
def add_matrix_with_feedback(matrix_name, cm):
    """Add a confusion matrix to session state and show appropriate feedback."""
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

# Initialize session state for storing confusion matrices
if 'matrices' not in st.session_state:
    st.session_state.matrices = {}

# Sidebar for adding confusion matrices
st.sidebar.header("Add Confusion Matrix")

input_method = st.sidebar.radio(
    "Input Method",
    ["Binary (TP/FN/FP/TN)", "Multi-class Matrix", "From Predictions"]
)

matrix_name = st.sidebar.text_input("Matrix Name", value=f"Model {len(st.session_state.matrices) + 1}", key="matrix_name_input")

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
            add_matrix_with_feedback(matrix_name, cm)
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
            add_matrix_with_feedback(matrix_name, cm)
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
            add_matrix_with_feedback(matrix_name, cm)
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# Clear all button
if st.sidebar.button("🗑️ Clear All Matrices"):
    st.session_state.matrices = {}
    st.rerun()

# Main content area - two top-level tabs
tab_comparison, tab_metric_completion = st.tabs(["📊 Model Comparison", "🔍 Metric Completion"])

with tab_comparison:
    if not st.session_state.matrices:
        st.info("👈 Add a confusion matrix using the sidebar to get started!")
    else:
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Visualizations", "📈 Metrics Comparison", "⚠️ Warnings & Quality", "📊 Statistical Testing", "💰 Cost-Sensitive Analysis", "📋 Detailed View"])

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
                            display_warnings(critical)
    
                        # Then regular warnings
                        if warning_level:
                            st.warning(f"**🟡 {len(warning_level)} WARNING(S)**")
                            display_warnings(warning_level)
    
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
                                    display_warnings(comp_warnings)
    
                                    # Show actual comparison (cm2 vs cm1, so positive = improvement)
                                    result = cm2.compare_with(cm1, show_warnings=False)
                                    st.markdown("**Metric Comparison:**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(name1, f"{result['value2']:.4f}")
                                    with col2:
                                        st.metric(name2, f"{result['value1']:.4f}")
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
            st.header("📊 Statistical Testing")
            st.markdown("""
            Perform statistical tests to compare models and estimate confidence intervals for metrics.
            These methods provide rigorous statistical evidence for model comparisons.
            """)
    
            # Bootstrap Confidence Intervals Section
            st.subheader("🎲 Bootstrap Confidence Intervals")
            st.markdown("""
            Estimate the uncertainty in your metrics using bootstrap resampling.
            This method doesn't assume any particular distribution and works well for small samples.
            """)
    
            # Select a model for CI
            if st.session_state.matrices:
                model_names = list(st.session_state.matrices.keys())
                selected_model = st.selectbox("Select Model", model_names, key="ci_model")
                cm_selected = st.session_state.matrices[selected_model]
    
                if cm_selected.n_classes == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        metric_for_ci = st.selectbox(
                            "Metric",
                            ["accuracy", "precision", "recall", "specificity", "f1_score"],
                            key="ci_metric"
                        )
                    with col2:
                        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
                    n_bootstrap = st.slider("Bootstrap Samples", 100, 5000, 1000, 100)
    
                    if st.button("Calculate Confidence Interval", key="calc_ci"):
                        with st.spinner("Running bootstrap resampling..."):
                            try:
                                result = cm_selected.get_bootstrap_confidence_interval(
                                    metric=metric_for_ci,
                                    confidence_level=confidence_level,
                                    n_bootstrap=n_bootstrap,
                                    random_state=42
                                )
    
                                st.success("✅ Confidence interval calculated successfully!")
    
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Point Estimate", f"{result['point_estimate']:.4f}")
                                with col2:
                                    st.metric("Lower Bound", f"{result['lower']:.4f}")
                                with col3:
                                    st.metric("Upper Bound", f"{result['upper']:.4f}")
    
                                st.info(f"**Interpretation:** We are {confidence_level*100:.0f}% confident that the true {metric_for_ci} "
                                       f"lies between {result['lower']:.4f} and {result['upper']:.4f}. "
                                       f"Standard error: {result['std_error']:.4f}")
    
                            except Exception as e:
                                st.error(f"Error calculating CI: {str(e)}")
                else:
                    st.warning("Bootstrap confidence intervals currently only support binary classification.")
    
            # McNemar's Test Section
            st.markdown("---")
            st.subheader("🔬 McNemar's Test (Paired Model Comparison)")
            st.markdown("""
            Compare two models tested on the same dataset using McNemar's test.
            This test determines if there's a statistically significant difference between the models.
            """)
    
            if len(st.session_state.matrices) >= 2:
                binary_models = {name: cm for name, cm in st.session_state.matrices.items() if cm.n_classes == 2}
    
                if len(binary_models) >= 2:
                    col1, col2 = st.columns(2)
                    model_names = list(binary_models.keys())
    
                    with col1:
                        model1_name = st.selectbox("Model 1", model_names, key="mcnemar_model1")
                    with col2:
                        model2_name = st.selectbox("Model 2", model_names, index=min(1, len(model_names)-1), key="mcnemar_model2")
    
                    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
                    if model1_name != model2_name:
                        if st.button("Run McNemar's Test", key="run_mcnemar"):
                            cm1 = binary_models[model1_name]
                            cm2 = binary_models[model2_name]
    
                            try:
                                result = cm1.mcnemar_test(cm2, alpha=alpha)
    
                                if result['significant']:
                                    st.success(f"✅ Statistically significant difference detected (p={result['p_value']:.4f})")
                                else:
                                    st.info(f"ℹ️ No statistically significant difference (p={result['p_value']:.4f})")
    
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Test Statistic (χ²)", f"{result['statistic']:.4f}")
                                with col2:
                                    st.metric("P-value", f"{result['p_value']:.4f}")
                                with col3:
                                    st.metric("Significant?", "Yes" if result['significant'] else "No")
    
                                st.markdown("**Interpretation:**")
                                st.write(result['interpretation'])
    
                                st.markdown("**Contingency Table:**")
                                st.write(f"- {model1_name} correct, {model2_name} wrong: {result['contingency_table']['disagree_b']}")
                                st.write(f"- {model1_name} wrong, {model2_name} correct: {result['contingency_table']['disagree_c']}")
    
                                if 'warning' in result:
                                    st.warning(f"⚠️ {result['warning']}")
    
                            except Exception as e:
                                st.error(f"Error running McNemar's test: {str(e)}")
                    else:
                        st.warning("Please select two different models to compare.")
                else:
                    st.warning("Need at least 2 binary classification models for McNemar's test.")
            else:
                st.info("Add at least 2 confusion matrices to use statistical testing.")
    
            # About Statistical Tests
            with st.expander("ℹ️ About These Tests", expanded=False):
                st.markdown("""
                ### Bootstrap Confidence Intervals
    
                Bootstrap resampling is a non-parametric method to estimate the sampling distribution
                of a statistic. It works by:
                1. Resampling your data with replacement many times (typically 1000+)
                2. Computing the metric for each resample
                3. Using the distribution of resampled metrics to construct confidence intervals
    
                **Advantages:**
                - No assumptions about underlying distributions
                - Works well for complex metrics (like F1 score)
                - Accounts for sample size limitations
    
                ### McNemar's Test
    
                McNemar's test is specifically designed for comparing two classifiers on the same dataset.
                Unlike comparing accuracy alone, it accounts for the paired nature of predictions.
    
                **Key Points:**
                - Null hypothesis: Both models have the same error rate
                - Focuses on cases where models disagree
                - More powerful than unpaired tests
                - Requires at least 10 disagreements for reliability
    
                **Reference:** McNemar, Q. (1947). "Note on the sampling error of the difference
                between correlated proportions or percentages". Psychometrika.
                """)
    
        with tab5:
            st.header("💰 Cost-Sensitive Analysis")
            st.markdown("""
            Evaluate your models based on real-world costs and benefits. Different classification errors
            have different business impacts - optimize for what matters to your use case.
            """)
    
            # Filter binary models only
            binary_models = {name: cm for name, cm in st.session_state.matrices.items() if cm.n_classes == 2}
    
            if not binary_models:
                st.warning("Cost-sensitive analysis is currently only available for binary classification. Please add binary confusion matrices.")
            else:
                # Subsections
                cost_tab1, cost_tab2, cost_tab3 = st.tabs(["💵 Cost Calculation", "🎯 Metric Recommendation", "⚖️ Model Comparison"])
    
                with cost_tab1:
                    st.subheader("Calculate Misclassification Costs")
                    st.markdown("Define the costs (or benefits) for each outcome and see your model's business impact.")
    
                    # Model selection
                    selected_model = st.selectbox("Select Model", list(binary_models.keys()), key="cost_calc_model")
                    cm_selected = binary_models[selected_model]
    
                    # Cost inputs
                    st.markdown("### Define Cost Structure")
                    col1, col2 = st.columns(2)
    
                    with col1:
                        st.markdown("**Costs (Errors)**")
                        cost_fp = st.number_input("Cost of False Positive (FP)", value=100.0, step=10.0,
                                                 help="Cost when model incorrectly predicts positive")
                        cost_fn = st.number_input("Cost of False Negative (FN)", value=1000.0, step=10.0,
                                                 help="Cost when model incorrectly predicts negative")
    
                    with col2:
                        st.markdown("**Benefits (Correct Predictions)**")
                        benefit_tp = st.number_input("Benefit of True Positive (TP)", value=50.0, step=10.0,
                                                    help="Benefit/revenue from correct positive prediction")
                        benefit_tn = st.number_input("Benefit of True Negative (TN)", value=10.0, step=10.0,
                                                    help="Benefit/savings from correct negative prediction")
    
                    if st.button("Calculate Costs", type="primary"):
                        try:
                            # Get cost-benefit summary
                            summary = cm_selected.get_cost_benefit_summary(
                                cost_fp=cost_fp,
                                cost_fn=cost_fn,
                                benefit_tp=benefit_tp,
                                benefit_tn=benefit_tn
                            )
    
                            st.success("✅ Cost analysis completed!")
    
                            # Display key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Your Model Cost", f"${summary['total_cost']:,.0f}")
                            with col2:
                                st.metric("Average Cost/Sample", f"${summary['average_cost']:.2f}")
                            with col3:
                                st.metric("Perfect Classifier", f"${summary['perfect_classifier_cost']:,.0f}")
                            with col4:
                                savings = summary['savings_vs_perfect']
                                st.metric("Gap to Perfect", f"${abs(savings):,.0f}",
                                         delta=f"{savings:,.0f}", delta_color="inverse")
    
                            st.markdown("---")
                            st.markdown("### 📊 Cost Breakdown")
    
                            # Cost components
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Cost Components:**")
                                st.write(f"- True Positives: ${summary['tp_component']:,.0f} ({cm_selected.true_positive} × ${-benefit_tp if benefit_tp else 0})")
                                st.write(f"- True Negatives: ${summary['tn_component']:,.0f} ({cm_selected.true_negative} × ${-benefit_tn if benefit_tn else 0})")
                                st.write(f"- False Positives: ${summary['fp_component']:,.0f} ({cm_selected.false_positive} × ${cost_fp})")
                                st.write(f"- False Negatives: ${summary['fn_component']:,.0f} ({cm_selected.false_negative} × ${cost_fn})")
    
                            with col2:
                                st.markdown("**Baseline Comparisons:**")
                                st.write(f"- Random Classifier: ${summary['random_classifier_cost']:,.0f}")
                                st.write(f"- Always Positive: ${summary['all_positive_cost']:,.0f}")
                                st.write(f"- Always Negative: ${summary['all_negative_cost']:,.0f}")
    
                            # Improvement metrics
                            st.markdown("### 📈 Performance vs Baselines")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                savings_random = summary['savings_vs_random']
                                improvement_pct = summary['cost_improvement_over_random'] * 100
                                st.metric("Savings vs Random", f"${savings_random:,.0f}",
                                         delta=f"{improvement_pct:.1f}% better")
                            with col2:
                                savings_pos = summary['savings_vs_all_positive']
                                st.metric("Savings vs Always Positive", f"${savings_pos:,.0f}",
                                         delta="Better" if savings_pos > 0 else "Worse")
                            with col3:
                                savings_neg = summary['savings_vs_all_negative']
                                st.metric("Savings vs Always Negative", f"${savings_neg:,.0f}",
                                         delta="Better" if savings_neg > 0 else "Worse")
    
                            # Interpretation
                            if improvement_pct > 50:
                                st.success(f"🎉 Excellent! Your model provides {improvement_pct:.1f}% cost improvement over random guessing.")
                            elif improvement_pct > 25:
                                st.info(f"✅ Good performance. Your model provides {improvement_pct:.1f}% cost improvement over random guessing.")
                            elif improvement_pct > 0:
                                st.warning(f"⚠️ Modest improvement. Your model provides only {improvement_pct:.1f}% cost improvement.")
                            else:
                                st.error("❌ Your model performs worse than random guessing from a cost perspective!")
    
                        except Exception as e:
                            st.error(f"Error calculating costs: {str(e)}")
    
                with cost_tab2:
                    st.subheader("🎯 Find Optimal Metric for Your Use Case")
                    st.markdown("Get recommendations on which metric to optimize based on your cost structure.")
    
                    # Model selection
                    selected_model = st.selectbox("Select Model", list(binary_models.keys()), key="metric_rec_model")
                    cm_selected = binary_models[selected_model]
    
                    # Cost ratio inputs
                    st.markdown("### Define Cost Ratio")
                    col1, col2 = st.columns(2)
    
                    with col1:
                        cost_fp_ratio = st.number_input("Cost of False Positive", value=1.0, min_value=0.01, step=0.1,
                                                       key="fp_ratio")
                    with col2:
                        cost_fn_ratio = st.number_input("Cost of False Negative", value=10.0, min_value=0.01, step=0.1,
                                                       key="fn_ratio")
    
                    cost_ratio = cost_fn_ratio / cost_fp_ratio if cost_fp_ratio > 0 else 1.0
                    st.info(f"📊 Cost Ratio (FN/FP): **{cost_ratio:.2f}** - "
                           f"False negatives are {cost_ratio:.1f}x {'more' if cost_ratio > 1 else 'less'} costly than false positives")
    
                    # Quick presets
                    st.markdown("**Quick Presets:**")
                    preset_col1, preset_col2, preset_col3 = st.columns(3)
                    with preset_col1:
                        if st.button("🏥 Medical Diagnosis (FN >> FP)", width='content'):
                            cost_fp_ratio = 1.0
                            cost_fn_ratio = 10.0
                            st.rerun()
                    with preset_col2:
                        if st.button("📧 Spam Detection (FP >> FN)", width='content'):
                            cost_fp_ratio = 100.0
                            cost_fn_ratio = 1.0
                            st.rerun()
                    with preset_col3:
                        if st.button("⚖️ Balanced Costs", width='content'):
                            cost_fp_ratio = 1.0
                            cost_fn_ratio = 1.0
                            st.rerun()
    
                    if st.button("Get Recommendation", type="primary"):
                        try:
                            recommendation = cm_selected.find_optimal_metric_for_cost(
                                cost_fp=cost_fp_ratio,
                                cost_fn=cost_fn_ratio
                            )
    
                            st.success("✅ Analysis complete!")
    
                            # Primary recommendation
                            st.markdown("### 🎯 Recommended Metric")
                            primary_metric = recommendation['primary_recommendation']
                            st.markdown(f"### **{primary_metric.upper().replace('_', ' ')}**")
    
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.info(recommendation['explanation'])
                            with col2:
                                current_value = recommendation['current_metric_values'].get(primary_metric, 0)
                                st.metric(f"Current {primary_metric.replace('_', ' ').title()}",
                                         f"{current_value:.4f}")
    
                            # Context and interpretation
                            st.markdown("### 📖 Context")
                            st.write(recommendation['interpretation'])
    
                            # Cost-weighted metric
                            st.markdown("### 📊 Cost-Weighted F-Beta Score")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("F-Beta Score", f"{recommendation['cost_weighted_f_beta']:.4f}")
                            with col2:
                                st.metric("Beta Value", f"{recommendation['beta_value']:.2f}",
                                         help="Beta > 1 weights recall higher, Beta < 1 weights precision higher")
    
                            # Secondary recommendations
                            st.markdown("### 🔄 Secondary Metrics to Monitor")
                            secondary_cols = st.columns(len(recommendation['secondary_recommendations']))
                            for i, metric in enumerate(recommendation['secondary_recommendations']):
                                with secondary_cols[i]:
                                    value = recommendation['current_metric_values'].get(metric, 0)
                                    st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
    
                            # All current metrics
                            with st.expander("📋 All Current Metrics", expanded=False):
                                metrics_df = pd.DataFrame([recommendation['current_metric_values']]).T
                                metrics_df.columns = ['Value']
                                metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
                                st.dataframe(metrics_df, width='content')
    
                        except Exception as e:
                            st.error(f"Error generating recommendation: {str(e)}")
    
                with cost_tab3:
                    st.subheader("⚖️ Compare Models by Cost")
                    st.markdown("Compare two models based on real business costs rather than abstract metrics.")
    
                    if len(binary_models) < 2:
                        st.warning("Need at least 2 binary classification models for cost comparison.")
                    else:
                        # Model selection
                        col1, col2 = st.columns(2)
                        model_names = list(binary_models.keys())
    
                        with col1:
                            model1_name = st.selectbox("Model A", model_names, key="cost_comp_model1")
                        with col2:
                            model2_name = st.selectbox("Model B", model_names,
                                                      index=min(1, len(model_names)-1),
                                                      key="cost_comp_model2")
    
                        # Cost inputs
                        st.markdown("### Define Cost Structure")
                        col1, col2 = st.columns(2)
    
                        with col1:
                            st.markdown("**Costs**")
                            cost_fp_comp = st.number_input("Cost of False Positive", value=100.0, step=10.0, key="fp_comp")
                            cost_fn_comp = st.number_input("Cost of False Negative", value=1000.0, step=10.0, key="fn_comp")
    
                        with col2:
                            st.markdown("**Benefits** (optional)")
                            cost_tp_comp = st.number_input("Cost/Benefit of True Positive (negative for benefit)",
                                                          value=-50.0, step=10.0, key="tp_comp")
                            cost_tn_comp = st.number_input("Cost/Benefit of True Negative (negative for benefit)",
                                                          value=-10.0, step=10.0, key="tn_comp")
    
                        if model1_name != model2_name:
                            if st.button("Compare Models", type="primary"):
                                cm1 = binary_models[model1_name]
                                cm2 = binary_models[model2_name]
    
                                try:
                                    comparison = cm1.compare_cost_with(
                                        cm2,
                                        cost_fp=cost_fp_comp,
                                        cost_fn=cost_fn_comp,
                                        cost_tp=cost_tp_comp,
                                        cost_tn=cost_tn_comp
                                    )
    
                                    # Winner announcement
                                    if comparison['better_model'] == 'model1':
                                        st.success(f"🏆 **{model1_name}** is more cost-effective!")
                                    elif comparison['better_model'] == 'model2':
                                        st.success(f"🏆 **{model2_name}** is more cost-effective!")
                                    else:
                                        st.info("⚖️ Both models have equal cost")
    
                                    st.markdown(f"**{comparison['recommendation']}**")
    
                                    # Cost comparison
                                    st.markdown("### 💰 Cost Comparison")
                                    col1, col2, col3 = st.columns(3)
    
                                    with col1:
                                        st.metric(f"{model1_name} Total Cost",
                                                 f"${comparison['model1_total_cost']:,.0f}")
                                        st.caption(f"Avg: ${comparison['model1_average_cost']:.2f}/sample")
    
                                    with col2:
                                        st.metric(f"{model2_name} Total Cost",
                                                 f"${comparison['model2_total_cost']:,.0f}")
                                        st.caption(f"Avg: ${comparison['model2_average_cost']:.2f}/sample")
    
                                    with col3:
                                        savings = comparison['cost_savings']
                                        savings_pct = comparison['relative_savings_percent']
                                        st.metric("Cost Savings",
                                                 f"${abs(savings):,.0f}",
                                                 delta=f"{savings_pct:.1f}%")
                                        if savings > 0:
                                            st.caption(f"{model1_name} saves ${savings:,.0f}")
                                        elif savings < 0:
                                            st.caption(f"{model2_name} saves ${abs(savings):,.0f}")
    
                                    # Detailed breakdown
                                    st.markdown("### 📊 Detailed Breakdown")
    
                                    # Create comparison table
                                    comparison_data = {
                                        'Metric': ['TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall'],
                                        model1_name: [
                                            cm1.true_positive, cm1.true_negative,
                                            cm1.false_positive, cm1.false_negative,
                                            f"{cm1.get_accuracy():.4f}",
                                            f"{cm1.get_precision():.4f}",
                                            f"{cm1.get_recall():.4f}"
                                        ],
                                        model2_name: [
                                            cm2.true_positive, cm2.true_negative,
                                            cm2.false_positive, cm2.false_negative,
                                            f"{cm2.get_accuracy():.4f}",
                                            f"{cm2.get_precision():.4f}",
                                            f"{cm2.get_recall():.4f}"
                                        ]
                                    }
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df, width='content', hide_index=True)
    
                                    # Cost structure summary
                                    with st.expander("📋 Cost Structure Details", expanded=False):
                                        st.write(f"- False Positive Cost: ${comparison['cost_structure']['cost_fp']:,.2f}")
                                        st.write(f"- False Negative Cost: ${comparison['cost_structure']['cost_fn']:,.2f}")
                                        st.write(f"- True Positive Cost: ${comparison['cost_structure']['cost_tp']:,.2f}")
                                        st.write(f"- True Negative Cost: ${comparison['cost_structure']['cost_tn']:,.2f}")
                                        st.write(f"- Cost Ratio (FN/FP): {comparison['cost_structure']['cost_ratio_fn_to_fp']:.2f}")
    
                                except Exception as e:
                                    st.error(f"Error comparing models: {str(e)}")
                        else:
                            st.warning("Please select two different models to compare.")
    
            # Information section
            with st.expander("ℹ️ About Cost-Sensitive Analysis", expanded=False):
                st.markdown("""
                ### What is Cost-Sensitive Analysis?
    
                Not all classification errors are equal. In the real world, different mistakes have different costs:
    
                - **Medical Diagnosis**: Missing a disease (FN) can be fatal, while a false alarm (FP) is inconvenient
                - **Spam Detection**: Blocking legitimate email (FP) is worse than letting spam through (FN)
                - **Fraud Detection**: Missing fraud (FN) loses money, but investigating legitimate transactions (FP) annoys customers
    
                Cost-sensitive analysis helps you:
                1. **Calculate total business impact** of your model's predictions
                2. **Identify the right metric to optimize** for your specific use case
                3. **Compare models** based on real costs, not abstract metrics
    
                ### Research Foundation
    
                - **Elkan (2001)** - The foundations of cost-sensitive learning
                - **Ling & Sheng (2008)** - Cost-sensitive learning and the class imbalance problem
                - **Drummond & Holte (2006)** - Cost curves for classifier performance
    
                ### When to Use This
    
                Use cost-sensitive analysis when:
                - Different error types have significantly different consequences
                - You need to justify model selection to stakeholders in business terms
                - Traditional metrics (accuracy, F1) don't capture what matters to your use case
                - You're working with imbalanced data where minority class errors are critical
                """)
    
        with tab6:
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

# Metric Completion Tab (always available, independent of models)
with tab_metric_completion:
    st.header("🔍 Metric Completion")

    st.markdown("""
    Reconstruct confusion matrices from partial metrics or infer missing metrics with confidence intervals.
    **No existing models required** - this tool works standalone!

    Perfect for:
    - 📄 Reproducing results from research papers
    - 🔬 Understanding uncertainty in incomplete data
    - 📊 Analyzing what's possible with limited information
    """)

    # Two subtabs
    completion_tab1, completion_tab2 = st.tabs(["🎯 Exact Reconstruction", "📊 Probabilistic Inference"])

    with completion_tab1:
        st.subheader("from_metrics() - Exact Reconstruction")
        st.markdown("Reconstruct a complete confusion matrix when you have **3+ metrics** reported.")

        col1, col2 = st.columns([2, 1])

        with col1:
            total_samples = st.number_input("Total Samples", min_value=1, value=100, step=1, key="from_total")

            st.markdown("#### Select Metrics (need at least 3)")
            met_col1, met_col2 = st.columns(2)

            with met_col1:
                use_acc = st.checkbox("Accuracy", key="from_acc_cb")
                accuracy = st.slider("Value", 0.0, 1.0, 0.85, 0.01, key="from_acc_val", disabled=not use_acc) if use_acc else None

                use_prec = st.checkbox("Precision", key="from_prec_cb")
                precision = st.slider("Value", 0.0, 1.0, 0.80, 0.01, key="from_prec_val", disabled=not use_prec) if use_prec else None

                use_rec = st.checkbox("Recall", key="from_rec_cb")
                recall = st.slider("Value", 0.0, 1.0, 0.75, 0.01, key="from_rec_val", disabled=not use_rec) if use_rec else None

            with met_col2:
                use_spec = st.checkbox("Specificity", key="from_spec_cb")
                specificity = st.slider("Value", 0.0, 1.0, 0.90, 0.01, key="from_spec_val", disabled=not use_spec) if use_spec else None

                use_f1 = st.checkbox("F1 Score", key="from_f1_cb")
                f1_score = st.slider("Value", 0.0, 1.0, 0.77, 0.01, key="from_f1_val", disabled=not use_f1) if use_f1 else None

                use_prev = st.checkbox("Prevalence", key="from_prev_cb")
                prevalence = st.slider("Value", 0.0, 1.0, 0.40, 0.01, key="from_prev_val", disabled=not use_prev) if use_prev else None

            selected_count = sum([use_acc, use_prec, use_rec, use_spec, use_f1, use_prev])

            if selected_count < 3:
                st.warning(f"⚠️ Need at least 3 metrics. Currently selected: {selected_count}")
            else:
                st.success(f"✅ {selected_count} metrics selected")

            if st.button("🔍 Reconstruct", type="primary", disabled=selected_count < 3):
                try:
                    cm = DConfusion.from_metrics(
                        total_samples=total_samples,
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        specificity=specificity,
                        f1_score=f1_score,
                        prevalence=prevalence
                    )

                    st.success("✅ Successfully reconstructed!")

                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric("TP", cm.true_positive)
                        st.metric("FN", cm.false_negative)
                    with res_col2:
                        st.metric("FP", cm.false_positive)
                        st.metric("TN", cm.true_negative)
                    with res_col3:
                        st.metric("Total", cm.total)
                        st.metric("Accuracy", f"{cm.get_accuracy():.3f}")

                    st.markdown("#### All Computed Metrics")
                    metrics = cm.get_all_metrics()
                    metrics_df = pd.DataFrame([{"Metric": k.replace('_', ' ').title(), "Value": f"{v:.4f}"} for k, v in metrics.items()])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                    add_name = st.text_input("Add to comparison as:", value="Reconstructed", key="add_recon_name")
                    if st.button("➕ Add to Model Comparison", key="add_recon_btn"):
                        st.session_state.matrices[add_name] = cm
                        st.success(f"✅ Added '{add_name}'! Switch to Model Comparison tab to view.")
                        st.rerun()

                except ValueError as e:
                    st.error(f"❌ {str(e)}")

        with col2:
            st.info("""
            **Good combinations:**
            - Precision + Recall + Prevalence
            - Accuracy + Recall + Prevalence
            - Recall + Specificity + Prevalence

            **Requirements:**
            - At least 3 metrics
            - Metrics must be consistent
            """)

    with completion_tab2:
        st.subheader("infer_metrics() - Probabilistic Inference")
        st.markdown("Infer missing metrics with confidence intervals using **2+ metrics**.")

        col1, col2 = st.columns([2, 1])

        with col1:
            infer_total = st.number_input("Total Samples", min_value=1, value=100, step=1, key="infer_total")

            st.markdown("#### Select Known Metrics (need at least 2)")
            inf_col1, inf_col2 = st.columns(2)

            with inf_col1:
                infer_use_acc = st.checkbox("Accuracy", value=True, key="infer_acc_cb")
                infer_accuracy = st.slider("Value", 0.0, 1.0, 0.85, 0.01, key="infer_acc_val", disabled=not infer_use_acc) if infer_use_acc else None

                infer_use_prec = st.checkbox("Precision", key="infer_prec_cb")
                infer_precision = st.slider("Value", 0.0, 1.0, 0.80, 0.01, key="infer_prec_val", disabled=not infer_use_prec) if infer_use_prec else None

                infer_use_rec = st.checkbox("Recall", key="infer_rec_cb")
                infer_recall = st.slider("Value", 0.0, 1.0, 0.75, 0.01, key="infer_rec_val", disabled=not infer_use_rec) if infer_use_rec else None

            with inf_col2:
                infer_use_spec = st.checkbox("Specificity", key="infer_spec_cb")
                infer_specificity = st.slider("Value", 0.0, 1.0, 0.90, 0.01, key="infer_spec_val", disabled=not infer_use_spec) if infer_use_spec else None

                infer_use_prev = st.checkbox("Prevalence", value=True, key="infer_prev_cb")
                infer_prevalence = st.slider("Value", 0.0, 1.0, 0.40, 0.01, key="infer_prev_val", disabled=not infer_use_prev) if infer_use_prev else None

            st.markdown("#### Simulation Parameters")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                conf_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            with sim_col2:
                n_sims = st.select_slider("Simulations", options=[1000, 2500, 5000, 10000], value=5000)

            infer_count = sum([infer_use_acc, infer_use_prec, infer_use_rec, infer_use_spec, infer_use_prev])

            if infer_count < 2:
                st.warning(f"⚠️ Need at least 2 metrics. Currently: {infer_count}")
            else:
                st.success(f"✅ {infer_count} metrics selected")

            if st.button("📊 Infer", type="primary", disabled=infer_count < 2):
                try:
                    with st.spinner(f"Running {n_sims:,} simulations..."):
                        result = DConfusion.infer_metrics(
                            total_samples=infer_total,
                            accuracy=infer_accuracy,
                            precision=infer_precision,
                            recall=infer_recall,
                            specificity=infer_specificity,
                            prevalence=infer_prevalence,
                            confidence_level=conf_level,
                            n_simulations=n_sims,
                            random_state=42
                        )

                    st.success(f"✅ Generated {result['n_valid_samples']:,}/{n_sims:,} valid matrices")

                    st.markdown("#### Provided Metrics")
                    prov_df = pd.DataFrame([{"Metric": k.title(), "Value": f"{v:.3f}"} for k, v in result['provided_metrics'].items()])
                    st.dataframe(prov_df, use_container_width=True, hide_index=True)

                    st.markdown(f"#### Inferred Metrics ({conf_level*100:.0f}% CI)")
                    inferred_data = []
                    for name, stats in result['inferred_metrics'].items():
                        inferred_data.append({
                            "Metric": name.replace('_', ' ').title(),
                            "Mean": f"{stats['mean']:.3f}",
                            f"{conf_level*100:.0f}% CI": f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]",
                            "Std": f"{stats['std']:.3f}"
                        })
                    inferred_df = pd.DataFrame(inferred_data)
                    st.dataframe(inferred_df, use_container_width=True, hide_index=True)

                    st.info("Wide confidence intervals = high uncertainty from limited info")

                except ValueError as e:
                    st.error(f"❌ {str(e)}")

        with col2:
            st.info("""
            **This method:**
            - Generates many possible matrices
            - Keeps those matching constraints
            - Provides confidence intervals

            **More simulations =**
            - Better estimates
            - Slower computation
            """)

    # Help section
    with st.expander("ℹ️ Learn More"):
        st.markdown("""
        ## Metric Completion

        Work with incomplete confusion matrix information:

        ### Two Approaches

        **1. Exact Reconstruction**
        - Finds exact confusion matrix
        - Requires 3+ metrics
        - Fast & deterministic
        - Best for paper reproduction

        **2. Probabilistic Inference**
        - Estimates with confidence intervals
        - Requires 2+ metrics
        - Quantifies uncertainty
        - Best for incomplete data

        ### Use Cases

        📄 **Paper:** "85% accuracy, 80% precision, 75% recall"
        → Use exact reconstruction

        🏥 **Study:** "85% accuracy, 30% prevalence"
        → Use probabilistic inference to estimate sensitivity/PPV
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with DConfusion 📊 | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
