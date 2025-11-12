"""Warning display components for DConfusion Streamlit App."""

import streamlit as st
from dconfusion import WarningSeverity, check_comparison_validity


def display_warnings(warnings):
    """
    Display warnings in a formatted way.

    Args:
        warnings: List of Warning objects from DConfusion
    """
    for w in warnings:
        st.markdown(f"**{w.category}**")
        st.markdown(f"*{w.message}*")
        if w.recommendation:
            st.info(f"💡 **Recommendation:** {w.recommendation}")
        st.markdown("---")


def render_warnings_tab(matrices):
    """
    Render the Warnings & Quality tab.

    Args:
        matrices: Dictionary of matrix_name -> DConfusion instances
    """
    st.header("⚠️ Data Quality Warnings")
    st.markdown("""
    This section checks for common pitfalls in confusion matrix analysis based on peer-reviewed research.
    Warnings help identify issues with sample size, class imbalance, metric reliability, and more.
    """)

    # Summary statistics
    total_matrices = len(matrices)
    matrices_with_critical = 0
    matrices_with_warnings = 0
    matrices_clean = 0

    for name, cm in matrices.items():
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
    for name, cm in matrices.items():
        warnings = cm.check_warnings(include_info=False)
        critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
        warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

        # Determine status icon
        if critical:
            status = "🔴"
            status_text = "CRITICAL ISSUES"
        elif warning_level:
            status = "🟡"
            status_text = "WARNINGS"
        else:
            status = "🟢"
            status_text = "NO ISSUES"

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
    if len(matrices) >= 2:
        st.markdown("---")
        st.subheader("🔄 Model Comparison Warnings")
        st.markdown("Check if comparing these models is statistically meaningful.")

        matrix_list = list(matrices.items())
        for i in range(len(matrix_list)):
            for j in range(i + 1, len(matrix_list)):
                name1, cm1 = matrix_list[i]
                name2, cm2 = matrix_list[j]

                # Only compare if both are same type (binary or multi-class)
                if cm1.n_classes == cm2.n_classes == 2:
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
