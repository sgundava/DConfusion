"""
Statistical Testing Tab Component

Provides bootstrap confidence intervals and McNemar's test for model comparison.
"""

import streamlit as st


def render_statistical_testing_tab(matrices):
    """
    Render the statistical testing tab.

    Args:
        matrices: Dictionary of {name: DConfusion} confusion matrices
    """
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
    if matrices:
        model_names = list(matrices.keys())
        selected_model = st.selectbox("Select Model", model_names, key="ci_model")
        cm_selected = matrices[selected_model]

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

    if len(matrices) >= 2:
        binary_models = {name: cm for name, cm in matrices.items() if cm.n_classes == 2}

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
