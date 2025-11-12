"""
Metric Completion Tab Component

Provides metric completion tools: exact reconstruction and probabilistic inference.
"""

import streamlit as st
import pandas as pd
from dconfusion import DConfusion


def render_metric_completion_tab():
    """
    Render the metric completion tab.

    Note: This tab doesn't require any matrices parameter as it works independently.
    """
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
                    st.dataframe(metrics_df, width='stretch', hide_index=True)

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
                    st.dataframe(prov_df, width='stretch', hide_index=True)

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
                    st.dataframe(inferred_df, width='stretch', hide_index=True)

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
