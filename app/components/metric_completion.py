"""
Metric Completion Tab Component

Provides metric completion tools: exact reconstruction and probabilistic inference.
"""

import streamlit as st
import pandas as pd
from dconfusion import DConfusion
from utils.session import add_matrix


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
            met_col1, met_col2, met_col3 = st.columns(3)

            with met_col1:
                st.markdown("**Primary Metrics**")
                use_acc = st.checkbox("Accuracy", key="from_acc_cb")
                accuracy = st.slider("Value", 0.0, 1.0, 0.85, 0.01, key="from_acc_val", disabled=not use_acc, label_visibility="collapsed") if use_acc else None

                use_prec = st.checkbox("Precision (PPV)", key="from_prec_cb")
                precision = st.slider("Value", 0.0, 1.0, 0.80, 0.01, key="from_prec_val", disabled=not use_prec, label_visibility="collapsed") if use_prec else None

                use_rec = st.checkbox("Recall (TPR)", key="from_rec_cb")
                recall = st.slider("Value", 0.0, 1.0, 0.75, 0.01, key="from_rec_val", disabled=not use_rec, label_visibility="collapsed") if use_rec else None

                use_spec = st.checkbox("Specificity (TNR)", key="from_spec_cb")
                specificity = st.slider("Value", 0.0, 1.0, 0.90, 0.01, key="from_spec_val", disabled=not use_spec, label_visibility="collapsed") if use_spec else None

                use_prev = st.checkbox("Prevalence", key="from_prev_cb")
                prevalence = st.slider("Value", 0.0, 1.0, 0.40, 0.01, key="from_prev_val", disabled=not use_prev, label_visibility="collapsed") if use_prev else None

            with met_col2:
                st.markdown("**Predictive Values**")
                use_npv = st.checkbox("NPV", key="from_npv_cb")
                npv = st.slider("Value", 0.0, 1.0, 0.90, 0.01, key="from_npv_val", disabled=not use_npv, label_visibility="collapsed") if use_npv else None

                st.markdown("**Error Rates**")
                use_fpr = st.checkbox("FPR (Type I)", key="from_fpr_cb")
                fpr = st.slider("Value", 0.0, 1.0, 0.10, 0.01, key="from_fpr_val", disabled=not use_fpr, label_visibility="collapsed") if use_fpr else None

                use_fnr = st.checkbox("FNR (Type II)", key="from_fnr_cb")
                fnr = st.slider("Value", 0.0, 1.0, 0.15, 0.01, key="from_fnr_val", disabled=not use_fnr, label_visibility="collapsed") if use_fnr else None

                use_err = st.checkbox("Error Rate", key="from_err_cb")
                error_rate = st.slider("Value", 0.0, 1.0, 0.15, 0.01, key="from_err_val", disabled=not use_err, label_visibility="collapsed") if use_err else None

            with met_col3:
                st.markdown("**Composite Metrics**")
                use_f1 = st.checkbox("F1 Score", key="from_f1_cb")
                f1_score = st.slider("Value", 0.0, 1.0, 0.77, 0.01, key="from_f1_val", disabled=not use_f1, label_visibility="collapsed") if use_f1 else None

            selected_count = sum([use_acc, use_prec, use_rec, use_spec, use_f1, use_prev, use_npv, use_fpr, use_fnr, use_err])

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
                        prevalence=prevalence,
                        npv=npv,
                        fpr=fpr,
                        fnr=fnr,
                        error_rate=error_rate
                    )

                    # Store in session state so it persists across reruns
                    st.session_state.reconstructed_cm = cm
                    st.success("✅ Successfully reconstructed!")

                except ValueError as e:
                    st.error(f"❌ {str(e)}")

            # Display results if we have a reconstructed matrix
            if 'reconstructed_cm' in st.session_state:
                cm = st.session_state.reconstructed_cm

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

                # Generate default name using model counter
                default_add_name = f"Model {st.session_state.get('model_counter', len(st.session_state.get('matrices', {})) + 1)}"
                add_name = st.text_input("Add to comparison as:", value=default_add_name, key="add_recon_name")
                if st.button("➕ Add to Model Comparison", key="add_recon_btn"):
                    # Use the add_matrix function which handles counter increment
                    add_matrix(add_name, cm)
                    # Clear the reconstructed matrix after adding
                    del st.session_state.reconstructed_cm
                    st.success(f"✅ Added '{add_name}' to Model Comparison!")
                    st.balloons()
                    st.rerun()

        with col2:
            st.info("""
            **Example combinations:**
            - Precision + Recall + Prevalence
            - Accuracy + Recall + Prevalence
            - Recall + Specificity + Prevalence
            - NPV + Specificity + Prevalence
            - FPR + FNR + Prevalence
            - Error Rate + Precision + Recall

            **Requirements:**
            - At least 3 metrics
            - Metrics must be consistent

            **Aliases:**
            - PPV = Precision
            - TPR = Recall
            - TNR = Specificity
            """)

    with completion_tab2:
        st.subheader("infer_metrics() - Probabilistic Inference")
        st.markdown("Infer missing metrics with confidence intervals using **2+ metrics**.")

        col1, col2 = st.columns([2, 1])

        with col1:
            infer_total = st.number_input("Total Samples", min_value=1, value=100, step=1, key="infer_total")

            st.markdown("#### Select Known Metrics (need at least 2)")
            inf_col1, inf_col2, inf_col3 = st.columns(3)

            with inf_col1:
                st.markdown("**Primary Metrics**")
                infer_use_acc = st.checkbox("Accuracy", value=True, key="infer_acc_cb")
                infer_accuracy = st.slider("Value", 0.0, 1.0, 0.85, 0.01, key="infer_acc_val", disabled=not infer_use_acc, label_visibility="collapsed") if infer_use_acc else None

                infer_use_prec = st.checkbox("Precision (PPV)", key="infer_prec_cb")
                infer_precision = st.slider("Value", 0.0, 1.0, 0.80, 0.01, key="infer_prec_val", disabled=not infer_use_prec, label_visibility="collapsed") if infer_use_prec else None

                infer_use_rec = st.checkbox("Recall (TPR)", key="infer_rec_cb")
                infer_recall = st.slider("Value", 0.0, 1.0, 0.75, 0.01, key="infer_rec_val", disabled=not infer_use_rec, label_visibility="collapsed") if infer_use_rec else None

                infer_use_spec = st.checkbox("Specificity (TNR)", key="infer_spec_cb")
                infer_specificity = st.slider("Value", 0.0, 1.0, 0.90, 0.01, key="infer_spec_val", disabled=not infer_use_spec, label_visibility="collapsed") if infer_use_spec else None

                infer_use_prev = st.checkbox("Prevalence", value=True, key="infer_prev_cb")
                infer_prevalence = st.slider("Value", 0.0, 1.0, 0.40, 0.01, key="infer_prev_val", disabled=not infer_use_prev, label_visibility="collapsed") if infer_use_prev else None

            with inf_col2:
                st.markdown("**Predictive Values**")
                infer_use_npv = st.checkbox("NPV", key="infer_npv_cb")
                infer_npv = st.slider("Value", 0.0, 1.0, 0.90, 0.01, key="infer_npv_val", disabled=not infer_use_npv, label_visibility="collapsed") if infer_use_npv else None

                st.markdown("**Error Rates**")
                infer_use_fpr = st.checkbox("FPR (Type I)", key="infer_fpr_cb")
                infer_fpr = st.slider("Value", 0.0, 1.0, 0.10, 0.01, key="infer_fpr_val", disabled=not infer_use_fpr, label_visibility="collapsed") if infer_use_fpr else None

                infer_use_fnr = st.checkbox("FNR (Type II)", key="infer_fnr_cb")
                infer_fnr = st.slider("Value", 0.0, 1.0, 0.15, 0.01, key="infer_fnr_val", disabled=not infer_use_fnr, label_visibility="collapsed") if infer_use_fnr else None

                infer_use_err = st.checkbox("Error Rate", key="infer_err_cb")
                infer_error_rate = st.slider("Value", 0.0, 1.0, 0.15, 0.01, key="infer_err_val", disabled=not infer_use_err, label_visibility="collapsed") if infer_use_err else None

            with inf_col3:
                pass  # Empty column for symmetry

            st.markdown("#### Simulation Parameters")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                conf_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            with sim_col2:
                n_sims = st.select_slider("Simulations", options=[1000, 2500, 5000, 10000], value=5000)

            infer_count = sum([infer_use_acc, infer_use_prec, infer_use_rec, infer_use_spec, infer_use_prev,
                              infer_use_npv, infer_use_fpr, infer_use_fnr, infer_use_err])

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
                            npv=infer_npv,
                            fpr=infer_fpr,
                            fnr=infer_fnr,
                            error_rate=infer_error_rate,
                            confidence_level=conf_level,
                            n_simulations=n_sims,
                            random_state=42
                        )

                    # Store in session state
                    st.session_state.inferred_result = result
                    st.session_state.infer_total_samples = infer_total
                    st.success(f"✅ Generated {result['n_valid_samples']:,}/{n_sims:,} valid matrices")

                except ValueError as e:
                    st.error(f"❌ {str(e)}")

            # Display results if we have inferred metrics
            if 'inferred_result' in st.session_state:
                result = st.session_state.inferred_result
                conf_level = result.get('confidence_level', 0.95)

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

                st.info("💡 Wide confidence intervals = high uncertainty from limited info. These are estimates, not exact values.")

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
