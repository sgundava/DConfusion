"""
Cost-Sensitive Analysis Tab Component

Provides cost calculation, metric recommendation, and model comparison based on business costs.
"""

import streamlit as st
import pandas as pd


def render_cost_analysis_tab(matrices):
    """
    Render the cost-sensitive analysis tab.

    Args:
        matrices: Dictionary of {name: DConfusion} confusion matrices
    """
    st.header("💰 Cost-Sensitive Analysis")
    st.markdown("""
    Evaluate your models based on real-world costs and benefits. Different classification errors
    have different business impacts - optimize for what matters to your use case.
    """)

    # Filter binary models only
    binary_models = {name: cm for name, cm in matrices.items() if cm.n_classes == 2}

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
