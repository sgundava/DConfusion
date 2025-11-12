"""
Metrics Display Tab Component

Shows side-by-side metrics comparison and best performers analysis.
"""

import streamlit as st
import pandas as pd


def render_metrics_tab(matrices):
    """
    Render the metrics comparison tab.

    Args:
        matrices: Dictionary of {name: DConfusion} confusion matrices
    """
    st.header("Metrics Comparison")

    # Collect all metrics
    metrics_data = {}
    for name, cm in matrices.items():
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
