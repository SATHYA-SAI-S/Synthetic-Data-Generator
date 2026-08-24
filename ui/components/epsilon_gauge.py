"""
SYNTHGUARD Component - Differential Privacy Epsilon Gauge
Fully adaptive: shows the actual spent epsilon or a pending placeholder.
"""
import streamlit as st
import plotly.graph_objects as go


def render_epsilon_gauge(spent=None, target: float = 1.0):
    """Render a semi-circular gauge of privacy budget expenditure (adaptive)."""
    if spent is None:
        # No run yet - show placeholder rather than fabricating a number.
        spent_val = 0.0
        display = "Pending"
        range_max = float(target * 1.2)
        ref = float(target)
    else:
        spent_val = float(spent)
        display = f"{spent_val:.4f}"
        range_max = float(target * 1.2)
        ref = float(target)

    pct_used = min(100.0, (spent_val / target) * 100.0) if target > 0 else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=spent_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Privacy Budget Spent (Epsilon)", 'font': {'size': 16, 'color': '#F8FAFC'}},
        number={'suffix': "", 'font': {'color': '#F8FAFC', 'size': 24}},
        gauge={
            'axis': {'range': [None, range_max], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#1A73E8"},
            'bgcolor': "#1B2A4A",
            'borderwidth': 2,
            'bordercolor': "#2A3F60",
            'steps': [
                {'range': [0, target * 0.5], 'color': 'rgba(34, 197, 94, 0.25)'},
                {'range': [target * 0.5, target * 0.85], 'color': 'rgba(245, 158, 11, 0.25)'},
                {'range': [target * 0.85, target * 1.2], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 4},
                'thickness': 0.75,
                'value': ref
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#F8FAFC", 'family': "Inter"},
        height=240,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, width='stretch')