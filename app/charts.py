import plotly.graph_objects as go
import streamlit as st


def score_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,

            number={
                "suffix": "%",
                "font": {
                    "size": 48,
                    "color": "#0A1628",
                    "family": "Inter, sans-serif",
                }
            },

            title={
                "text": "ATS Match Score",
                "font": {
                    "size": 22,
                    "color": "#475569",
                    "family": "Inter, sans-serif",
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#94A3B8",
                    "tickwidth": 1,
                    "dtick": 20,
                    "tickfont": {"color": "#64748B", "size": 12},
                },

                "bar": {
                    "color": "#0077B6",
                    "thickness": 0.75,
                },

                "bgcolor": "#F1F5F9",
                "borderwidth": 0,

                "steps": [
                    {"range": [0, 40],  "color": "#FEE2E2"},
                    {"range": [40, 60], "color": "#FEF3C7"},
                    {"range": [60, 75], "color": "#FEF9C3"},
                    {"range": [75, 90], "color": "#D1FAE5"},
                    {"range": [90, 100],"color": "#DBEAFE"},
                ],

                "threshold": {
                    "line": {"color": "#0077B6", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        font=dict(
            color="#0A1628",
            size=14,
            family="Inter, sans-serif",
        ),
        margin=dict(t=40, b=20, l=40, r=40),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )