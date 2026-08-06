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
                    "color": "#1D1D1F"
                }
            },

            title={
                "text": "ATS Match Score",
                "font": {
                    "size": 24,
                    "color": "#1D1D1F"
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#1D1D1F"
                },

                "bar": {
                    "color": "#007AFF"
                },

                "steps": [
                    {"range": [0, 60], "color": "#FFE5E5"},
                    {"range": [60, 75], "color": "#FFF4CC"},
                    {"range": [75, 90], "color": "#D8F5D0"},
                    {"range": [90, 100], "color": "#CDE8FF"},
                ]
            }
        )
    )

    fig.update_layout(
    paper_bgcolor="#F5F5F7",
    plot_bgcolor="white",
    height=500,

    font=dict(
        color="#1D1D1F",
        size=14
    ),

    xaxis=dict(
        title="Value",
        title_font=dict(color="#1D1D1F"),
        tickfont=dict(color="#1D1D1F")
    ),

    yaxis=dict(
        title="Feature",
        title_font=dict(color="#1D1D1F"),
        tickfont=dict(color="#1D1D1F")
    ),

    coloraxis_colorbar=dict(
        title="Value",
        tickfont=dict(color="#1D1D1F"),
        title_font=dict(color="#1D1D1F")
    )
)
    st.plotly_chart(
    fig,
    use_container_width=True
)