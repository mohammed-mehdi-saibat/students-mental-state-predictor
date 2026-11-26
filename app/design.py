import plotly.graph_objects as go
import streamlit as st

NAVY = "#242217"
GOLD = "#b08a08"
COLORS = ["#f7df1e", "#bd9714", "#554825"]


def heading(title, description):
    if title:
        st.title(title)
    st.markdown(f'<p class="intro">{description}</p>', unsafe_allow_html=True)


def plot(figure, *, height=310):
    figure.update_layout(
        height=height,
        font=dict(color=NAVY, family="Arial, sans-serif", size=13),
        margin=dict(l=15, r=20, t=20, b=35),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend_title_text="",
    )
    figure.update_xaxes(gridcolor="#ece7ce", zeroline=False)
    figure.update_yaxes(gridcolor="#ece7ce", zeroline=False)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def counts(labels, values):
    return go.Figure(go.Bar(
        x=labels, y=values, marker_color=COLORS, text=values,
        textposition="outside", cliponaxis=False,
    )).update_yaxes(rangemode="tozero", title="Records")
