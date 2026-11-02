"""
AI Innovation and the U.S. Labor Market
Interactive Dash dashboard with five visualizations.
Run: python -m pip install -r requirements.txt
     python app.py
Open http://127.0.0.1:8050
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

DATA_FILE = Path("data/AI_BLS_AIOE_2020_2023_final.csv")
YEARS = [2020, 2021, 2022, 2023]
EXPOSURE_GROUPS = ["Lowest exposure", "Lower-middle exposure",
                   "Upper-middle exposure", "Highest exposure"]
COLORS = ["#440154", "#31688e", "#35b779", "#fde725"]


def load_data():
    """Read the combined BLS, AI-patent, and AIOE dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Download the analysis CSV into the data folder."
        )
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["AIOE"]).copy()

    for col in ["TOT_EMP", "A_MEAN", "AI_patents", "AIOE", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log scale makes patent-count differences easier to display.
    df["ln_ai_patents"] = np.log1p(df["AI_patents"].clip(lower=0))

    # Create exposure quartiles once per occupation, not once per row.
    scores = df[["OCC_CODE", "AIOE"]].drop_duplicates("OCC_CODE").sort_values("AIOE")
    scores["exposure_group"] = pd.qcut(scores["AIOE"], 4, labels=EXPOSURE_GROUPS)
    return df.merge(scores[["OCC_CODE", "exposure_group"]],
                    on="OCC_CODE", how="left", validate="many_to_one")


def filter_rows(df, state, year):
    """Apply the dashboard's shared state and year selections."""
    out = df
    if state != "All states":
        out = out[out["PRIM_STATE"] == state]
    if year != "All years":
        out = out[out["year"] == int(year)]
    return out


def add_descriptive_trendline(fig, data, x, y):
    """Add an ordinary least-squares descriptive line (not a causal estimate)."""
    valid = data[[x, y]].dropna()
    if len(valid) >= 2 and valid[x].nunique() > 1:
        slope, intercept = np.polyfit(valid[x], valid[y], 1)
        xs = np.linspace(valid[x].min(), valid[x].max(), 100)
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines",
            name="Descriptive linear trend",
            line=dict(color="#222222", width=3, dash="dash")
        ))
    return fig


# VISUAL 1: Geographic distribution of AI patents.
def chart_patent_map(df, year):
    map_year = 2023 if year == "All years" else int(year)
    state_year = (df[df["year"] == map_year]
                  .groupby("PRIM_STATE", as_index=False)
                  .agg(AI_patents=("AI_patents", "first"),
                       unique_patents=("unique_patents", "first")))
    fig = px.choropleth(
        state_year, locations="PRIM_STATE", locationmode="USA-states",
        color="AI_patents", scope="usa", hover_name="PRIM_STATE",
        hover_data={"AI_patents": ":,.2f", "unique_patents": ":,.0f"},
        color_continuous_scale="Blues",
        title=f"1. AI Patent Activity by State — {map_year}"
    )
    fig.update_layout(template="plotly_white", height=520, title_x=.02)
    return fig


# VISUAL 2: Occupations with the lowest and highest AIOE scores.
def chart_occupation_exposure(df):
    occ = df[["OCC_CODE", "OCC_TITLE", "AIOE"]].drop_duplicates("OCC_CODE")
    extremes = pd.concat([occ.nsmallest(15, "AIOE"),
                          occ.nlargest(15, "AIOE")]).drop_duplicates("OCC_CODE")
    fig = px.bar(
        extremes.sort_values("AIOE"), x="AIOE", y="OCC_TITLE",
        orientation="h", color="AIOE", color_continuous_scale="Viridis",
        hover_data=["OCC_CODE"],
        title="2. Occupations with the Lowest and Highest AI Exposure",
        labels={"AIOE": "AI occupational exposure score", "OCC_TITLE": "Occupation"}
    )
    fig.update_layout(template="plotly_white", height=650, title_x=.02,
                      yaxis={"categoryorder": "total ascending"})
    return fig


# VISUAL 3: Relationship between state AI patents and occupational employment.
def chart_employment_scatter(df):
    d = df.dropna(subset=["ln_ai_patents", "ln_employment", "AIOE"]).copy()
    fig = px.scatter(
        d, x="ln_ai_patents", y="ln_employment", color="AIOE",
        hover_name="OCC_TITLE", opacity=.45, color_continuous_scale="Viridis",
        hover_data={"PRIM_STATE": True, "year": True, "OCC_CODE": True,
                    "AI_patents": ":,.2f", "AIOE": ":.3f"},
        title="3. State AI Innovation and Occupational Employment",
        labels={"ln_ai_patents": "ln(1 + fractional AI patents)",
                "ln_employment": "ln(occupational employment)", "AIOE": "AI exposure"}
    )
    fig = add_descriptive_trendline(fig, d, "ln_ai_patents", "ln_employment")
    fig.update_layout(template="plotly_white", height=570, title_x=.02,
                      coloraxis_colorbar_title="AIOE")
    return fig


# VISUAL 4: Relationship between state AI patents and occupational wages.
def chart_wage_scatter(df):
    d = df.dropna(subset=["ln_ai_patents", "A_MEAN", "AIOE"]).copy()
    fig = px.scatter(
        d, x="ln_ai_patents", y="A_MEAN", color="AIOE",
        hover_name="OCC_TITLE", opacity=.45, color_continuous_scale="Viridis",
        hover_data={"PRIM_STATE": True, "year": True, "OCC_CODE": True,
                    "AI_patents": ":,.2f", "A_MEAN": ":$,.0f", "AIOE": ":.3f"},
        title="4. State AI Innovation and Occupational Mean Wages",
        labels={"ln_ai_patents": "ln(1 + fractional AI patents)",
                "A_MEAN": "Annual mean wage ($)", "AIOE": "AI exposure"}
    )
    fig = add_descriptive_trendline(fig, d, "ln_ai_patents", "A_MEAN")
    fig.update_layout(template="plotly_white", height=570, title_x=.02,
                      coloraxis_colorbar_title="AIOE",
                      yaxis=dict(tickprefix="$", tickformat=","))
    return fig


# VISUAL 5: Percent change in average employment since the 2020 baseline.
def chart_employment_growth(df, state):
    d = df.dropna(subset=["TOT_EMP", "exposure_group"]).copy()
    if state != "All states":
        d = d[d["PRIM_STATE"] == state]

    avg = (d.groupby(["year", "exposure_group"], observed=False, as_index=False)
           .agg(mean_employment=("TOT_EMP", "mean")))
    baseline = (avg[avg["year"] == 2020][["exposure_group", "mean_employment"]]
                .rename(columns={"mean_employment": "baseline_2020"}))
    avg = avg.merge(baseline, on="exposure_group", how="left")
    avg["percent_change"] = (avg["mean_employment"] / avg["baseline_2020"] - 1) * 100

    fig = px.line(
        avg, x="year", y="percent_change", color="exposure_group", markers=True,
        color_discrete_sequence=COLORS,
        hover_data={"mean_employment": ":,.0f",
                    "baseline_2020": ":,.0f", "percent_change": ":.2f"},
        title="5. Employment Change Since 2020 by AI-Exposure Group",
        labels={"year": "Year", "percent_change": "Change since 2020 (%)",
                "exposure_group": "AI exposure group"}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555555",
                  annotation_text="2020 baseline")
    fig.update_layout(template="plotly_white", height=520, title_x=.02,
                      xaxis=dict(dtick=1))
    return fig


def create_app(df):
    app = Dash(__name__)
    app.title = "AI and the Labor Market"

    state_options = [{"label": "All states", "value": "All states"}] + [
        {"label": s, "value": s} for s in sorted(df["PRIM_STATE"].dropna().unique())
    ]
    year_options = [{"label": "All years", "value": "All years"}] + [
        {"label": str(y), "value": y} for y in YEARS
    ]

    card = {"backgroundColor": "white", "padding": "10px",
            "borderRadius": "10px", "marginBottom": "18px"}

    app.layout = html.Div(
        style={"fontFamily": "Arial", "maxWidth": "1450px",
               "margin": "0 auto", "padding": "24px",
               "backgroundColor": "#f7f9fc"},
        children=[
            html.H1("AI Innovation and the U.S. Labor Market",
                    style={"color": "#203354"}),
            html.P("Explore state AI patent activity, occupational AI exposure, "
                   "employment, and wages from 2020–2023."),
            html.Div(style={**card, "display": "flex", "gap": "24px",
                            "flexWrap": "wrap"},
                     children=[
                html.Div([html.Label("Select a state"),
                          dcc.Dropdown(id="state-filter", options=state_options,
                                       value="All states", clearable=False)],
                         style={"width": "240px"}),
                html.Div([html.Label("Select a year"),
                          dcc.Dropdown(id="year-filter", options=year_options,
                                       value="All years", clearable=False)],
                         style={"width": "200px"})
            ]),
            html.Div(dcc.Graph(id="patent-map"), style=card),
            html.Div(dcc.Graph(id="exposure-chart",
                               figure=chart_occupation_exposure(df)), style=card),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit,minmax(420px,1fr))",
                            "gap": "18px"},
                     children=[
                html.Div(dcc.Graph(id="employment-scatter"), style=card),
                html.Div(dcc.Graph(id="wage-scatter"), style=card)
            ]),
            html.Div(dcc.Graph(id="employment-growth"), style=card),
            html.Footer("Sources: BLS OEWS, AI patent data, and Felten, Raj, and "
                        "Seamans AIOE. Descriptive associations do not establish "
                        "causation. BLS OEWS estimates have limitations for direct "
                        "year-to-year comparisons.",
                        style={"fontSize": "12px", "color": "#64748b"})
        ]
    )

    @app.callback(
        Output("patent-map", "figure"),
        Output("employment-scatter", "figure"),
        Output("wage-scatter", "figure"),
        Output("employment-growth", "figure"),
        Input("state-filter", "value"),
        Input("year-filter", "value")
    )
    def update_charts(state, year):
        filtered = filter_rows(df, state, year)
        return (
            chart_patent_map(df, year),
            chart_employment_scatter(filtered),
            chart_wage_scatter(filtered),
            chart_employment_growth(df, state)
        )

    return app


def main():
    data = load_data()
    app = create_app(data)
    print("Dashboard running at http://127.0.0.1:8050")
    app.run(debug=False)


if __name__ == "__main__":
    main()
