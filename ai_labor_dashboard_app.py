"""
AI Innovation and the Labor Market — Interactive Dashboard
===========================================================

This version improves the original five-visualization project.

Main improvements:
- Visuals 3 and 4 use ln(1 + AI patent count), clearer labels,
  consistent marker styling, useful hover details, and trendlines.
- Shared State and Year filters update the dashboard together.
- Visual 5 shows percentage employment change relative to 2020.
- Each visualization is written as a separate function.

RUN:
    python -m pip install -r requirements_ai_labor_project.txt
    python ai_labor_dashboard_app.py

Then open the local URL printed in the terminal (usually http://127.0.0.1:8050).

DATA:
    AI_BLS_AIOE_2020_2023_final.csv
"""

# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output


# ============================================================
# 2. SETTINGS
# ============================================================

DATA_FILE = "AI_BLS_AIOE_2020_2023_final.csv"
YEARS = [2020, 2021, 2022, 2023]

# A consistent color palette for the entire dashboard.
EXPOSURE_COLORS = [
    "#440154",  # Lowest exposure
    "#31688e",
    "#35b779",
    "#fde725",  # Highest exposure
]

PLOT_TEMPLATE = "plotly_white"


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

def load_data():
    """Load the merged dataset and prepare variables for charts."""

    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Put the CSV in the same folder "
            "as this Python script."
        )

    df = pd.read_csv(data_path)

    # The AIOE score is required for exposure-based comparisons.
    df = df.dropna(subset=["AIOE"]).copy()

    # Keep employment and wage outcomes numeric.
    for column in ["TOT_EMP", "A_MEAN", "AI_patents", "AIOE", "year"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Log transform patent counts. This reduces the visual dominance
    # of states with exceptionally high patent counts.
    # The original patent count is retained in the data and hover details.
    df["ln_ai_patents"] = np.log1p(df["AI_patents"].clip(lower=0))

    # Assign each occupation to one of four exposure groups.
    # First create one row per occupation so each occupation is classified
    # consistently across all states and years.
    occupation_scores = (
        df[["OCC_CODE", "AIOE"]]
        .drop_duplicates(subset="OCC_CODE")
        .sort_values("AIOE")
        .copy()
    )

    occupation_scores["exposure_group"] = pd.qcut(
        occupation_scores["AIOE"],
        q=4,
        labels=[
            "Lowest exposure",
            "Lower-middle exposure",
            "Upper-middle exposure",
            "Highest exposure",
        ],
    )

    df = df.merge(
        occupation_scores[["OCC_CODE", "exposure_group"]],
        on="OCC_CODE",
        how="left",
        validate="many_to_one",
    )

    return df


# ============================================================
# 4. SHARED FILTER FUNCTION
# ============================================================

def filter_data(df, selected_state, selected_year):
    """Apply the shared state and year filters."""

    filtered = df.copy()

    if selected_state != "All states":
        filtered = filtered[filtered["PRIM_STATE"] == selected_state]

    if selected_year != "All years":
        filtered = filtered[filtered["year"] == int(selected_year)]

    return filtered


# ============================================================
# 5. VISUAL 1 — AI PATENT ACTIVITY BY STATE
# ============================================================

def make_ai_patent_map(df, selected_year):
    """
    Create a U.S. choropleth map.

    Color represents fractional AI patent counts for each state.
    The map shows one year at a time, controlled by the shared year filter.
    """

    map_year = 2023 if selected_year == "All years" else int(selected_year)

    # Patent counts repeat across occupations. Keep one state-year value.
    state_data = (
        df[df["year"] == map_year]
        .groupby("PRIM_STATE", as_index=False)
        .agg(
            AI_patents=("AI_patents", "first"),
            unique_patents=("unique_patents", "first"),
        )
    )

    fig = px.choropleth(
        state_data,
        locations="PRIM_STATE",
        locationmode="USA-states",
        color="AI_patents",
        scope="usa",
        hover_name="PRIM_STATE",
        hover_data={
            "AI_patents": ":,.2f",
            "unique_patents": ":,.0f",
        },
        color_continuous_scale="Blues",
        title=f"1. AI Patent Activity by State — {map_year}",
        labels={
            "AI_patents": "Fractional AI patents",
            "unique_patents": "Unique AI patents",
        },
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title_x=0.02,
        height=520,
        margin=dict(l=15, r=15, t=65, b=15),
    )

    return fig


# ============================================================
# 6. VISUAL 2 — OCCUPATIONAL AI EXPOSURE
# ============================================================

def make_occupation_exposure_chart(df):
    """
    Show the 15 occupations with the lowest and highest AIOE scores.

    AIOE is an occupation-level measure, so this chart does not change
    when the user changes the state or year filters.
    """

    occupations = (
        df[["OCC_CODE", "OCC_TITLE", "AIOE"]]
        .drop_duplicates(subset="OCC_CODE")
    )

    lowest = occupations.nsmallest(15, "AIOE")
    highest = occupations.nlargest(15, "AIOE")

    chart_data = (
        pd.concat([lowest, highest])
        .drop_duplicates(subset="OCC_CODE")
        .sort_values("AIOE")
    )

    fig = px.bar(
        chart_data,
        x="AIOE",
        y="OCC_TITLE",
        orientation="h",
        color="AIOE",
        color_continuous_scale="Viridis",
        hover_data=["OCC_CODE"],
        title="2. Occupations with the Lowest and Highest AI Exposure",
        labels={
            "AIOE": "AI Occupational Exposure Score",
            "OCC_TITLE": "Occupation",
        },
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title_x=0.02,
        height=650,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=15, r=20, t=65, b=35),
    )

    return fig


# ============================================================
# 7. HELPER — ADD A LINEAR TRENDLINE
# ============================================================

def add_trendline(fig, chart_data, x_column, y_column):
    """
    Add a simple least-squares trendline to a scatterplot.

    This is a descriptive trendline, not a causal estimate.
    """

    valid = chart_data[[x_column, y_column]].dropna()

    if len(valid) < 2 or valid[x_column].nunique() < 2:
        return fig

    slope, intercept = np.polyfit(
        valid[x_column].to_numpy(),
        valid[y_column].to_numpy(),
        1,
    )

    x_values = np.linspace(
        valid[x_column].min(),
        valid[x_column].max(),
        100,
    )
    y_values = slope * x_values + intercept

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name="Overall linear trend",
            line=dict(color="#222222", width=3, dash="dash"),
            hovertemplate=(
                "Trendline<br>"
                "Log AI patents: %{x:.2f}<br>"
                "Predicted outcome: %{y:.2f}<extra></extra>"
            ),
        )
    )

    return fig


# ============================================================
# 8. VISUAL 3 — AI INNOVATION AND EMPLOYMENT
# ============================================================

def make_employment_scatter(df):
    """
    Improved employment scatterplot.

    X-axis: ln(1 + fractional AI patents)
    Y-axis: log occupational employment
    Color: AIOE score
    Each point: one state-occupation-year observation

    A logarithmic patent scale makes high-patent states less dominant.
    """

    chart_data = df.dropna(
        subset=["ln_ai_patents", "ln_employment", "AIOE"]
    ).copy()

    fig = px.scatter(
        chart_data,
        x="ln_ai_patents",
        y="ln_employment",
        color="AIOE",
        hover_name="OCC_TITLE",
        hover_data={
            "PRIM_STATE": True,
            "year": True,
            "OCC_CODE": True,
            "AI_patents": ":,.2f",
            "AIOE": ":.3f",
            "ln_ai_patents": False,
            "ln_employment": ":.2f",
        },
        color_continuous_scale="Viridis",
        opacity=0.48,
        title="3. State AI Innovation and Occupational Employment",
        labels={
            "ln_ai_patents": "ln(1 + fractional AI patents)",
            "ln_employment": "ln(occupational employment)",
            "AIOE": "AI exposure score",
        },
    )

    fig = add_trendline(
        fig,
        chart_data,
        "ln_ai_patents",
        "ln_employment",
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title_x=0.02,
        height=570,
        legend_title_text="Legend",
        coloraxis_colorbar_title="AIOE",
        xaxis=dict(
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
        ),
        margin=dict(l=65, r=35, t=75, b=65),
    )

    return fig


# ============================================================
# 9. VISUAL 4 — AI INNOVATION AND WAGES
# ============================================================

def make_wage_scatter(df):
    """
    Improved wage scatterplot.

    X-axis: ln(1 + fractional AI patents)
    Y-axis: annual mean wage in dollars
    Color: AIOE score
    Each point: one state-occupation-year observation

    A fixed marker size avoids making large-employment occupations
    visually overwhelm the rest of the observations.
    """

    chart_data = df.dropna(
        subset=["ln_ai_patents", "A_MEAN", "AIOE"]
    ).copy()

    fig = px.scatter(
        chart_data,
        x="ln_ai_patents",
        y="A_MEAN",
        color="AIOE",
        hover_name="OCC_TITLE",
        hover_data={
            "PRIM_STATE": True,
            "year": True,
            "OCC_CODE": True,
            "AI_patents": ":,.2f",
            "A_MEAN": ":$,.0f",
            "AIOE": ":.3f",
            "ln_ai_patents": False,
        },
        color_continuous_scale="Viridis",
        opacity=0.48,
        title="4. State AI Innovation and Occupational Mean Wages",
        labels={
            "ln_ai_patents": "ln(1 + fractional AI patents)",
            "A_MEAN": "Annual mean wage ($)",
            "AIOE": "AI exposure score",
        },
    )

    fig = add_trendline(
        fig,
        chart_data,
        "ln_ai_patents",
        "A_MEAN",
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title_x=0.02,
        height=570,
        yaxis=dict(
            tickprefix="$",
            tickformat=",",
            showgrid=True,
            zeroline=False,
        ),
        xaxis=dict(
            showgrid=True,
            zeroline=False,
        ),
        coloraxis_colorbar_title="AIOE",
        margin=dict(l=75, r=35, t=75, b=65),
    )

    return fig


# ============================================================
# 10. VISUAL 5 — EMPLOYMENT GROWTH SINCE 2020
# ============================================================

def make_employment_growth_chart(df, selected_state):
    """
    Calculate average employment by occupation exposure group and year.

    Then calculate percentage change relative to each group's 2020 average:

        Percent change = ((current average / 2020 average) - 1) * 100

    This chart shows relative changes, not raw employment levels.
    """

    # For this chart we need 2020 as the baseline, even if the user
    # selects another year in the shared year filter.
    trend_data = df.copy()

    if selected_state != "All states":
        trend_data = trend_data[
            trend_data["PRIM_STATE"] == selected_state
        ]

    trend_data = trend_data.dropna(
        subset=["TOT_EMP", "exposure_group"]
    )

    # Average employment for each exposure group and year.
    averages = (
        trend_data.groupby(
            ["year", "exposure_group"],
            observed=False,
            as_index=False,
        )
        .agg(mean_employment=("TOT_EMP", "mean"))
    )

    # Find each exposure group's 2020 baseline.
    baseline = (
        averages[averages["year"] == 2020]
        [["exposure_group", "mean_employment"]]
        .rename(columns={"mean_employment": "baseline_2020"})
    )

    averages = averages.merge(
        baseline,
        on="exposure_group",
        how="left",
    )

    averages["percent_change"] = (
        (averages["mean_employment"] / averages["baseline_2020"]) - 1
    ) * 100

    fig = px.line(
        averages,
        x="year",
        y="percent_change",
        color="exposure_group",
        markers=True,
        color_discrete_sequence=EXPOSURE_COLORS,
        hover_data={
            "mean_employment": ":,.0f",
            "baseline_2020": ":,.0f",
            "percent_change": ":.2f",
        },
        title="5. Employment Change Since 2020 by AI-Exposure Group",
        labels={
            "year": "Year",
            "percent_change": "Change in average employment since 2020 (%)",
            "exposure_group": "AI exposure group",
            "mean_employment": "Average employment",
            "baseline_2020": "2020 average employment",
        },
    )

    # A horizontal zero line makes increases and decreases easy to identify.
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#555555",
        annotation_text="2020 baseline",
        annotation_position="bottom right",
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title_x=0.02,
        height=520,
        xaxis=dict(dtick=1),
        margin=dict(l=65, r=35, t=75, b=65),
        legend_title_text="AI exposure",
    )

    return fig


# ============================================================
# 11. CREATE THE DASH APPLICATION
# ============================================================

def create_app(df):
    """Create the dashboard and connect shared filters to all charts."""

    app = Dash(__name__)
    app.title = "AI and the Labor Market"

    state_options = [
        {"label": "All states", "value": "All states"}
    ] + [
        {"label": state, "value": state}
        for state in sorted(df["PRIM_STATE"].dropna().unique())
    ]

    year_options = [
        {"label": "All years", "value": "All years"}
    ] + [
        {"label": str(year), "value": year}
        for year in YEARS
    ]

    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "maxWidth": "1500px",
            "margin": "0 auto",
            "padding": "25px",
            "backgroundColor": "#f7f9fc",
        },
        children=[
            html.H1(
                "AI Innovation and the U.S. Labor Market",
                style={"color": "#203354", "marginBottom": "5px"},
            ),
            html.P(
                "Exploring state AI patent activity, occupational AI exposure, "
                "employment, and wages from 2020 through 2023.",
                style={"color": "#526174"},
            ),

            # Shared controls: one selection updates the relevant charts.
            html.Div(
                style={
                    "display": "flex",
                    "gap": "25px",
                    "flexWrap": "wrap",
                    "backgroundColor": "white",
                    "padding": "18px",
                    "borderRadius": "10px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        style={"width": "240px"},
                        children=[
                            html.Label("Select a state"),
                            dcc.Dropdown(
                                id="state-filter",
                                options=state_options,
                                value="All states",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"width": "200px"},
                        children=[
                            html.Label("Select a year"),
                            dcc.Dropdown(
                                id="year-filter",
                                options=year_options,
                                value="All years",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "alignSelf": "end",
                            "color": "#526174",
                            "fontSize": "13px",
                        },
                        children=(
                            "The state and year selections update the map "
                            "and analytical charts together. The exposure "
                            "ranking remains occupation-level."
                        ),
                    ),
                ],
            ),

            html.Div(
                style={
                    "backgroundColor": "white",
                    "padding": "10px",
                    "borderRadius": "10px",
                    "marginBottom": "20px",
                },
                children=dcc.Graph(id="patent-map"),
            ),

            html.Div(
                style={
                    "backgroundColor": "white",
                    "padding": "10px",
                    "borderRadius": "10px",
                    "marginBottom": "20px",
                },
                children=dcc.Graph(
                    id="exposure-chart",
                    figure=make_occupation_exposure_chart(df),
                ),
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(450px, 1fr))",
                    "gap": "20px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        style={"backgroundColor": "white", "padding": "8px", "borderRadius": "10px"},
                        children=dcc.Graph(id="employment-scatter"),
                    ),
                    html.Div(
                        style={"backgroundColor": "white", "padding": "8px", "borderRadius": "10px"},
                        children=dcc.Graph(id="wage-scatter"),
                    ),
                ],
            ),

            html.Div(
                style={
                    "backgroundColor": "white",
                    "padding": "10px",
                    "borderRadius": "10px",
                },
                children=dcc.Graph(id="employment-growth"),
            ),

            html.Footer(
                "Sources: BLS Occupational Employment and Wage Statistics; "
                "AI patent dataset; Felten, Raj, and Seamans AIOE. "
                "Visual associations are descriptive and do not establish causation. "
                "BLS OEWS estimates have limitations for direct year-to-year comparisons.",
                style={
                    "fontSize": "12px",
                    "color": "#64748b",
                    "padding": "20px 5px",
                },
            ),
        ],
    )

    # One callback updates the map and all state/year-sensitive charts.
    @app.callback(
        Output("patent-map", "figure"),
        Output("employment-scatter", "figure"),
        Output("wage-scatter", "figure"),
        Output("employment-growth", "figure"),
        Input("state-filter", "value"),
        Input("year-filter", "value"),
    )
    def update_dashboard(selected_state, selected_year):
        filtered = filter_data(df, selected_state, selected_year)

        map_figure = make_ai_patent_map(df, selected_year)
        employment_figure = make_employment_scatter(filtered)
        wage_figure = make_wage_scatter(filtered)
        growth_figure = make_employment_growth_chart(df, selected_state)

        return (
            map_figure,
            employment_figure,
            wage_figure,
            growth_figure,
        )

    return app


# ============================================================
# 12. RUN THE DASHBOARD
# ============================================================

def main():
    """Load the data, create the app, and start the local server."""

    print("Loading AI labor-market dataset...")
    df = load_data()

    print("Creating dashboard...")
    app = create_app(df)

    print("\nDashboard is starting.")
    print("Open http://127.0.0.1:8050 in your browser.")
    print("Stop the dashboard by pressing CTRL+C in this terminal.\n")

    app.run(debug=False)


if __name__ == "__main__":
    main()
