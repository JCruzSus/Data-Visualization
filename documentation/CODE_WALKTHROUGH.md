# Project code walkthrough

- app.py: Dash application entry point.
- load_data(): loads the merged CSV and creates the AI patent log variable and exposure groups.
- filter_rows(): applies shared state and year filters.
- add_descriptive_trendline(): adds an OLS descriptive trendline to scatterplots.
- chart_patent_map(): Visual 1, state-level AI patent map.
- chart_occupation_exposure(): Visual 2, highest and lowest AIOE occupations.
- chart_employment_scatter(): Visual 3, AI patents and log employment.
- chart_wage_scatter(): Visual 4, AI patents and annual mean wages.
- chart_employment_growth(): Visual 5, percentage change from the 2020 baseline.
- create_app(): builds the Dash layout and links the filters to the charts.
- main(): loads the data and starts the local server.

## Run locally

Install dependencies:
python -m pip install -r requirements.txt

Start the app:
python app.py

Open:
http://127.0.0.1:8050

Note: the local Dash application requires Python. A static HTML dashboard is separate and can be viewed in a browser without running Dash.
