# AI Innovation and the U.S. Labor Market

An interactive data visualization project exploring the relationship between state-level AI innovation, occupational AI exposure, employment, and wages across U.S. states from 2020–2023.

## Research question

How is state-level AI innovation associated with employment and wages in occupations with different levels of exposure to artificial intelligence?

## Data sources

- **BLS Occupational Employment and Wage Statistics (OEWS):** state-by-occupation employment and wage estimates.
- **AI patent data:** state-year AI patent counts, including fractional counts and unique patent counts.
- **AIOE:** occupational AI exposure scores from Felten, Raj, and Seamans.

The merged analysis file contains state–occupation–year observations for 2020–2023. The analysis is restricted to occupations with a published AIOE score.

## Visualizations

1. AI patent activity by state.
2. Occupational AI exposure.
3. State AI innovation and occupational employment.
4. State AI innovation and occupational mean wages.
5. Employment change by AI-exposure group, relative to 2020.

The HTML dashboard is an interactive, static Plotly report. The Dash application provides shared state/year filters and linked updates when run locally.

## Run the interactive Dash application

1. Download or clone this repository.
2. Put `AI_BLS_AIOE_2020_2023_final.csv` in the project root (or update `DATA_FILE` in the script).
3. Install dependencies:

   ```bash
   python -m pip install -r requirements_ai_labor_project_dash.txt
   ```

4. Start the app:

   ```bash
   python ai_labor_dashboard_app.py
   ```

5. Open http://127.0.0.1:8050 in your browser.

## Interpretation and limitations

- AIOE measures occupational exposure to AI-related capabilities; it is not a probability of job loss.
- The scatterplot trendlines are descriptive and do not establish causation.
- Associations between state patent activity and employment or wages may reflect state size, industry composition, education, and other confounding factors.
- BLS OEWS estimates have limitations for direct year-to-year comparisons. Interpret employment trends cautiously.
- Predictive model performance should be evaluated on held-out observations and should not be interpreted as causal evidence.

## Repository contents

The source code, merged data, HTML dashboard, requirements, and model evaluation output should be included in this repository. See the folders and filenames as they are added.
