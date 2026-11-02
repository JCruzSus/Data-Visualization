# AI Innovation and the U.S. Labor Market

An interactive data visualization project exploring how state-level AI innovation and occupational AI exposure are associated with employment and wages across U.S. states from 2020–2023.

## Research question

How is state-level AI innovation associated with employment and wages in occupations with different levels of exposure to artificial intelligence?

## Data sources

- **BLS Occupational Employment and Wage Statistics (OEWS):** state-by-occupation employment and wage estimates.
- **AI patent data:** state-year AI patent counts, including fractional and unique patent counts.
- **AIOE:** occupation-level AI exposure scores from Felten, Raj, and Seamans.

The merged analysis dataset covers 2020–2023 and retains occupations with a published AIOE score.

## View the dashboard online

The public preview is in [docs/index.html](docs/index.html). It contains five interactive Plotly visualizations with state and year filters.

To publish it with GitHub Pages:

1. Open this repository on GitHub.
2. Select **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select branch **main** and folder **/docs**, then click **Save**.
5. After GitHub Pages finishes deploying, the site will be available at:

   https://jcruzsus.github.io/Data-Visualization/

The static dashboard preview uses state-year averages for the scatterplots so it loads quickly in a browser. It is intended for exploration and presentation; it is not the full occupation-level dataset.

## Run the full interactive Dash application locally

The full Dash application uses the merged occupation-level CSV.

1. Clone or download this repository.
2. Put the full merged file named `AI_BLS_AIOE_2020_2023_final.csv` inside the `data/` folder. See [data/README.md](data/README.md) for details.
3. Install the packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run:

   ```bash
   python app.py
   ```

5. Open http://127.0.0.1:8050 in your browser.

## Repository layout

- `app.py` — organized Dash app with five separate chart functions and linked state/year filters.
- `docs/index.html` — lightweight browser dashboard for GitHub Pages.
- `results/model_comparison.csv` — model evaluation results.
- `documentation/CODE_WALKTHROUGH.md` — explanation of the source-code structure.
- `requirements.txt` — Python dependencies.
- `data/` — location for the full merged CSV used by the local Dash app.

## Interpretation and limitations

- AIOE measures occupational exposure to AI-related capabilities; it is not a probability of job loss.
- Scatterplot trendlines are descriptive and do not establish causation.
- Associations between patent activity and employment or wages may reflect state size, industry composition, education, or other confounding factors.
- BLS OEWS estimates have limitations for direct year-to-year comparisons. Interpret employment trends cautiously.
- Predictive model performance describes out-of-sample prediction; it is not evidence of causal effects.
