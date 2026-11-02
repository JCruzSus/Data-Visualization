# Full analysis dataset

The local Dash application expects this file at:

`data/AI_BLS_AIOE_2020_2023_final.csv`

The merged CSV contains approximately 127,575 state–occupation–year observations for 2020–2023 and is about 20 MB. It is not included in this initial GitHub upload because the GitHub connector upload supports text content and the dataset is a large file.

To make the full Python application runnable:

1. Download the merged CSV from the project files provided by the project owner.
2. Open this repository on GitHub and select **Add file → Upload files**.
3. Upload the CSV into this `data/` directory.
4. Commit the upload to the `main` branch.

The public `docs/index.html` dashboard already includes a compact state-year summary for browser viewing. The full CSV is needed for the detailed occupation-level Dash application.
