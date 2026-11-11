# Streamlit Community Cloud deployment

The app is committed at `app.py` and is configured to download the official UCI Online Retail archive when `data/raw/Online Retail.xlsx` is missing. The data are downloaded to the hosting environment at first launch; the raw workbook is not committed.

## Deployment steps (requires the repository owner's Streamlit account)

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Select **Create app** → **Deploy a public app from GitHub**.
3. Repository: `AlDSChriSOB/UK-Retail-Intelligence-Apriori`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Select **Deploy** and wait for the dataset download and initial model preparation.

The app uses `requirements.txt` and should not need a separate secrets file. Public hosting can have memory, network, package compatibility or startup-time constraints; inspect deployment logs if it fails.

**Deployment is not yet verified.** Committing a Streamlit app does not create a public URL or authorise the assistant to sign in on the repository owner's behalf. Do not add an invented Streamlit URL to the README.

## Research safeguards

- The first 70% of invoices (ordered by earliest timestamp) fit popularity, co-occurrence and two-item association-rule models.
- The later 30% are excluded from model fitting.
- This demo is historical **basket completion**, not personalised future purchasing prediction.
- Avoid committing the raw workbook or personal customer information.

## CI

The `Python tests` GitHub Actions workflow checks Python syntax, notebook schema and pytest. Notebook schema validation does **not** execute the notebook or verify its figures.
