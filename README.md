# Yachida 2019 CRC Metabolomics + Metagenomics Dashboard

Streamlit dashboard for exploring the simplified Yachida 2019 colorectal cancer metabolomics and metagenomics dataset.

## Features

- Metabolite abundance explorer with ordered disease stages and tumor-side stratification
- Spearman correlations in both directions:
  - selected metabolite vs all bacteria
  - selected bacteria vs all metabolites
- Location bubble plot for invasive CRC, ordered as right side, left side, rectum
- Bacteria co-occurrence heatmap
- CRC enrichment explorer for bacteria abundance across Healthy, StageI_II, and Stage_III_IV

The app normalizes common stage/location labels, excludes `Stage_HS`, and ignores multi-site tumor locations in side-stratified views.

## Local Setup

```bash
cd yachida_crc_dashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app expects `data/yachida_simplified.xlsx` by default. You can also upload another compatible Excel file from the sidebar.

The bundled workbook is cached locally under `data/.cache/` after the first run to make later launches faster.

## Deploy On Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to <https://share.streamlit.io/>.
3. Create a new app from the repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

Streamlit Community Cloud should use the root `requirements.txt` file for Python dependencies.

No secrets are required for the current version.

## Data

The default dataset is stored at:

```text
data/yachida_simplified.xlsx
```

Before making the repository public, confirm that this simplified dataset can be redistributed publicly.
