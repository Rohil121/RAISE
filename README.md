# RAISE-India

RAISE-India is a Regime-Adaptive Indian Sector Equity strategy created for the Financial Markets Analytics project.

## Files

- `RAISE_India_Colab.ipynb` — one-click research notebook.
- `app.py` — Streamlit dashboard.
- `strategy_engine.py` — shared backtest and risk engine.
- `requirements.txt` — deployment dependencies.
- `RAISE_India_Report.docx` — explainable academic report.
- `RAISE_India_Report.pdf` — submission-ready PDF copy of the report.
- `RAISE_India_Presentation.pptx` — faculty presentation.

## Run in Google Colab

Upload and open `RAISE_India_Colab.ipynb`. Select **Runtime → Run all**. The notebook installs dependencies, downloads adjusted daily data, runs the walk-forward backtest, saves all CSV results, and creates a ZIP download.

## Deploy with Streamlit Community Cloud

1. Put `app.py`, `strategy_engine.py`, and `requirements.txt` in one GitHub repository.
2. Sign in to Streamlit Community Cloud and choose **Create app**.
3. Select the repository and set the main file to `app.py`.
4. Deploy and share the generated `streamlit.app` URL.

## Method summary

- 25 established NSE stocks across nine sectors.
- Weekly signals with a one-trading-day execution lag.
- Hybrid regime detector: expanding-window Gaussian Mixture Model plus transparent trend/volatility overrides.
- Regime-specific momentum, mean-reversion, and defensive signals.
- Inverse-volatility allocation with 15% stock caps, 25% sector caps, and a 12% portfolio volatility target.
- ₹10 lakh initial capital and 15 bps one-way implementation cost.
- Training: 2015–2019; validation: 2020–2022; untouched test: 2023 onward.

The offline demonstration mode exists only for code and user-interface testing. It must never be presented as real investment evidence.

## Final submission checklist

1. Replace the group-name placeholders in the report and cover slide.
2. Run the Colab notebook with **Runtime → Run all**.
3. Copy the Test-period metrics into the report and presentation result tables.
4. Upload `app.py`, `strategy_engine.py`, and `requirements.txt` to GitHub and deploy the app on Streamlit Community Cloud.
5. Verify the public dashboard link, then add it to the presentation.
