"""Streamlit dashboard for RAISE-India v2."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from strategy_engine import (
    BENCHMARK_TICKER,
    UNIVERSE,
    StrategyConfig,
    download_market_data,
    make_demo_data,
    run_backtest,
)


st.set_page_config(page_title="RAISE-India", page_icon="📈", layout="wide")
st.title("RAISE-India v2")
st.caption("Regime-Aware Core-Satellite Indian Equity Strategy | Academic research dashboard")

with st.sidebar:
    st.header("Backtest settings")
    source = st.radio("Data source", ["Live market data", "Offline demonstration data"], index=0)
    start = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end = st.date_input("End date", value=pd.Timestamp.today().normalize())
    initial_capital = st.number_input("Starting capital (₹)", 100_000, 100_000_000, 1_000_000, 100_000)
    cost_bps = st.slider("One-way trading cost (bps)", 0, 50, 15)
    target_vol = st.slider("Target annual volatility", 5, 25, 18) / 100
    max_positions = st.slider("Maximum positions", 5, 15, 10)
    run = st.button("Run backtest", type="primary", use_container_width=True)
    st.info("Signals formed at the weekly close are applied one trading day later.")


@st.cache_data(show_spinner=False, ttl=3600)
def load_live(start_text: str, end_text: str):
    return download_market_data(list(UNIVERSE), start_text, end_text, BENCHMARK_TICKER)


@st.cache_data(show_spinner=False)
def load_demo(start_text: str, end_text: str):
    return make_demo_data(start_text, end_text)


if not run:
    st.markdown(
        """
        **How the strategy works**

        1. A Gaussian Mixture Model classifies the NIFTY environment using momentum, volatility, drawdown and trend.
        2. Transparent trend/volatility rules confirm or override ambiguous ML classifications.
        3. A persistent six- and twelve-month risk-adjusted-momentum core is retained in every regime.
        4. Regime-specific trend, mild mean-reversion and defensive tilts adjust rankings without replacing the core.
        5. Exposure remains equity-like at 100% / 95% / 60% for Trend / Sideways / Stress, before volatility scaling.
        6. Equal-plus-inverse-volatility weights, stock caps and sector caps control concentration.
        """
    )
    st.stop()

if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

config = StrategyConfig(
    start=str(start),
    end=str(end),
    initial_capital=float(initial_capital),
    transaction_cost=cost_bps / 10_000,
    target_volatility=target_vol,
    max_positions=max_positions,
)

try:
    with st.spinner("Downloading data and running the walk-forward backtest..."):
        if source == "Live market data":
            prices, volume, benchmark = load_live(str(start), str(end + pd.Timedelta(days=1)))
            data_label = "Live adjusted market data"
        else:
            prices, volume, benchmark = load_demo(str(start), str(end))
            data_label = "Synthetic demonstration data — not investment evidence"
        result = run_backtest(prices, volume, benchmark, UNIVERSE, config)
except Exception as exc:
    st.error(f"Backtest could not run: {exc}")
    if source == "Live market data":
        st.warning("Try again shortly or switch explicitly to Offline demonstration data to inspect the dashboard workflow.")
    st.stop()

daily = result["daily"]
metrics = result["metrics"]
test_metrics = metrics.query("Period == 'Test'").set_index("Strategy")
strategy_metrics = test_metrics.loc["RAISE-India"] if "RAISE-India" in test_metrics.index else metrics.iloc[0]
nifty_metrics = test_metrics.loc["NIFTY 50"] if "NIFTY 50" in test_metrics.index else None

st.success(data_label)
c1, c2, c3, c4 = st.columns(4)
cagr_delta = strategy_metrics["CAGR"] - nifty_metrics["CAGR"] if nifty_metrics is not None else None
sharpe_delta = strategy_metrics["Sharpe"] - nifty_metrics["Sharpe"] if nifty_metrics is not None else None
drawdown_delta = strategy_metrics["Max Drawdown"] - nifty_metrics["Max Drawdown"] if nifty_metrics is not None else None
c1.metric("Test CAGR", f"{strategy_metrics['CAGR']:.2%}", f"{cagr_delta:+.2%} vs NIFTY" if cagr_delta is not None else None)
c2.metric("Test Sharpe", f"{strategy_metrics['Sharpe']:.2f}", f"{sharpe_delta:+.2f} vs NIFTY" if sharpe_delta is not None else None)
c3.metric("Max drawdown", f"{strategy_metrics['Max Drawdown']:.2%}", f"{drawdown_delta:+.2%} vs NIFTY" if drawdown_delta is not None else None)
c4.metric("Ending value", f"₹{strategy_metrics['Final Value']:,.0f}")

if nifty_metrics is not None and strategy_metrics["CAGR"] > nifty_metrics["CAGR"]:
    st.success("RAISE-India v2 beat NIFTY 50 on test-period CAGR. Check Sharpe and drawdown before concluding that it dominated on risk-adjusted performance.")
elif nifty_metrics is not None:
    st.warning("RAISE-India v2 did not beat NIFTY 50 on test-period CAGR. Treat this as evidence, not a result to hide or tune away.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Performance", "Regimes", "Portfolio", "Stress test", "Downloads"]
)

with tab1:
    equity_cols = [c for c in daily.columns if c.endswith(" Equity")]
    equity_long = daily[equity_cols].rename(columns=lambda x: x.replace(" Equity", "")).rename_axis("Date").reset_index()
    equity_long = equity_long.melt("Date", var_name="Strategy", value_name="Portfolio Value")
    fig = px.line(equity_long, x="Date", y="Portfolio Value", color="Strategy", title="Growth of ₹10 lakh")
    fig.update_layout(legend_title_text="", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    dd_cols = [c for c in daily.columns if c.endswith(" Drawdown")]
    dd_long = daily[dd_cols].rename(columns=lambda x: x.replace(" Drawdown", "")).rename_axis("Date").reset_index()
    dd_long = dd_long.melt("Date", var_name="Strategy", value_name="Drawdown")
    fig_dd = px.line(dd_long, x="Date", y="Drawdown", color="Strategy", title="Drawdown paths")
    fig_dd.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)

    shown = metrics.copy()
    for col in ["CAGR", "Volatility", "Max Drawdown"]:
        shown[col] = shown[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    for col in ["Sharpe", "Sortino", "Calmar"]:
        shown[col] = shown[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    shown["Final Value"] = shown["Final Value"].map(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    st.dataframe(shown, use_container_width=True, hide_index=True)

with tab2:
    regime_map = {"Trend": 1, "Sideways": 0, "Stress": -1}
    regime_frame = daily[["Regime", "Gross Exposure"]].copy()
    regime_frame["Regime Code"] = regime_frame["Regime"].map(regime_map)
    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(x=regime_frame.index, y=regime_frame["Regime Code"], mode="lines", name="Regime"))
    fig_regime.add_trace(go.Scatter(x=regime_frame.index, y=regime_frame["Gross Exposure"], mode="lines", name="Gross exposure", yaxis="y2"))
    fig_regime.update_layout(
        title="Detected regime and risk exposure",
        yaxis=dict(tickvals=[-1, 0, 1], ticktext=["Stress", "Sideways", "Trend"]),
        yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]),
        hovermode="x unified",
    )
    st.plotly_chart(fig_regime, use_container_width=True)
    counts = daily["Regime"].value_counts(normalize=True).rename("Share").rename_axis("Regime").reset_index()
    st.plotly_chart(px.bar(counts, x="Regime", y="Share", color="Regime", title="Time spent in each regime"), use_container_width=True)

with tab3:
    weights = result["weights"]
    last = weights.iloc[-1]
    current = last[last > 0].sort_values(ascending=False).rename("Weight").rename_axis("Ticker").reset_index()
    current["Sector"] = current["Ticker"].map(UNIVERSE)
    current["Weight"] = current["Weight"].map(lambda x: f"{x:.2%}")
    st.subheader("Latest portfolio")
    st.dataframe(current, hide_index=True, use_container_width=True)
    sector = result["sector_weights"].iloc[-1].sort_values(ascending=False)
    sector = sector[sector > 0].rename("Weight").rename_axis("Sector").reset_index()
    st.plotly_chart(px.bar(sector, x="Sector", y="Weight", title="Latest sector allocation"), use_container_width=True)
    st.caption(f"Cash weight: {daily['Cash Weight'].iloc[-1]:.2%}")

with tab4:
    stress = result["stress_summary"]
    if stress.empty:
        st.warning("The test period is too short for block-bootstrap stress testing.")
    else:
        shown = stress.copy()
        shown["One Year Return"] = shown["One Year Return"].map(lambda x: f"{x:.2%}")
        shown["Maximum Drawdown"] = shown["Maximum Drawdown"].map(lambda x: f"{x:.2%}")
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.caption("1,000 one-year paths generated by resampling 20-day blocks from the untouched test-period returns.")

with tab5:
    def csv_bytes(frame: pd.DataFrame, index: bool = True) -> bytes:
        return frame.to_csv(index=index).encode("utf-8")

    downloadable = {
        "strategy_daily.csv": result["daily"],
        "performance_metrics.csv": result["metrics"],
        "portfolio_weights.csv": result["weights"],
        "sector_weights.csv": result["sector_weights"],
        "rebalance_holdings.csv": result["holdings"],
        "monte_carlo_summary.csv": result["stress_summary"],
        "market_prices.csv": prices,
        "market_volume.csv": volume,
        "benchmark_prices.csv": benchmark.rename("NIFTY 50").to_frame(),
    }
    for name, frame in downloadable.items():
        st.download_button(name, csv_bytes(frame, index=not isinstance(frame.index, pd.RangeIndex)), name, "text/csv")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in downloadable.items():
            zf.writestr(name, csv_bytes(frame, index=not isinstance(frame.index, pd.RangeIndex)))
    st.download_button("Download all results (.zip)", archive.getvalue(), "RAISE_India_results.zip", "application/zip", type="primary")

with st.expander("Audit checks"):
    st.json(result["validation"])
    st.caption("Research use only. Backtests are hypothetical and do not guarantee future performance.")
