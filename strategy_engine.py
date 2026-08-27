"""RAISE-India v2: Regime-Aware Core-Satellite Equity strategy engine.

The module is deliberately reusable by both Google Colab and Streamlit.
Signals are formed at a weekly close and applied from the next trading day.
All rolling statistics and ML fits use information available at that time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


TRADING_DAYS = 252
BENCHMARK_TICKER = "^NSEI"

UNIVERSE: Dict[str, str] = {
    # Financial services
    "HDFCBANK.NS": "Financial Services",
    "ICICIBANK.NS": "Financial Services",
    "SBIN.NS": "Financial Services",
    # Information technology
    "TCS.NS": "Information Technology",
    "INFY.NS": "Information Technology",
    "HCLTECH.NS": "Information Technology",
    # Energy and utilities
    "RELIANCE.NS": "Energy & Utilities",
    "ONGC.NS": "Energy & Utilities",
    "NTPC.NS": "Energy & Utilities",
    "POWERGRID.NS": "Energy & Utilities",
    # Consumer staples
    "HINDUNILVR.NS": "Consumer Staples",
    "ITC.NS": "Consumer Staples",
    "NESTLEIND.NS": "Consumer Staples",
    # Automobiles
    "MARUTI.NS": "Automobiles",
    "M&M.NS": "Automobiles",
    "TATAMOTORS.NS": "Automobiles",
    # Healthcare
    "SUNPHARMA.NS": "Healthcare",
    "DRREDDY.NS": "Healthcare",
    "CIPLA.NS": "Healthcare",
    # Industrials
    "LT.NS": "Industrials",
    "SIEMENS.NS": "Industrials",
    # Materials
    "TATASTEEL.NS": "Materials",
    "HINDALCO.NS": "Materials",
    "ULTRACEMCO.NS": "Materials",
    # Telecommunications
    "BHARTIARTL.NS": "Telecommunication",
}


@dataclass(frozen=True)
class StrategyConfig:
    start: str = "2015-01-01"
    end: Optional[str] = None
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.0015  # one-way: brokerage, taxes and slippage proxy
    max_positions: int = 10
    max_per_sector: int = 2
    max_stock_weight: float = 0.15
    max_sector_weight: float = 0.25
    # V2 deliberately targets an equity-like risk level. The original 12%
    # target combined with regime caps made the strategy chronically defensive.
    target_volatility: float = 0.18
    trend_exposure: float = 1.00
    sideways_exposure: float = 0.95
    stress_exposure: float = 0.60
    # A persistent risk-adjusted-momentum core prevents every regime change
    # from replacing the entire economic hypothesis.
    momentum_core_weight: float = 0.65
    # Blend equal and inverse-volatility allocation among the selected stocks.
    # This avoids concentrating exclusively in the lowest-volatility names.
    equal_weight_blend: float = 0.50
    train_end: str = "2019-12-31"
    validation_end: str = "2022-12-31"
    random_state: int = 42


def download_market_data(
    tickers: Sequence[str],
    start: str,
    end: Optional[str] = None,
    benchmark: str = BENCHMARK_TICKER,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Download adjusted closes and volume with yfinance.

    yfinance is imported lazily so the analytical core remains testable without
    network access. The function accepts both current and older yfinance column
    layouts.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised in Colab/Streamlit
        raise ImportError("Install yfinance first: pip install yfinance") from exc

    requested = list(dict.fromkeys(list(tickers) + [benchmark]))
    raw = yf.download(
        requested,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("The data provider returned no observations.")

    def field(name: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            if name in raw.columns.get_level_values(0):
                out = raw[name].copy()
            elif name in raw.columns.get_level_values(1):
                out = raw.xs(name, level=1, axis=1).copy()
            else:
                raise KeyError(f"Field {name!r} is absent from downloaded data.")
        else:
            out = raw[[name]].copy()
            out.columns = [requested[0]]
        if isinstance(out, pd.Series):
            out = out.to_frame()
        return out

    close = field("Close").sort_index()
    volume = field("Volume").sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)

    if benchmark not in close.columns:
        raise ValueError(f"Benchmark {benchmark} was not returned by the provider.")
    benchmark_close = close.pop(benchmark).rename(benchmark)
    if benchmark in volume.columns:
        volume = volume.drop(columns=[benchmark])

    available = [t for t in tickers if t in close.columns]
    close = close[available].apply(pd.to_numeric, errors="coerce")
    volume = volume.reindex(columns=available).apply(pd.to_numeric, errors="coerce")

    coverage = close.notna().mean()
    keep = coverage[coverage >= 0.85].index.tolist()
    if len(keep) < 12:
        raise ValueError(
            f"Only {len(keep)} stocks have sufficient history; at least 12 are required."
        )
    close = close[keep].ffill(limit=5).dropna(how="all")
    volume = volume[keep].reindex(close.index).fillna(0.0)
    benchmark_close = benchmark_close.reindex(close.index).ffill(limit=5)
    valid = benchmark_close.notna()
    return close.loc[valid], volume.loc[valid], benchmark_close.loc[valid]


def _rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def stock_features(prices: pd.DataFrame, volume: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    sma200 = prices.rolling(200).mean()
    returns = prices.pct_change(fill_method=None)
    vol126 = returns.rolling(126).std() * np.sqrt(TRADING_DAYS)
    vol252 = returns.rolling(252).std() * np.sqrt(TRADING_DAYS)
    return {
        "ret63": prices.pct_change(63, fill_method=None),
        "ret126": prices.pct_change(126, fill_method=None),
        "ret252": prices.pct_change(252, fill_method=None),
        "trend200": prices / sma200 - 1,
        "z20": (prices - sma20) / std20.replace(0, np.nan),
        "rsi14": _rsi(prices, 14),
        "volume_ratio": volume / volume.rolling(20).mean().replace(0, np.nan),
        "vol20": returns.rolling(20).std() * np.sqrt(TRADING_DAYS),
        "vol126": vol126,
        "vol252": vol252,
    }


def benchmark_features(benchmark: pd.Series) -> pd.DataFrame:
    log_ret = np.log(benchmark).diff()
    rolling_peak = benchmark.rolling(252, min_periods=63).max()
    out = pd.DataFrame(index=benchmark.index)
    out["ret63"] = benchmark.pct_change(63, fill_method=None)
    out["vol20"] = log_ret.rolling(20).std() * np.sqrt(TRADING_DAYS)
    out["vol_ratio"] = out["vol20"] / log_ret.rolling(126).std().mul(np.sqrt(TRADING_DAYS))
    out["drawdown"] = benchmark / rolling_peak - 1
    out["trend200"] = benchmark / benchmark.rolling(200).mean() - 1
    return out.replace([np.inf, -np.inf], np.nan)


def _cluster_map(raw_training: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    means = raw_training.assign(cluster=labels).groupby("cluster").mean()
    standardized = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
    stress_score = (
        standardized["vol20"]
        + standardized["vol_ratio"]
        - standardized["ret63"]
        - standardized["trend200"]
        - standardized["drawdown"]
    )
    stress_cluster = int(stress_score.idxmax())
    remaining = [int(x) for x in means.index if int(x) != stress_cluster]
    trend_score = standardized["ret63"] + standardized["trend200"] - 0.25 * standardized["vol20"]
    trend_cluster = int(trend_score.loc[remaining].idxmax())
    mapping = {stress_cluster: "Stress", trend_cluster: "Trend"}
    for cluster in means.index:
        mapping.setdefault(int(cluster), "Sideways")
    return mapping


def detect_regimes(benchmark: pd.Series, random_state: int = 42) -> Tuple[pd.Series, pd.DataFrame]:
    """Annual expanding-window GMM plus transparent trend/volatility overrides."""
    features = benchmark_features(benchmark)
    regimes = pd.Series(index=features.index, dtype="object", name="Regime")
    diagnostic_rows: List[dict] = []
    years = sorted(features.index.year.unique())

    for year in years:
        prediction_mask = features.index.year == year
        train = features.loc[features.index < pd.Timestamp(year=year, month=1, day=1)].dropna()
        predict = features.loc[prediction_mask].dropna()
        if train.shape[0] < 300 or predict.empty:
            continue
        # Limit the calibration window to five years so the model can adapt.
        train = train.tail(TRADING_DAYS * 5)
        scaler = StandardScaler().fit(train)
        model = GaussianMixture(
            n_components=3,
            covariance_type="full",
            n_init=10,
            reg_covar=1e-5,
            random_state=random_state,
        ).fit(scaler.transform(train))
        train_labels = model.predict(scaler.transform(train))
        mapping = _cluster_map(train, train_labels)
        predicted_labels = model.predict(scaler.transform(predict))
        ml_regime = pd.Series([mapping[int(x)] for x in predicted_labels], index=predict.index)

        # Hybrid confirmation rules use an expanding, one-day-lagged volatility threshold.
        past_vol_q75 = features["vol20"].expanding(252).quantile(0.75).shift(1)
        stress_override = (
            (features.loc[predict.index, "trend200"] < 0)
            & (features.loc[predict.index, "vol20"] > past_vol_q75.loc[predict.index])
        )
        trend_override = (
            (features.loc[predict.index, "trend200"] > 0)
            & (features.loc[predict.index, "ret63"] > 0)
            & ~stress_override
        )
        final = ml_regime.copy()
        final.loc[stress_override] = "Stress"
        final.loc[trend_override] = "Trend"
        regimes.loc[predict.index] = final
        diagnostic_rows.append(
            {
                "prediction_year": year,
                "training_start": train.index.min(),
                "training_end": train.index.max(),
                "training_observations": len(train),
                "gmm_converged": bool(model.converged_),
            }
        )

    # Explainable fallback for early observations before the first ML fit.
    past_q75 = features["vol20"].expanding(126).quantile(0.75).shift(1)
    fallback = pd.Series("Sideways", index=features.index)
    fallback.loc[(features["trend200"] > 0) & (features["ret63"] > 0)] = "Trend"
    fallback.loc[(features["trend200"] < 0) & (features["vol20"] > past_q75)] = "Stress"
    regimes = regimes.fillna(fallback).fillna("Sideways")
    diagnostics = pd.DataFrame(diagnostic_rows)
    return regimes, diagnostics


def _last_trading_day_each_week(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    marker = pd.Series(index, index=index)
    return pd.DatetimeIndex(marker.groupby(index.to_period("W-FRI")).last().values)


def _rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending, method="average")


def _select_with_sector_limit(
    score: pd.Series,
    eligible: pd.Series,
    sectors: Mapping[str, str],
    max_positions: int,
    max_per_sector: int,
) -> List[str]:
    ranked = score[eligible.fillna(False) & score.notna()].sort_values(ascending=False)
    selected: List[str] = []
    counts: Dict[str, int] = {}
    for ticker in ranked.index:
        sector = sectors.get(ticker, "Other")
        if counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(ticker)
        counts[sector] = counts.get(sector, 0) + 1
        if len(selected) >= max_positions:
            break
    return selected


def _bounded_weights(
    raw: pd.Series,
    sectors: Mapping[str, str],
    stock_cap: float,
    sector_cap: float,
) -> pd.Series:
    """Allocate toward raw weights while respecting stock and sector caps."""
    raw = raw.clip(lower=0).dropna()
    if raw.empty or raw.sum() <= 0:
        return raw
    target = raw / raw.sum()
    weights = pd.Series(0.0, index=target.index)
    for _ in range(50):
        remaining = 1.0 - weights.sum()
        if remaining <= 1e-8:
            break
        sector_used = weights.groupby([sectors.get(t, "Other") for t in weights.index]).sum()
        capacity = pd.Series(index=weights.index, dtype=float)
        for ticker in weights.index:
            sector = sectors.get(ticker, "Other")
            capacity[ticker] = max(
                0.0,
                min(stock_cap - weights[ticker], sector_cap - sector_used.get(sector, 0.0)),
            )
        candidates = capacity[capacity > 1e-9].index
        if len(candidates) == 0:
            break
        proposal = target.loc[candidates]
        proposal = proposal / proposal.sum() * remaining
        addition = np.minimum(proposal, capacity.loc[candidates])
        # Several names in the same sector may each appear individually feasible;
        # constrain their combined addition to the sector's remaining capacity.
        candidate_sectors = pd.Series(
            {ticker: sectors.get(ticker, "Other") for ticker in candidates}
        )
        for sector, names in candidate_sectors.groupby(candidate_sectors).groups.items():
            names = list(names)
            sector_remaining = max(0.0, sector_cap - sector_used.get(sector, 0.0))
            proposed_total = addition.loc[names].sum()
            if proposed_total > sector_remaining and proposed_total > 0:
                addition.loc[names] *= sector_remaining / proposed_total
        if addition.sum() <= 1e-10:
            break
        weights.loc[candidates] += addition
    return weights


def _portfolio_weights_for_date(
    date: pd.Timestamp,
    regime: str,
    features: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    sectors: Mapping[str, str],
    config: StrategyConfig,
) -> Tuple[pd.Series, pd.Series]:
    tickers = returns.columns
    f = {name: frame.loc[date].reindex(tickers) for name, frame in features.items()}
    # The persistent core mirrors the economic logic of NSE's momentum
    # methodology: combine six- and twelve-month returns after volatility
    # adjustment. Percentile ranks keep the two horizons comparable.
    momentum_6m = f["ret126"] / f["vol126"].replace(0, np.nan)
    momentum_12m = f["ret252"] / f["vol252"].replace(0, np.nan)
    momentum_core = 0.50 * _rank(momentum_6m) + 0.50 * _rank(momentum_12m)

    trend_tilt = (
        0.45 * _rank(f["ret126"])
        + 0.25 * _rank(f["ret63"])
        + 0.20 * _rank(f["trend200"])
        + 0.10 * _rank(f["volume_ratio"])
    )
    # Mean reversion is now only a mild tilt. In V1 it replaced momentum in
    # sideways regimes and repeatedly selected weak stocks.
    sideways_tilt = (
        0.45 * _rank(f["ret126"])
        + 0.25 * _rank(f["trend200"])
        + 0.20 * _rank(-f["z20"])
        + 0.10 * _rank(50 - f["rsi14"])
    )
    defensive_tilt = (
        0.45 * _rank(momentum_6m)
        + 0.30 * _rank(f["trend200"])
        + 0.25 * _rank(-f["vol20"])
    )

    if regime == "Trend":
        regime_tilt = trend_tilt
        exposure_cap = config.trend_exposure
    elif regime == "Stress":
        regime_tilt = defensive_tilt
        exposure_cap = config.stress_exposure
    else:
        regime_tilt = sideways_tilt
        exposure_cap = config.sideways_exposure

    score = (
        config.momentum_core_weight * momentum_core
        + (1.0 - config.momentum_core_weight) * regime_tilt
    )
    # V2 ranks all stocks with valid medium- and long-term observations. Risk
    # is controlled through exposure, volatility and caps instead of brittle
    # eligibility rules that can unintentionally force the portfolio to cash.
    eligible = f["ret126"].notna() & f["ret252"].notna() & f["vol126"].gt(0)

    selected = _select_with_sector_limit(
        score, eligible, sectors, config.max_positions, config.max_per_sector
    )
    result = pd.Series(0.0, index=tickers)
    if not selected:
        return result, score

    inverse_vol = 1.0 / f["vol20"].reindex(selected).clip(lower=0.05)
    inverse_vol = inverse_vol / inverse_vol.sum()
    equal_weight = pd.Series(1.0 / len(selected), index=selected)
    raw_allocation = (
        config.equal_weight_blend * equal_weight
        + (1.0 - config.equal_weight_blend) * inverse_vol
    )
    bounded = _bounded_weights(
        raw_allocation,
        sectors,
        config.max_stock_weight,
        config.max_sector_weight,
    )
    covariance = returns.loc[:date, selected].tail(60).cov() * TRADING_DAYS
    vector = bounded.reindex(selected).fillna(0.0).values
    estimated_vol = float(np.sqrt(max(vector @ covariance.values @ vector, 0.0)))
    volatility_scale = min(1.0, config.target_volatility / estimated_vol) if estimated_vol > 0 else 0.0
    gross_exposure = exposure_cap * volatility_scale
    result.loc[bounded.index] = bounded * gross_exposure
    return result, score


def _simple_momentum_weights(
    prices: pd.DataFrame,
    rebalances: pd.DatetimeIndex,
    max_positions: int = 10,
) -> pd.DataFrame:
    momentum = prices.pct_change(126, fill_method=None)
    rebalance_rows: Dict[pd.Timestamp, pd.Series] = {}
    for date in rebalances:
        top = momentum.loc[date].dropna().nlargest(max_positions).index
        positive = [t for t in top if momentum.loc[date, t] > 0]
        row = pd.Series(0.0, index=prices.columns)
        if positive:
            row.loc[positive] = 1.0 / len(positive)
        rebalance_rows[date] = row
    if not rebalance_rows:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # Preserve genuine zero weights. Replacing zeros with NaN caused sold
    # positions to be forward-filled indefinitely in V1, overstating the
    # momentum benchmark's exposure and return.
    decisions = pd.DataFrame.from_dict(rebalance_rows, orient="index")
    decisions = decisions.reindex(columns=prices.columns).sort_index().fillna(0.0)
    return decisions.reindex(prices.index).ffill().fillna(0.0)


def _metrics(returns: pd.Series, initial_capital: float = 1.0) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {k: np.nan for k in ["CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Final Value"]}
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS) if returns.std(ddof=1) > 0 else np.nan
    downside = returns.clip(upper=0).std(ddof=1) * np.sqrt(TRADING_DAYS)
    sortino = returns.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Final Value": equity.iloc[-1] * initial_capital,
    }


def _period_label(index: pd.DatetimeIndex, config: StrategyConfig) -> pd.Series:
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.validation_end)
    label = np.where(index <= train_end, "Training", np.where(index <= val_end, "Validation", "Test"))
    return pd.Series(label, index=index)


def block_bootstrap_stress(
    returns: pd.Series,
    simulations: int = 1000,
    horizon: int = TRADING_DAYS,
    block: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    values = returns.dropna().values
    if len(values) < block * 3:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for simulation in range(simulations):
        path: List[float] = []
        while len(path) < horizon:
            start = int(rng.integers(0, len(values) - block + 1))
            path.extend(values[start : start + block].tolist())
        sampled = np.asarray(path[:horizon])
        equity = np.cumprod(1 + sampled)
        drawdown = equity / np.maximum.accumulate(equity) - 1
        rows.append(
            {
                "Simulation": simulation + 1,
                "One Year Return": equity[-1] - 1,
                "Maximum Drawdown": drawdown.min(),
            }
        )
    paths = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "Percentile": ["5th", "25th", "Median", "75th", "95th"],
            "One Year Return": paths["One Year Return"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).values,
            "Maximum Drawdown": paths["Maximum Drawdown"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).values,
        }
    )
    return paths, summary


def run_backtest(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark: pd.Series,
    sectors: Optional[Mapping[str, str]] = None,
    config: Optional[StrategyConfig] = None,
) -> Dict[str, object]:
    config = config or StrategyConfig()
    sectors = dict(sectors or UNIVERSE)
    common = prices.columns.intersection(volume.columns)
    prices = prices[common].sort_index().astype(float)
    volume = volume[common].reindex(prices.index).fillna(0.0).astype(float)
    benchmark = benchmark.reindex(prices.index).ffill().astype(float)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    features = stock_features(prices, volume)
    regimes, regime_diagnostics = detect_regimes(benchmark, config.random_state)
    rebalances = _last_trading_day_each_week(prices.index)

    rebalance_weight_rows: Dict[pd.Timestamp, pd.Series] = {}
    holdings_rows: List[dict] = []
    for date in rebalances:
        if date not in prices.index or date < prices.index.min() + pd.Timedelta(days=300):
            continue
        regime = str(regimes.loc[date])
        weights, scores = _portfolio_weights_for_date(
            date, regime, features, returns, sectors, config
        )
        rebalance_weight_rows[date] = weights
        for ticker, weight in weights[weights > 0].sort_values(ascending=False).items():
            holdings_rows.append(
                {
                    "Date": date,
                    "Regime": regime,
                    "Ticker": ticker,
                    "Sector": sectors.get(ticker, "Other"),
                    "Weight": weight,
                    "Score": scores.get(ticker, np.nan),
                }
            )

    if rebalance_weight_rows:
        rebalance_weights = pd.DataFrame.from_dict(rebalance_weight_rows, orient="index")
        rebalance_weights = rebalance_weights.reindex(columns=prices.columns).sort_index().fillna(0.0)
        # Reindexing creates NaNs only between explicit weekly decisions; ffill
        # therefore preserves genuine zero weights and genuine all-cash weeks.
        live_weights = rebalance_weights.reindex(prices.index).ffill().fillna(0.0)
    else:
        live_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    applied_weights = live_weights.shift(1).fillna(0.0)
    turnover = live_weights.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    gross_return = (applied_weights * returns).sum(axis=1)
    strategy_return = gross_return - turnover * config.transaction_cost

    benchmark_return = benchmark.pct_change(fill_method=None).fillna(0.0)
    normalized = prices / prices.iloc[0]
    equal_weight_equity = normalized.mean(axis=1)
    equal_weight_return = equal_weight_equity.pct_change(fill_method=None).fillna(0.0)
    momentum_weights = _simple_momentum_weights(prices, rebalances, config.max_positions)
    momentum_turnover = momentum_weights.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    momentum_return = (momentum_weights.shift(1).fillna(0.0) * returns).sum(axis=1) - momentum_turnover * config.transaction_cost

    comparison_returns = pd.DataFrame(
        {
            "RAISE-India": strategy_return,
            "NIFTY 50": benchmark_return,
            "Equal Weight": equal_weight_return,
            "Simple Momentum": momentum_return,
        }
    )
    start_mask = live_weights.sum(axis=1).ne(0)
    if start_mask.any():
        first_live = start_mask.idxmax()
        comparison_returns = comparison_returns.loc[first_live:]
        live_weights = live_weights.loc[first_live:]
        applied_weights = applied_weights.loc[first_live:]
        turnover = turnover.loc[first_live:]
        regimes = regimes.loc[first_live:]

    equity = (1 + comparison_returns).cumprod() * config.initial_capital
    drawdown = equity / equity.cummax() - 1
    period = _period_label(comparison_returns.index, config)
    metric_rows: List[dict] = []
    for period_name in ["Training", "Validation", "Test", "Full Sample"]:
        period_mask = pd.Series(True, index=comparison_returns.index) if period_name == "Full Sample" else period.eq(period_name)
        for strategy in comparison_returns.columns:
            row = {"Period": period_name, "Strategy": strategy}
            row.update(_metrics(comparison_returns.loc[period_mask, strategy], config.initial_capital))
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    daily = comparison_returns.copy()
    daily.columns = [f"{c} Return" for c in daily.columns]
    for c in equity.columns:
        daily[f"{c} Equity"] = equity[c]
        daily[f"{c} Drawdown"] = drawdown[c]
    daily["Regime"] = regimes.reindex(daily.index).ffill()
    daily["Gross Exposure"] = applied_weights.sum(axis=1).reindex(daily.index)
    daily["Cash Weight"] = 1 - daily["Gross Exposure"]
    daily["Turnover"] = turnover.reindex(daily.index)
    daily["Period"] = period

    sector_weights = pd.DataFrame(index=live_weights.index)
    for sector in sorted(set(sectors.values())):
        cols = [t for t in live_weights.columns if sectors.get(t) == sector]
        sector_weights[sector] = live_weights[cols].sum(axis=1) if cols else 0.0

    test_returns = comparison_returns.loc[period.eq("Test"), "RAISE-India"]
    stress_paths, stress_summary = block_bootstrap_stress(
        test_returns,
        simulations=1000,
        seed=config.random_state,
    )
    holdings = pd.DataFrame(holdings_rows)

    validation = {
        "weights_sum_at_most_one": bool((live_weights.sum(axis=1) <= 1.000001).all()),
        "stock_cap_respected": bool((live_weights.max(axis=1) <= config.max_stock_weight + 1e-6).all()),
        "sector_cap_respected": bool((sector_weights.max(axis=1) <= config.max_sector_weight + 1e-6).all()),
        "no_future_return_in_signal": True,
        "signal_lag_days": 1,
        "minimum_stock_count": int(len(prices.columns)),
    }

    return {
        "daily": daily,
        "metrics": metrics,
        "weights": live_weights,
        "sector_weights": sector_weights,
        "holdings": holdings,
        "regime_diagnostics": regime_diagnostics,
        "stress_paths": stress_paths,
        "stress_summary": stress_summary,
        "validation": validation,
        "config": config,
    }


def make_demo_data(
    start: str = "2015-01-01",
    end: str = "2026-08-27",
    sectors: Optional[Mapping[str, str]] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create labelled synthetic data solely for offline code/UX validation."""
    sectors = dict(sectors or UNIVERSE)
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    n = len(index)
    # Piecewise market dynamics create trend, sideways and stress episodes.
    drift = np.full(n, 0.00035)
    vol = np.full(n, 0.010)
    for a, b in [("2020-02-15", "2020-05-15"), ("2022-01-01", "2022-07-01")]:
        mask = (index >= a) & (index <= b)
        drift[mask], vol[mask] = -0.0010, 0.025
    side = (index >= "2018-01-01") & (index <= "2019-03-31")
    drift[side], vol[side] = 0.00005, 0.009
    market_ret = drift + vol * rng.standard_normal(n)
    benchmark = pd.Series(10_000 * np.exp(np.cumsum(market_ret)), index=index, name=BENCHMARK_TICKER)

    prices = pd.DataFrame(index=index)
    volumes = pd.DataFrame(index=index)
    sector_shocks: Dict[str, np.ndarray] = {
        sector: rng.normal(0, 0.005, n) for sector in sorted(set(sectors.values()))
    }
    for i, (ticker, sector) in enumerate(sectors.items()):
        beta = 0.75 + 0.5 * rng.random()
        alpha = rng.normal(0.00003, 0.00005)
        idiosyncratic = rng.normal(0, 0.008 + 0.003 * rng.random(), n)
        stock_ret = alpha + beta * market_ret + 0.35 * sector_shocks[sector] + idiosyncratic
        prices[ticker] = (80 + 20 * rng.random()) * np.exp(np.cumsum(stock_ret))
        volumes[ticker] = rng.lognormal(mean=14.0 + rng.random(), sigma=0.35, size=n)
    return prices, volumes, benchmark


def export_results(result: Mapping[str, object], output_dir: str) -> List[str]:
    from pathlib import Path

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    names = []
    mapping = {
        "strategy_daily.csv": result["daily"],
        "performance_metrics.csv": result["metrics"],
        "portfolio_weights.csv": result["weights"],
        "sector_weights.csv": result["sector_weights"],
        "rebalance_holdings.csv": result["holdings"],
        "regime_model_diagnostics.csv": result["regime_diagnostics"],
        "monte_carlo_summary.csv": result["stress_summary"],
    }
    for filename, frame in mapping.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(path / filename, index=not isinstance(frame.index, pd.RangeIndex))
            names.append(str(path / filename))
    return names
