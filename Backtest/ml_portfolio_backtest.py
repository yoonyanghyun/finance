"""Walk-forward comparison of 60/40, ML, deep learning, and RL portfolios.

Install:
    pip install yfinance pandas numpy matplotlib scikit-learn

Example:
    python ml_portfolio_backtest.py --start 2005-01-01 --transaction-cost-bps 5

The signal for a month uses only prices available at the preceding month-end.
Models are re-fit with an expanding window; this makes the reported test period
out-of-sample rather than an in-sample fit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRADING_DAYS = 252

# FRED public series.  Rates/spreads describe the financing and risk regime;
# CPI, unemployment, and industrial production describe the real economy.
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "yield_curve_10y_2y": "T10Y2Y",
    "treasury_10y": "DGS10",
    "high_yield_oas": "BAMLH0A0HYM2",
    "unemployment_rate": "UNRATE",
    "industrial_production": "INDPRO",
    "cpi": "CPIAUCSL",
}


def load_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("가격 데이터를 받지 못했습니다. 티커와 인터넷 연결을 확인하세요.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna().rename_axis("Date")


def load_macro_features(month_end_index: pd.DatetimeIndex, lag_months: int = 1) -> pd.DataFrame:
    """Download FRED macro data and return point-in-time-safe monthly features.

    FRED observations are not necessarily known on their observation date.  A
    one-month lag is deliberately imposed on *every* macro input, which is a
    conservative approximation of release timing and avoids a common source
    of macro backtest look-ahead bias.
    """
    start = (month_end_index.min() - pd.DateOffset(months=15)).strftime("%Y-%m-%d")
    end = month_end_index.max().strftime("%Y-%m-%d")
    raw_series: dict[str, pd.Series] = {}
    failures: list[str] = []
    for name, series_id in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            csv = pd.read_csv(url)
            csv["observation_date"] = pd.to_datetime(csv["observation_date"])
            values = pd.to_numeric(csv[series_id], errors="coerce")
            raw_series[name] = pd.Series(values.to_numpy(), index=csv["observation_date"], name=name).loc[start:end]
        except Exception as error:  # Network outages should name the missing series.
            failures.append(f"{series_id}: {error}")
    if failures:
        raise RuntimeError("FRED macro data download failed. " + " | ".join(failures))

    levels = pd.DataFrame(raw_series).sort_index().resample("ME").last().ffill()
    macro = pd.DataFrame(index=levels.index)
    macro["fed_funds_rate"] = levels["fed_funds_rate"]
    macro["fed_funds_change_3m"] = levels["fed_funds_rate"].diff(3)
    macro["yield_curve_10y_2y"] = levels["yield_curve_10y_2y"]
    macro["yield_curve_change_3m"] = levels["yield_curve_10y_2y"].diff(3)
    macro["treasury_10y"] = levels["treasury_10y"]
    macro["high_yield_oas"] = levels["high_yield_oas"]
    macro["hy_oas_change_3m"] = levels["high_yield_oas"].diff(3)
    macro["unemployment_rate"] = levels["unemployment_rate"]
    macro["unemployment_change_3m"] = levels["unemployment_rate"].diff(3)
    macro["industrial_production_yoy"] = levels["industrial_production"].pct_change(12)
    macro["cpi_yoy"] = levels["cpi"].pct_change(12)
    # Use only information that would have been published before the trade.
    macro = macro.shift(lag_months).replace([np.inf, -np.inf], np.nan)
    # Keep only series that actually contain values over the requested period.
    # This protects the backtest if FRED retires or temporarily withholds one
    # particular series; valid series remain usable.
    macro = macro.reindex(month_end_index).ffill()
    macro = macro.dropna(axis=1, how="all")
    return macro.rename_axis("Date")


def make_monthly_features(prices: pd.DataFrame, equity: str, bond: str,
                          macro_features: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return month-end features and next month's equity-minus-bond return."""
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change()
    excess = returns[equity] - returns[bond]
    features = pd.DataFrame(index=monthly.index)
    features["eq_mom_1"] = returns[equity]
    features["eq_mom_3"] = monthly[equity].pct_change(3)
    features["eq_mom_6"] = monthly[equity].pct_change(6)
    features["eq_mom_12"] = monthly[equity].pct_change(12)
    features["bond_mom_3"] = monthly[bond].pct_change(3)
    features["bond_mom_12"] = monthly[bond].pct_change(12)
    features["eq_vol_3m"] = returns[equity].rolling(3).std()
    features["bond_vol_3m"] = returns[bond].rolling(3).std()
    features["corr_3m"] = returns[equity].rolling(3).corr(returns[bond])
    features["eq_drawdown_12m"] = monthly[equity] / monthly[equity].rolling(12).max() - 1
    if macro_features is not None:
        features = features.join(macro_features.reindex(features.index))
    # Label is intentionally shifted: features at t predict t+1.
    return features, excess.shift(-1).rename("next_month_excess_return")


def clip_weight(weight: float, minimum: float, maximum: float) -> float:
    return float(np.clip(weight, minimum, maximum))


def ridge_weight(x: pd.DataFrame, y: pd.Series, current: pd.Series, minimum: float, maximum: float) -> float:
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x, y)
    predicted_excess = float(model.predict(current.to_frame().T)[0])
    # The scale makes a one-month forecast economically meaningful but bounded.
    return clip_weight(0.60 + predicted_excess / 0.04 * 0.20, minimum, maximum)


def deep_learning_weight(x: pd.DataFrame, y: pd.Series, current: pd.Series, minimum: float, maximum: float) -> float:
    # Small network and fixed seed: suitable for the relatively small monthly sample.
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(16, 8), alpha=0.02, early_stopping=True,
                     validation_fraction=0.2, max_iter=500, random_state=7),
    )
    model.fit(x, y)
    predicted_excess = float(model.predict(current.to_frame().T)[0])
    return clip_weight(0.60 + predicted_excess / 0.04 * 0.20, minimum, maximum)


def rl_weight(x: pd.DataFrame, y: pd.Series, current: pd.Series, minimum: float, maximum: float) -> float:
    """Full-information contextual bandit with linear Q-functions.

    Actions are equity weights.  Since historical returns of both ETFs are
    known, each action's counterfactual one-month reward is observable.  It is
    a compact, reproducible reinforcement-learning baseline without an
    external RL package; use an environment with execution/slippage models for
    a production RL implementation.
    """
    actions = np.linspace(minimum, maximum, 11)
    scaler = StandardScaler().fit(x)
    train_x = scaler.transform(x)
    state = np.column_stack([np.ones(len(train_x)), train_x])
    now = np.r_[1.0, scaler.transform(current.to_frame().T)[0]]
    # Reward: portfolio return for every possible allocation, minus a modest
    # turnover proxy from the neutral 60/40 allocation.
    q_values = []
    for action in actions:
        reward = action * y.to_numpy() - 0.001 * abs(action - 0.60)
        beta = np.linalg.solve(state.T @ state + 1.0 * np.eye(state.shape[1]), state.T @ reward)
        q_values.append(float(now @ beta))
    return float(actions[int(np.argmax(q_values))])


@dataclass
class Strategy:
    name: str
    weights: pd.Series


def walk_forward_weights(features: pd.DataFrame, label: pd.Series, warmup_months: int,
                         minimum: float, maximum: float) -> list[Strategy]:
    valid = features.dropna().index.intersection(label.dropna().index)
    ml, dl, rl = (pd.Series(0.60, index=features.index, dtype=float) for _ in range(3))
    # At t, a label from t is unknown, so the fitting set ends at t-1.
    for position, date in enumerate(valid):
        historical = valid[:position]
        if len(historical) < warmup_months:
            continue
        x, y = features.loc[historical], label.loc[historical]
        current = features.loc[date]
        ml.loc[date] = ridge_weight(x, y, current, minimum, maximum)
        dl.loc[date] = deep_learning_weight(x, y, current, minimum, maximum)
        rl.loc[date] = rl_weight(x, y, current, minimum, maximum)
    # A signal dated at the prior month-end is traded on the first business day
    # of the next month, avoiding any same-day information use.
    return [
        Strategy("60/40", pd.Series(0.60, index=features.index)),
        Strategy("ML Ridge", ml), Strategy("DL MLP", dl), Strategy("RL Bandit", rl),
    ]


def daily_backtest(prices: pd.DataFrame, equity: str, bond: str, month_end_weights: pd.Series,
                   transaction_cost_bps: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Monthly rebalanced backtest, matching the user's original 60/40 timing.

    Returns on date t use the *previous* day's drifted allocation.  A target
    selected at the first trading day of a month is recorded after that day's
    close and applies to t+1.  Therefore neither the signal nor the rebalance
    can use that day's return.
    """
    returns = prices.pct_change().fillna(0.0)
    targets = month_end_weights.reindex(prices.resample("ME").last().index).ffill().fillna(0.60)
    target_by_day = targets.copy()
    # A target learned at January month-end is first executable in February.
    target_by_day.index = (target_by_day.index.to_period("M") + 1).to_timestamp(how="start")
    target_by_day = target_by_day[~target_by_day.index.duplicated(keep="last")]
    weights, portfolio_returns = [], []
    # Keep unscaled holdings as well as the normalized allocation.  This lets
    # weights drift naturally between monthly rebalances, just like the
    # monthly_rebalanced_portfolio function in the original script.
    previous_scaled_weight = pd.Series({equity: 0.60, bond: 0.40}, dtype=float)
    previous_rebalanced_weight = previous_scaled_weight.copy()
    pending_cost = 0.0
    cost_rate = transaction_cost_bps / 10_000
    for i, (date, asset_return) in enumerate(returns.iterrows()):
        if i == 0:
            scaled_weight, net = previous_scaled_weight.copy(), 0.0
        else:
            gross = float((previous_scaled_weight * asset_return).sum())
            net = (1 + gross) * (1 - pending_cost) - 1
            pending_cost = 0.0
            # End-of-day holdings after market movement, before any rebalance.
            portfolio_weight = previous_rebalanced_weight * (1 + asset_return)
            if date in target_by_day.index:
                equity_weight = float(target_by_day.loc[date])
                target_weight = pd.Series({equity: equity_weight, bond: 1 - equity_weight})
                # Exact convention from the existing script: reset after the
                # day's calculation; target becomes effective on the next row.
                rebalanced_weight = target_weight * previous_rebalanced_weight.sum()
                turnover = 0.5 * float((target_weight - previous_scaled_weight).abs().sum())
                pending_cost = turnover * cost_rate
                scaled_weight = target_weight
            else:
                rebalanced_weight = portfolio_weight
                scaled_weight = rebalanced_weight / rebalanced_weight.sum()
        weights.append(float(scaled_weight[equity]))
        portfolio_returns.append(net)
        previous_rebalanced_weight = rebalanced_weight if i > 0 else previous_rebalanced_weight
        previous_scaled_weight = scaled_weight
    daily_returns = pd.Series(portfolio_returns, index=prices.index, name="Daily Return")
    return (1 + daily_returns).cumprod().rename("NAV"), daily_returns, pd.Series(weights, index=prices.index, name="Equity Weight")


def performance_summary(nav: pd.Series, daily_returns: pd.Series) -> pd.Series:
    years = (len(daily_returns) - 1) / TRADING_DAYS
    if years <= 0:
        raise ValueError("성과 계산에는 최소 두 거래일이 필요합니다.")
    drawdown = nav / nav.cummax() - 1
    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    downside = np.sqrt(np.mean(np.minimum(daily_returns.to_numpy(), 0.0) ** 2))
    underwater = drawdown < 0
    longest_drawdown = int(underwater.groupby((~underwater).cumsum()).sum().max()) if underwater.any() else 0
    return pd.Series({
        "Total Return": nav.iloc[-1] / nav.iloc[0] - 1, "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe (rf=0%)": daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS),
        "Maximum Drawdown": drawdown.min(),
        "Calmar": cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
        "Sortino (target=0%)": (daily_returns.mean() * TRADING_DAYS) / (downside * np.sqrt(TRADING_DAYS)),
        "Ulcer Index": np.sqrt(np.mean(drawdown.to_numpy() ** 2)),
        "Longest Drawdown (days)": longest_drawdown,
        "Final NAV": nav.iloc[-1],
    })


def main() -> None:
    # Spyder Variable Explorer: these names remain available after Run (F5).
    global prices_df, monthly_prices, macro_df, feature_df, target_df, strategies
    global strategy_signal_weights, strategy_nav, strategy_daily_returns
    global strategy_daily_equity_weights, strategy_monthly_equity_weights
    global strategy_statistics, strategy_statistics_full, strategy_statistics_oos
    global oos_start_date, comparison_figure
    parser = argparse.ArgumentParser(description="Walk-forward ML/DL/RL portfolio comparison")
    parser.add_argument("--start", default="2002-07-30")
    parser.add_argument("--end", default=None)
    parser.add_argument("--equity", default="SPY")
    parser.add_argument("--bond", default="IEF")
    parser.add_argument("--warmup-months", type=int, default=60, help="expanding-window training minimum")
    parser.add_argument("--min-equity", type=float, default=0.20)
    parser.add_argument("--max-equity", type=float, default=0.80)
    parser.add_argument("--macro-lag-months", type=int, default=1,
                        help="FRED macro observation lag; 1 prevents release-date look-ahead")
    parser.add_argument("--no-macro", action="store_true", help="use price features only")
    # Default matches the existing 60/40 script.  Set (e.g.) 5 explicitly for
    # a more realistic comparison that charges all strategies identically.
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--plot-file", default="ml_portfolio_comparison.png")
    args = parser.parse_args()
    if not 0 <= args.min_equity <= args.max_equity <= 1:
        raise ValueError("Equity weight bounds must satisfy 0 <= min <= max <= 1.")
    prices_df = load_prices([args.equity, args.bond], args.start, args.end)
    monthly_prices = prices_df.resample("ME").last()
    macro_df = None if args.no_macro else load_macro_features(monthly_prices.index, args.macro_lag_months)
    feature_df, target_df = make_monthly_features(prices_df, args.equity, args.bond, macro_df)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill()
    # A discontinued FRED series must not erase every training row. It is
    # removed and reported, while the other economic inputs remain in use.
    # A feature that exists only for a short recent subperiod (for example a
    # revised/retired FRED spread series) would make every earlier row invalid.
    # Retain only inputs available for at least 80% of the backtest history.
    minimum_coverage = int(len(feature_df) * 0.80)
    unavailable_features = feature_df.columns[feature_df.notna().sum() < minimum_coverage].tolist()
    if unavailable_features:
        print("Excluded low-coverage features:", ", ".join(unavailable_features))
        feature_df = feature_df.drop(columns=unavailable_features)
    # Friendly aliases for notebook / Spyder inspection.
    prices = prices_df
    features, label = feature_df, target_df
    macro_columns_used = [column for column in features.columns if column in (macro_df.columns if macro_df is not None else [])]
    print(f"Features: {features.shape[1]} ({len(macro_columns_used)} FRED macro features)")
    strategies = walk_forward_weights(features, label, args.warmup_months, args.min_equity, args.max_equity)
    strategy_signal_weights = pd.DataFrame({strategy.name: strategy.weights for strategy in strategies})
    valid_dates = features.dropna().index.intersection(label.dropna().index)
    if False and len(valid_dates) <= args.warmup_months:
        raise ValueError("기간이 너무 짧습니다. warmup-months를 줄이거나 시작일을 앞당기세요.")
    if len(valid_dates) <= args.warmup_months:
        availability = features.notna().sum().sort_values().to_dict()
        raise ValueError(
            "Insufficient valid monthly observations. Reduce --warmup-months, "
            "use an earlier start date, or inspect feature availability: " + str(availability)
        )
    first_signal_date = valid_dates[args.warmup_months]
    first_trade_month = (first_signal_date.to_period("M") + 1).to_timestamp(how="start")
    first_oos_date = prices.index[prices.index >= first_trade_month][0]
    oos_start_date = first_oos_date
    navs, full_summaries, oos_summaries, allocations, daily_returns_by_strategy = {}, {}, {}, {}, {}
    for strategy in strategies:
        nav, daily_return, allocation = daily_backtest(prices, args.equity, args.bond, strategy.weights, args.transaction_cost_bps)
        navs[strategy.name] = nav
        daily_returns_by_strategy[strategy.name] = daily_return
        # Full-history result: directly comparable with the existing Excel
        # 60/40 sheet, whose first observation is 2002-07-30.
        full_summaries[strategy.name] = performance_summary(nav, daily_return)
        # All headline metrics begin when the first model allocation is usable.
        oos_return = daily_return.loc[first_oos_date:]
        oos_nav = (1 + oos_return).cumprod()
        oos_summaries[strategy.name] = performance_summary(oos_nav, oos_return)
        allocations[strategy.name] = allocation
    strategy_nav = pd.DataFrame(navs)
    strategy_daily_returns = pd.DataFrame(daily_returns_by_strategy)
    strategy_daily_equity_weights = pd.DataFrame(allocations)
    strategy_monthly_equity_weights = strategy_daily_equity_weights.resample("ME").last()
    strategy_statistics_full = pd.DataFrame(full_summaries).T
    strategy_statistics_oos = pd.DataFrame(oos_summaries).T
    # Default Spyder object is full-history, matching the existing Excel
    # workbook.  Keep the OOS table separately for a fair ML comparison.
    strategy_statistics = strategy_statistics_full
    summary = strategy_statistics.copy()
    for column in ["Total Return", "CAGR", "Annualized Volatility", "Maximum Drawdown", "Ulcer Index"]:
        summary[column] = summary[column].map(lambda value: f"{value:.2%}")
    print("\nFull-history performance (matches the 60/40 Excel reporting period)")
    print(summary.to_string())
    print(f"\nOut-of-sample period for fair ML/DL/RL comparison: {first_oos_date.date()} ~ {prices.index[-1].date()}")
    print(strategy_statistics_oos.to_string())
    strategy_nav.to_csv("strategy_nav.csv")
    strategy_daily_equity_weights.to_csv("strategy_daily_equity_weights.csv")
    feature_df.to_csv("model_features.csv")
    if macro_df is not None:
        macro_df.to_csv("macro_features_lagged.csv")
    comparison_figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    # Direct Matplotlib calls avoid the blank upper panel that can occur when
    # pandas.DataFrame.plot is used on shared date axes in some Spyder setups.
    for name in strategy_nav.columns:
        axes[0].plot(strategy_nav.index, strategy_nav[name].to_numpy(), label=name, linewidth=1.5)
    axes[0].axvline(oos_start_date, color="black", linestyle="--", linewidth=1.0, label="OOS start")
    axes[0].axvspan(strategy_nav.index[0], oos_start_date, color="grey", alpha=0.10, label="Training / 60-40 only")
    axes[0].set_title("Walk-forward portfolio NAV")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.3)
    for name in strategy_monthly_equity_weights.columns:
        axes[1].plot(strategy_monthly_equity_weights.index, strategy_monthly_equity_weights[name].to_numpy(), label=name, linewidth=1.1)
    axes[1].axvline(oos_start_date, color="black", linestyle="--", linewidth=1.0, label="OOS start")
    axes[1].axvspan(strategy_monthly_equity_weights.index[0], oos_start_date, color="grey", alpha=0.10)
    axes[1].set_title("Equity allocation selected for each month")
    axes[1].set_ylabel("Equity weight")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    comparison_figure.tight_layout()
    comparison_figure.savefig(args.plot_file, dpi=160)
    plt.show()
    print(f"\nSaved: {args.plot_file}, strategy_nav.csv, strategy_daily_equity_weights.csv")


if __name__ == "__main__":
    main()
