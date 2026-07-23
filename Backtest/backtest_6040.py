"""60/40 portfolio backtest with monthly rebalancing.

Example:
    pip install yfinance pandas numpy matplotlib
    python backtest_6040.py --start 2005-01-01 --transaction-cost-bps 5
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS_PER_YEAR = 252


def load_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Download adjusted closing prices and remove dates missing for either asset."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("가격 데이터를 받지 못했습니다. 티커와 인터넷 연결을 확인하세요.")

    # yfinance returns a multi-index DataFrame for multiple tickers.
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna().rename_axis("Date")


def monthly_rebalanced_portfolio(
    prices: pd.DataFrame, weights: pd.Series, transaction_cost_bps: float = 0.0
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the same timing convention as the Excel workbook.

    For date *t*, the portfolio return uses the prior row's Scaled Weight and
    date *t* asset returns.  When *t* is the first trading day of a month, the
    target allocation is written to that row's Rebalanced/Scaled Weight and
    becomes effective for date *t+1* returns.  This is a close-to-next-day
    execution convention and deliberately avoids same-day look-ahead bias.
    """
    returns = prices.pct_change().fillna(0.0)
    target = weights.reindex(prices.columns).astype(float)
    if not np.isclose(target.sum(), 1.0):
        raise ValueError("포트폴리오 비중의 합은 1이어야 합니다.")

    return_history: list[float] = []
    portfolio_weight_history: list[pd.Series] = []
    rebalanced_weight_history: list[pd.Series] = []
    scaled_weight_history: list[pd.Series] = []
    rebalance_records: list[dict[str, float | pd.Timestamp]] = []
    cost_rate = transaction_cost_bps / 10_000

    # Row 0: initial 60/40 allocation, with no return available yet.
    previous_rebalanced_weight = target.copy()
    previous_scaled_weight = target.copy()
    pending_cost = 0.0
    for position, (date, daily_return) in enumerate(returns.iterrows()):
        if position == 0:
            portfolio_weight = target.copy()
            rebalanced_weight = target.copy()
            scaled_weight = target.copy()
            net_return = 0.0
        else:
            # Excel: Portfolio Return[t] = SUMPRODUCT(Scaled Weight[t-1], Return[t]).
            gross_return = float((previous_scaled_weight * daily_return).sum())
            net_return = (1 + gross_return) * (1 - pending_cost) - 1
            pending_cost = 0.0

            # "Portfolio Weight": unscaled end-of-day holdings without a rebalance.
            portfolio_weight = previous_rebalanced_weight * (1 + daily_return)
            month_changed = date.month != prices.index[position - 1].month
            if month_changed:
                # Match the workbook: reset weights using the prior row's total.
                rebalanced_weight = target * previous_rebalanced_weight.sum()
                scaled_weight = target.copy()
                turnover = 0.5 * (target - previous_scaled_weight).abs().sum()
                pending_cost = float(turnover * cost_rate)
                rebalance_records.append(
                    {
                        "Signal Date": date,
                        "Effective Date": prices.index[position + 1] if position + 1 < len(prices) else pd.NaT,
                        "Turnover": float(turnover),
                        "Transaction Cost": pending_cost,
                        **{f"Weight Before: {ticker}": float(previous_scaled_weight[ticker]) for ticker in target.index},
                    }
                )
            else:
                rebalanced_weight = portfolio_weight.copy()
                scaled_weight = rebalanced_weight / rebalanced_weight.sum()

        portfolio_weight_history.append(portfolio_weight.rename(date))
        rebalanced_weight_history.append(rebalanced_weight.rename(date))
        scaled_weight_history.append(scaled_weight.rename(date))
        return_history.append(net_return)
        previous_rebalanced_weight = rebalanced_weight
        previous_scaled_weight = scaled_weight

    daily_returns = pd.Series(return_history, index=prices.index, name="60/40 Return")
    portfolio_nav = (1 + daily_returns).cumprod().rename("60/40 NAV")
    portfolio_weights = pd.DataFrame(portfolio_weight_history, index=prices.index).rename_axis("Date")
    rebalanced_weights = pd.DataFrame(rebalanced_weight_history, index=prices.index).rename_axis("Date")
    scaled_weights = pd.DataFrame(scaled_weight_history, index=prices.index).rename_axis("Date")
    rebalance_log = pd.DataFrame(rebalance_records).set_index("Signal Date") if rebalance_records else pd.DataFrame()
    return portfolio_nav, daily_returns, portfolio_weights, rebalanced_weights, scaled_weights, rebalance_log


def longest_drawdown_duration(drawdown: pd.Series) -> int:
    """Return the longest consecutive underwater period in trading days."""
    underwater = drawdown < 0
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max())


def performance_summary(nav: pd.Series, daily_returns: pd.Series) -> pd.Series:
    years = (len(daily_returns) - 1) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        raise ValueError("성과를 계산하려면 두 개 이상의 거래일이 필요합니다.")
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    drawdown = nav / nav.cummax() - 1
    maximum_drawdown = drawdown.min()
    calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan
    downside_deviation = np.sqrt(np.mean(np.minimum(daily_returns.to_numpy(), 0.0) ** 2))
    sortino = (daily_returns.mean() * TRADING_DAYS_PER_YEAR) / (downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR))
    ulcer_index = np.sqrt(np.mean(drawdown.to_numpy() ** 2))
    return pd.Series(
        {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Volatility": annual_volatility,
            "Sharpe Ratio (rf=0%)": sharpe,
            "Maximum Drawdown": maximum_drawdown,
            "Calmar Ratio": calmar,
            "Sortino Ratio (target=0%)": sortino,
            "Ulcer Index": ulcer_index,
            "Longest Drawdown (Trading Days)": longest_drawdown_duration(drawdown),
            "Final Portfolio Value": nav.iloc[-1],
        }
    )


def plot_backtest(
    portfolio_nav: pd.Series,
    prices: pd.DataFrame,
    drawdown: pd.Series,
    initial_capital: float,
    output_file: str,
) -> None:
    """Plot asset and portfolio growth above the portfolio drawdown chart."""
    asset_growth = prices.div(prices.iloc[0]).mul(initial_capital)
    figure, (growth_axis, drawdown_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    asset_growth.plot(ax=growth_axis, linewidth=1.2, alpha=0.8)
    portfolio_nav.plot(ax=growth_axis, color="black", linewidth=2.2, label="60/40 Portfolio")
    growth_axis.set_title(f"Portfolio and Asset Growth (Initial Capital: ${initial_capital:,.0f})")
    growth_axis.set_ylabel("Portfolio Value")
    growth_axis.grid(alpha=0.3)
    growth_axis.legend()

    drawdown_percent = drawdown * 100
    drawdown_axis.fill_between(drawdown.index, drawdown_percent, 0, color="firebrick", alpha=0.35)
    drawdown_axis.plot(drawdown.index, drawdown_percent, color="firebrick", linewidth=0.8)
    drawdown_axis.set_title("Portfolio Drawdown")
    drawdown_axis.set_ylabel("Drawdown (%)")
    drawdown_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)


def make_calendar_return_table(monthly_returns: pd.Series) -> pd.DataFrame:
    """Convert a monthly return series into the familiar Year × Month table."""
    return_frame = monthly_returns.to_frame("Monthly Return")
    return_frame["Year"] = return_frame.index.year
    return_frame["Month"] = return_frame.index.month
    calendar_table = return_frame.pivot(index="Year", columns="Month", values="Monthly Return")
    calendar_table = calendar_table.reindex(columns=range(1, 13))
    calendar_table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    calendar_table["Annual Return"] = (1 + monthly_returns).groupby(monthly_returns.index.year).prod() - 1
    return calendar_table


def plot_return_calendar(return_table: pd.DataFrame, output_file: str) -> None:
    """Save a Year × Month return table as an annotated heatmap image."""
    values = return_table.to_numpy(dtype=float) * 100
    max_absolute_return = np.nanmax(np.abs(values))
    color_limit = max(max_absolute_return, 1.0)
    figure_height = max(7, len(return_table) * 0.38 + 2)
    figure, axis = plt.subplots(figsize=(17, figure_height))
    image = axis.imshow(values, cmap="RdYlGn", vmin=-color_limit, vmax=color_limit, aspect="auto")
    axis.set_xticks(range(len(return_table.columns)))
    axis.set_xticklabels(return_table.columns)
    axis.set_yticks(range(len(return_table.index)))
    axis.set_yticklabels(return_table.index.astype(str))
    axis.set_title("60/40 Portfolio Calendar Returns")
    axis.set_xlabel("Month")
    axis.set_ylabel("Year")

    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        if not np.isnan(value):
            text_color = "white" if abs(value) > color_limit * 0.55 else "black"
            axis.text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=8, color=text_color)

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Return (%)")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")


def main() -> None:
    # These globals are deliberately kept for inspection in Spyder's Variable Explorer.
    global prices_df, asset_returns, target_weights
    global portfolio_nav, portfolio_returns, portfolio_weights, rebalanced_weights, scaled_weights, daily_weights, rebalance_log
    global rolling_peak, drawdown, monthly_returns, annual_returns, debug_table, statistics, portfolio_growth, asset_growth
    global monthly_prices, monthly_asset_returns, monthly_portfolio_nav, monthly_drawdown, monthly_dataset
    global monthly_return_table, annual_return_table
    parser = argparse.ArgumentParser(description="60/40 포트폴리오 백테스트")
    # IEF의 Yahoo Finance 조정주가가 제공되는 첫 거래일입니다.
    parser.add_argument("--start", default="2002-07-30", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD), 기본값: 오늘")
    parser.add_argument("--equity", default="SPY", help="주식 ETF 티커")
    parser.add_argument("--bond", default="IEF", help="채권 ETF 티커")
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0, help="편도 거래비용 (bp)")
    parser.add_argument("--initial-capital", type=float, default=1.0, help="초기 투자금 (기본값: 1.0)")
    parser.add_argument("--plot-file", default="6040_backtest.png", help="성과 차트 파일명")
    parser.add_argument("--return-table-file", default="6040_return_table.png", help="월별·연도별 수익률 표 이미지 파일명")
    args = parser.parse_args()

    tickers = [args.equity, args.bond]
    prices_df = load_prices(tickers, args.start, args.end)
    target_weights = pd.Series({args.equity: 0.60, args.bond: 0.40}, name="Target Weight")
    asset_returns = prices_df.pct_change().rename(columns=lambda ticker: f"{ticker} Return")
    if args.initial_capital <= 0:
        raise ValueError("초기 투자금은 0보다 커야 합니다.")
    portfolio_nav, portfolio_returns, portfolio_weights, rebalanced_weights, scaled_weights, rebalance_log = monthly_rebalanced_portfolio(
        prices_df, target_weights, args.transaction_cost_bps
    )
    portfolio_nav = (portfolio_nav * args.initial_capital).rename("60/40 NAV")
    portfolio_growth = portfolio_nav
    asset_growth = prices_df.div(prices_df.iloc[0]).mul(args.initial_capital)
    # Kept as an alias for the existing Spyder workflow; it is the Excel
    # workbook's "Scaled Weight" table.
    daily_weights = scaled_weights
    rolling_peak = portfolio_nav.cummax().rename("Rolling Peak")
    drawdown = (portfolio_nav / rolling_peak - 1).rename("Drawdown")
    # Monthly dataset: compound daily returns within each calendar month.
    # This preserves the daily backtest's exact economics while providing a
    # clean month-end return series for analysis and reporting.
    monthly_prices = prices_df.resample("M").last()
    monthly_asset_returns = ((1 + asset_returns.fillna(0.0)).resample("M").prod() - 1).rename(
        columns=lambda name: name.replace(" Return", " Monthly Return")
    )
    monthly_returns = ((1 + portfolio_returns).resample("M").prod() - 1).rename("Monthly Return")
    monthly_portfolio_nav = portfolio_nav.resample("M").last().rename("Month-end Portfolio NAV")
    monthly_drawdown = (monthly_portfolio_nav / monthly_portfolio_nav.cummax() - 1).rename("Month-end Drawdown")
    annual_returns = ((1 + portfolio_returns).resample("Y").prod() - 1).rename("Annual Return")
    monthly_return_table = make_calendar_return_table(monthly_returns)
    annual_return_table = annual_returns.to_frame()
    debug_table = pd.concat(
        [
            prices_df,
            asset_returns,
            portfolio_weights.add_prefix("Portfolio Weight: "),
            rebalanced_weights.add_prefix("Rebalanced Weight: "),
            scaled_weights.add_prefix("Scaled Weight: "),
            portfolio_returns,
            portfolio_nav,
            rolling_peak,
            drawdown,
        ],
        axis=1,
    )
    monthly_dataset = pd.concat(
        [monthly_prices, monthly_asset_returns, monthly_returns, monthly_portfolio_nav, monthly_drawdown], axis=1
    )
    statistics = performance_summary(portfolio_nav, portfolio_returns)

    print(f"기간: {prices_df.index[0].date()} ~ {prices_df.index[-1].date()}")
    print(f"구성: {args.equity} 60%, {args.bond} 40% | 월간 리밸런싱")
    formatted_stats = statistics.copy()
    for metric in ["Total Return", "CAGR", "Annualized Volatility", "Maximum Drawdown", "Ulcer Index"]:
        formatted_stats[metric] = f"{statistics[metric]:.2%}"
    formatted_stats["Sharpe Ratio (rf=0%)"] = f"{statistics['Sharpe Ratio (rf=0%)']:.2f}"
    formatted_stats["Calmar Ratio"] = f"{statistics['Calmar Ratio']:.2f}"
    formatted_stats["Sortino Ratio (target=0%)"] = f"{statistics['Sortino Ratio (target=0%)']:.2f}"
    formatted_stats["Longest Drawdown (Trading Days)"] = f"{statistics['Longest Drawdown (Trading Days)']:.0f}"
    formatted_stats["Final Portfolio Value"] = f"${statistics['Final Portfolio Value']:,.2f}"
    print(formatted_stats.to_string())

    plot_backtest(portfolio_nav, prices_df, drawdown, args.initial_capital, args.plot_file)
    plot_return_calendar(monthly_return_table, args.return_table_file)
    print(f"차트 저장 완료: {args.plot_file}")
    print(f"수익률 표 이미지 저장 완료: {args.return_table_file}")


if __name__ == "__main__":
    main()
