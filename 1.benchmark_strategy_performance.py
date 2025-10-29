import time
import pandas as pd
from benchmark_structure import *
from volmodels import EWMAVolModel, RealizedVolModel, GARCHVolModel

if __name__ == "__main__":
    # need to fill in diff tick/lot/fee based on the trading pair
    # in OKX
    ## for the spot, maker fee: 0.08%, taker fee: 0.1%
    ## for the swap, maker fee: 0.02%, taker fee: 0.05%
    ex = ExchangeSpec(tick_size=0.1, lot_size=0.001, maker_fee=-0.0008, taker_fee=0.0010)
    cfg = ASConfig(
        gamma=5,
        k=8.0,
        tau=0.5,
        spread_cap_frac = 0.0005,                     # spread cap: 0.05% of mid
        min_spread_frac = 0.00001,                     # 0.001% of mid
        max_order_notional_frac=0.01,       # each order amount= 0.01 * equity
        min_cash_buffer_frac=0.05,          # cash buffer
        vol_scale_k=1.0,                    # higher -> smaller order qty
        target_inv_frac=0.0,                # target inventory: 0: neutral view; >0: bullish view; <0: bearish view
        max_inv_frac=0.25,                  # > max inventory: sell inventory
        hard_liq_frac=0.35
    )

    # data
    csv_path = "data/merged/BTC-USDC.csv.gz"
    out_put_csv_name = "performance/benchmark_BTC-USDC.csv"
    out_put_report_name = "performance/benchmark_BTC-USDC_report.html"
    out_put_quotes_name = "performance/benchmark_BTC-USDC_quotes.html"
    out_put_spread_analysis = "performance/benchmark_BTC-USDC_spread.html"
    initial_cash = 100_000.0

    # define models to benchmark
    models = {
        "EWMA": EWMAVolModel(lam=0.97),
        "GARCH": GARCHVolModel(window=600, p=3, q=1, refit_freq=600, dist='t', fast_mode=True),
        "Realized": RealizedVolModel(window=60),
    }

    leaderboard_rows = []

    for name, vol_model in models.items():
        print(f"\n============ Running model: {name} ============")
        feed = OKXTop1CSVFeed(csv_path)  # fresh iterator each run
        start_time = time.time()
        sim = MMSimulator(feed, ex, cfg, initial_cash=initial_cash, step_seconds=1.0, vol_model=vol_model)
        df = sim.run()

        metrics = compute_metrics(df)
        win_ratio = compute_win_ratio(sim.trade_steps, sim.trade_equity_delta)

        save_csv(df, out_put_csv_name)
        save_html_report(df, metrics, win_ratio, sim.total_fees, out_put_report_name, initial_cash)
        save_quotes_report(df, out_put_quotes_name)
        save_spread_analysis(df, out_put_spread_analysis, cfg)

        end_time = time.time()
        runtime_seconds = end_time - start_time

        leaderboard_rows.append({
            "model": name,
            "Final Equity": metrics["Final Equity"],
            "CAGR": metrics["CAGR"],
            "Sharpe": metrics["Sharpe"],
            "Cumulative Return": metrics["Cumulative Return"],
            "Max Drawdown": metrics["Max Drawdown"],
            "Max DD Period (seconds)": metrics["Max DD Period (seconds)"],
            "Win Ratio": win_ratio,
            "Total Fees": sim.total_fees,
            "Runtime (s)": runtime_seconds,
        })

        print("Final Equity:", metrics["Final Equity"]) 
        print("CAGR:", f"{metrics['CAGR']:.2%}", "Sharpe:", f"{metrics['Sharpe']:.2f}",
              "CumRet:", f"{metrics['Cumulative Return']:.2%}",
              "MDD:", f"{metrics['Max Drawdown']:.2%}",
              "DD Period(s):", f"{metrics['Max DD Period (seconds)']:.0f}",
              "Win:", f"{win_ratio:.2%}",
              "Fees:", f"{sim.total_fees:.2f}")
        print("Simulation Time:", f"{runtime_seconds:.2f} seconds")

    # write combined leaderboard
    try:
        leaderboard_df = pd.DataFrame(leaderboard_rows)
        leaderboard_df.to_csv("performance/hparam_leaderboard.csv", index=False)
        print("\nSaved leaderboard to performance/hparam_leaderboard.csv")
        print(leaderboard_df.sort_values("Sharpe", ascending=False).to_string(index=False))
    except Exception as e:
        print("Failed to write leaderboard:", e)















