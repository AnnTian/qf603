import math
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from benchmark_structure import *

if __name__ == "__main__":
    # need to fill in diff tick/lot/fee based on the trading pair
    # in OKX
    ## for the spot, maker fee: 0.08%, taker fee: 0.1%
    ## for the swap, maker fee: 0.02%, taker fee: 0.05%
    ex = ExchangeSpec(tick_size=0.1, lot_size=0.001, maker_fee=-0.0008, taker_fee=0.0010)
    cfg = ASConfig(
        gamma=20.0, k=8.0, tau=1.0,
        spread_cap_frac = 0.001,                     # spread cap: 0.1% of mid
        min_spread_frac = 0.0002,                     # 0.02% of mid
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
    initial_cash = 100_000.0

    feed = OKXTop1CSVFeed(csv_path)

    # vol_model
    vol_model = EWMAVolModel(lam=0.97)

    sim = MMSimulator(feed, ex, cfg, initial_cash=initial_cash, step_seconds=1.0, vol_model=vol_model)
    df = sim.run()

    metrics = compute_metrics(df)
    win_ratio = compute_win_ratio(sim.trade_steps, sim.trade_equity_delta)

    save_csv(df, out_put_csv_name)
    save_html_report(df, metrics, win_ratio, sim.total_fees, out_put_report_name, initial_cash)

    print("Final Equity:", metrics["Final Equity"])
    print("CAGR:", f"{metrics['CAGR']:.2%}", "Sharpe:", f"{metrics['Sharpe']:.2f}",
          "CumRet:", f"{metrics['Cumulative Return']:.2%}",
          "MDD:", f"{metrics['Max Drawdown']:.2%}",
          "DD Period(s):", f"{metrics['Max DD Period (seconds)']:.0f}",
          "Win:", f"{win_ratio:.2%}",
          "Fees:", f"{sim.total_fees:.2f}")















