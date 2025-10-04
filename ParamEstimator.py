import math
import copy
import time
from typing import Callable, Tuple, List, Optional
import numpy as np
import pandas as pd

from benchmark_structure import MMSimulator, compute_metrics, ASConfig, ExchangeSpec, OKXTop1CSVFeed
from volmodels import VolatilityModelBase, EWMAVolModel



class ParamsEstimator:
    """
    Minimal hyperparameter tuner for (gamma, k, tau).
    - Rebuilds feed and vol_model per trial via the provided factories.
    - Scores by Sharpe only.
    - Logs timing and ETA per trial when verbose=True.
    """

    def __init__(
        self,
        feed_factory: Callable[[], object],        # returns a fresh feed
        ex: "ExchangeSpec",
        cfg_base: "ASConfig",
        vol_model_factory: Callable[[], VolatilityModelBase],   # returns a fresh vol model
        initial_cash: float = 100_000.0,
        step_seconds: float = 1.0,
        seed: int = 123,
        verbose: bool = True,
        log_fn: Optional[Callable[[str], None]] = print,
    ):
        self.feed_factory = feed_factory
        self.ex = ex
        self.cfg_base = copy.deepcopy(cfg_base)
        self.vol_model_factory = vol_model_factory
        self.initial_cash = float(initial_cash)
        self.step_seconds = float(step_seconds)
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.log = (log_fn or (lambda *_: None))

    def _run_once(self, gamma: float, k: float, tau: float) -> Tuple[pd.DataFrame, float]:
        cfg = copy.deepcopy(self.cfg_base)
        cfg.gamma = float(gamma); cfg.k = float(k); cfg.tau = float(tau)

        feed = self.feed_factory()
        vol  = self.vol_model_factory()
        sim  = MMSimulator(feed, self.ex, cfg,
                           initial_cash=self.initial_cash,
                           step_seconds=self.step_seconds,
                           vol_model=vol)
        df = sim.run()
        m = compute_metrics(df)
        sharpe = m.get("Sharpe", float("nan"))
        if not np.isfinite(sharpe):
            sharpe = -1e9
        return df, sharpe

    def random_search(
        self,
        n_trials: int = 50,
        gamma_range: Tuple[float, float] = (0.5, 20.0),
        k_range: Tuple[float, float] = (3.0, 15.0),
        tau_range: Tuple[float, float] = (0.05, 1.0),
        log_gamma: bool = True,
        log_tau: bool = True,
    ):
        rows: List[dict] = []
        best = (-1e9, None, None)  # (score, params, df)

        t0 = time.time()
        for i in range(1, n_trials + 1):
            # sample
            gamma = (10 ** self.rng.uniform(np.log10(gamma_range[0]), np.log10(gamma_range[1]))
                     if log_gamma else self.rng.uniform(*gamma_range))
            k     = self.rng.uniform(*k_range)
            tau   = (10 ** self.rng.uniform(np.log10(tau_range[0]), np.log10(tau_range[1]))
                     if log_tau else self.rng.uniform(*tau_range))

            # run + time
            t_start = time.time()
            df, score = self._run_once(gamma, k, tau)
            dt = time.time() - t_start

            rows.append({"trial": i, "gamma": gamma, "k": k, "tau": tau, "Sharpe": score})

            # update best
            if score > best[0]:
                best = (score, (gamma, k, tau), df)

            # logging + ETA
            if self.verbose:
                elapsed = time.time() - t0
                avg = elapsed / i
                eta = avg * (n_trials - i)
                self.log(
                    f"[trial {i:>3}/{n_trials}] "
                    f"gamma={gamma:.4f} k={k:.4f} tau={tau:.4f} | "
                    f"Sharpe={score:.3f} | "
                    f"dt={dt:.2f}s avg={avg:.2f}s ETA={eta:.1f}s | "
                    f"best={best[0]:.3f}"
                )

        leaderboard = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
        best_score, (bg, bk, bt), best_df = best
        return best_score, (bg, bk, bt), best_df, leaderboard


if __name__ == "__main__":
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
    csv_path = "data/merged/BTC-USDC.csv.gz"
    initial_cash = 100_000.0

    def feed_factory():
        return OKXTop1CSVFeed(csv_path)


    def vol_model_factory():
        return EWMAVolModel(lam=0.97)  # or RealizedVolModel(window=60)


    est = ParamsEstimator(
        feed_factory=feed_factory,
        ex=ex,
        cfg_base=cfg,
        vol_model_factory=vol_model_factory,
        initial_cash=initial_cash,
        step_seconds=1.0,
        seed=123,
    )

    # random search
    best_score, (g, k, t), best_df, lb = est.random_search(
        n_trials=60,
        gamma_range=(0.5, 20.0),
        k_range=(3.0, 15.0),
        tau_range=(0.05, 1.0),
    )

    print(f"best Sharpe: {best_score:.3f} at gamma={g:.3f}, k={k:.3f}, tau={t:.3f}")
    lb.to_csv("performance/hparam_leaderboard.csv", index=False)

    # if you prefer a tiny grid:
    # best_score, (g,k,t), best_df, lb = est.grid_search(
    #     gammas=[1,2,4,6,8,12],
    #     ks=[4,6,8,10,12],
    #     taus=[0.1,0.25,0.5,1.0],
    # )
