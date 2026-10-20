"""Combine member forecasts online and evaluate everything.

Reads outputs/preds/*.npz (written by run_online.py), runs the online combiners
over the whole online period (burn-in included, so their weights are warm), and
evaluates on EVAL_START onwards: MAE, RMSE, rMAE vs naive, sMAPE, multivariate
Diebold-Mariano tests (Lago et al., 2021) and figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from data import MARKET, OUT, DayAheadData
from models import BOACombiner, InverseErrorCombiner

ROOT = Path(__file__).resolve().parents[1]
PRED, FIG = OUT / "preds", OUT / "figures"       # OUT: outputs/ (GB) or outputs/<market>/
EVAL_START = "2024-01-01"

HYBRID = ["rf_long", "rf_short", "lstm", "lstm_rf"]          # members of the online hybrid
# combinations fixed a priori (before seeing the new members' results)
HYBRID_ML = HYBRID + ["gbm", "tcn", "transformer", "dnn", "ddnn"]                 # all machine-learning members
HYBRID_ALL = HYBRID_ML + ["lear", "lear4", "merit", "chronos"]                    # everything
# ablation-selected subset: greedy backward elimination on 2024 only, validated on the untouched
# 2025-26 holdout. Before serving-time weight smoothing: holdout MAE 9.01 vs 9.18 for the full
# pool (DM p = 0.011), full period 8.59 vs 8.74 (p = 0.002). With smoothing: 8.99 vs 9.18
# (p = 0.007) and 8.56 vs 8.73 (p < 0.001). One member per model family.
HYBRID_SEL = ["lstm_rf", "gbm", "transformer", "lear", "merit", "chronos"]
LABEL = {
    "naive": "Naive (d-1/d-7)", "lear": "LEAR (daily recal.)", "lear4": "LEAR 4 windows",
    "merit": "Merit-order regression", "chronos": "Chronos-Bolt (zero-shot)",
    "rf_static": "RF static", "lstm_static": "LSTM static",
    "rf_long": "RF long window", "rf_short": "RF short window", "gbm": "Gradient boosting",
    "lstm": "LSTM online", "lstm_rf": "LSTM + RF error corr.", "tcn": "TCN online",
    "transformer": "Transformer online", "dnn": "DNN (Lago-style) online", "ddnn": "Distributional LSTM",
    "hyb_eq": "Hybrid equal weights", "hyb_inv": "Hybrid inverse-MAE",
    "hyb_boa": "Hybrid BOA", "hyb_boa_lear": "Hybrid BOA + LEAR",
    "hyb_boa_ml": "Hybrid BOA all ML", "hyb_boa_all": "Hybrid BOA all models", "hyb_eq_all": "Equal weights all models",
    "hyb_boa_sel": "Hybrid BOA selected 6",
}
# reference categorical palette (light), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False, "lines.linewidth": 2})


def load():
    data = DayAheadData()
    P = {}
    for f in sorted(PRED.glob("*.npz")):
        z = np.load(f)
        P[f.stem] = pd.DataFrame(z["pred"], index=pd.DatetimeIndex(z["days"]))   # dates, robust to data changes
    common = sorted(set.intersection(*[set(p.index) for p in P.values()]))
    P = {k: v.loc[common].values for k, v in P.items()}
    days = pd.DatetimeIndex(common)
    y = data.y[[data.index_of(d) for d in days]].astype(np.float64)
    return days, y, P


def run_combiner(comb, members, P, y):
    F = np.stack([P[m] for m in members], -1)                 # (n, 24, K)
    out, W = np.zeros(y.shape), np.zeros((len(y), F.shape[-1]))
    for i in range(len(y)):
        out[i] = comb.combine(F[i])
        W[i] = getattr(comb, "w", np.full((24, F.shape[-1]), np.nan)).mean(0)
        comb.update(F[i], y[i])                               # realised prices become known after the day
    return out, W


def dm_test(y, p1, p2):
    """Multivariate DM test (Lago et al., 2021): H0 'p1 not less accurate than p2'
    is tested against 'p2 is more accurate'; returns the one-sided p-value."""
    d = np.abs(y - p1).sum(1) - np.abs(y - p2).sum(1)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return 1 - stats.norm.cdf(t)


def metrics(y, p, naive):
    e = p - y
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "rMAE": np.abs(e).mean() / np.abs(naive - y).mean(),
            "sMAPE %": 100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-9))}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    days, y, P = load()
    print(f"online days: {days[0].date()} -> {days[-1].date()} ({len(days)}), members: {sorted(P)}")

    hyb = [m for m in HYBRID if m in P]
    P["hyb_eq"] = np.mean([P[m] for m in hyb], 0)
    P["hyb_inv"], _ = run_combiner(InverseErrorCombiner(len(hyb)), hyb, P, y)
    P["hyb_boa"], W = run_combiner(BOACombiner(len(hyb)), hyb, P, y)
    if "lear" in P:
        P["hyb_boa_lear"], _ = run_combiner(BOACombiner(len(hyb) + 1), hyb + ["lear"], P, y)
    for key, members in [("hyb_boa_ml", HYBRID_ML), ("hyb_boa_all", HYBRID_ALL), ("hyb_boa_sel", HYBRID_SEL)]:
        if all(m in P for m in members):
            P[key], _ = run_combiner(BOACombiner(len(members)), members, P, y)
    if all(m in P for m in HYBRID_ALL):
        P["hyb_eq_all"] = np.mean([P[m] for m in HYBRID_ALL], 0)

    ev = days >= pd.Timestamp(EVAL_START)
    order = [k for k in LABEL if k in P]
    rows = []
    periods = [("2024-01 → end", ev)] + [(str(y), days.year == y) for y in sorted(set(days.year)) if y >= 2024]
    for period, mask in periods:
        for k in order:
            rows.append({"period": period, "model": LABEL[k], **metrics(y[mask], P[k][mask], P["naive"][mask])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        for period, g in res.groupby("period", sort=False):
            print(f"\n=== {period} ===\n" + g.drop(columns="period").set_index("model").to_string())

    # ---- DM p-value matrix: cell (row, col) = p-value that col is more accurate than row
    dm = pd.DataFrame(index=[LABEL[k] for k in order], columns=[LABEL[k] for k in order], dtype=float)
    for a in order:
        for b in order:
            if a != b:
                dm.loc[LABEL[a], LABEL[b]] = dm_test(y[ev], P[a][ev], P[b][ev])
    dm.to_csv(OUT / "dm_pvalues.csv")

    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(order)), max(7.5, 0.52 * len(order))))
    im = ax.imshow(dm.values.astype(float), cmap="Blues_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(order)), dm.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(order)), dm.index)
    ax.grid(False)
    for i in range(len(order)):
        for j in range(len(order)):
            v = dm.values[i, j]
            if np.isfinite(v) and v < 0.1:
                ax.text(j, i, f"{v:.2f}" if v >= 0.01 else "<.01", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.03 else INK)
    fig.colorbar(im, ax=ax, label="p-value (column more accurate than row)")
    ax.set_title("Diebold–Mariano tests, 2024-01 → end", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "dm_test.png", dpi=150); plt.close(fig)

    # ---- monthly MAE of key models
    show = [k for k in ["naive", "lear", "rf_static", "lstm_static", "hyb_boa"] if k in P]
    mon = pd.DataFrame({LABEL[k]: np.abs(P[k] - y).mean(1) for k in show}, index=days).resample("MS").mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for c, k in zip(C, show):
        ax.plot(mon.index, mon[LABEL[k]], color=c, label=LABEL[k], marker="o", ms=4)
    ax.axvline(pd.Timestamp(EVAL_START), color=INK2, lw=1, ls="--")
    ax.text(pd.Timestamp(EVAL_START), ax.get_ylim()[1], " evaluation starts", va="top", color=INK2, fontsize=9)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Monthly MAE: online hybrid vs static models and benchmarks", loc="left", color=INK)
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "monthly_mae.png", dpi=150); plt.close(fig)

    # ---- BOA weights over time (averaged over the 24 hours)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    Wd = pd.DataFrame(W, index=days, columns=[LABEL[m] for m in hyb]).rolling(7).mean()
    ax.stackplot(Wd.index, Wd.T.values, colors=C[:len(hyb)], labels=Wd.columns, edgecolor="#fcfcfb", linewidth=0.5)
    ax.set_ylim(0, 1); ax.set_ylabel("weight (7-day mean)")
    ax.set_title("BOA combination weights of the hybrid members", loc="left", color=INK)
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(FIG / "boa_weights.png", dpi=150); plt.close(fig)

    # ---- example fortnight
    sel = (days >= "2025-01-13") & (days < "2025-01-27")
    t = pd.date_range(days[sel][0], periods=sel.sum() * 24, freq="h")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, y[sel].ravel(), color=INK, lw=1.5, label="Actual")
    for c, k in zip([C[0], C[1], C[2]], ["lear", "lstm", "hyb_boa"]):
        if k in P:
            ax.plot(t, P[k][sel].ravel(), color=c, lw=1.4, label=LABEL[k])
    ax.set_ylabel("EUR/MWh"); ax.legend(ncol=4, fontsize=9)
    ax.set_title("Day-ahead forecasts, two weeks in January 2025", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "example_forecast.png", dpi=150); plt.close(fig)

    hours = (days.values[:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
    pd.DataFrame({k: P[k].ravel() for k in order} | {"actual": y.ravel()}, index=pd.DatetimeIndex(hours)) \
        .to_csv(OUT / "forecasts_hourly.csv", index_label=f"datetime_{ {'GB': 'uk'}.get(MARKET, MARKET.lower())}_local")
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
