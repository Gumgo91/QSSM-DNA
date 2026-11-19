"""Data figures of the paper (figures 2 to 5) from results/ (CPU,
matplotlib).

  python scripts/make_figures.py

  fig2  supervision density: lift over the untrained register against the
        number of supervised targets, transfer by objective and budget, and
        next-base loss on 10,240 held-out windows with count references
        (conditions_table.csv, all_metrics_long.csv, nll_large.csv)
  fig3  what carries the information: architecture ablations, learned
        damping rates, memory kernel and register size
        (all_metrics_long.csv, readout_full.csv, damping_rates.csv,
        memory_kernel.csv, scaling_long.csv)
  fig4  reference frame: QSSM against GRUs, a composition reference, k-mer
        regression and a CNN (all_metrics_long.csv, composition_floor.csv)
  fig5  terminal interface: hardware-noise sensitivity, error vs length and
        the lift over the untrained register under noise and finite shots
        (noise_metrics.csv, noise_vs_length.csv, noise_reference.csv)
Each figure is written to figures/<name>.png and .pdf. Panels show mean +/-
SD over replication seeds unless a 95 % interval is named; comparison panels
also show the seed-level points. Filled markers are encoders trained on
0.25 M tokens, open markers on 2.5 M tokens.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, "src")
from qssm_dna.experiments.jobs_noise import NOISE_SETTINGS  # noqa: E402

RES = Path("results")
OUT = Path("figures")
SEEDS = [22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, YELLOW, MAGENTA = "#6250d6", "#eda100", "#e87ba4"
GREY, DARK, LIGHT = "#8a8984", "#2b2b2a", "#c9c8c3"
EVALS = [("holdout_promoter", "Promoter holdout\n(chr9, chr10)"),
         ("gb_cohn:test", "Enhancers"),
         ("gb_promoter:test", "Promoter benchmark\n(development-exposed)"),
         ("gb_ocr:test", "Open chromatin")]
B10 = "_b2500000"
# objectives on the categorical axes of fig. 2b-d; those in TENFOLD were
# also trained on 2.5 M tokens
OBJECTIVES = [("qssm_endpoint", "1"), ("qssm_dense2", "2"),
              ("qssm_dense4", "4"), ("qssm_dense8", "8"),
              ("qssm_spread8", "8\nspread"), ("qssm_allpos", "all")]
TENFOLD = ("qssm_endpoint", "qssm_dense8", "qssm_allpos")
# registers of fig. 3b-c: condition, label, colour, marker face
REGISTERS = [("qssm_random_init", "untrained register", ORANGE, None),
             ("qssm_endpoint" + B10, "endpoint", GREY, "white"),
             ("qssm_dense8" + B10, "dense-tail-8", AQUA, "white"),
             ("qssm_allpos" + B10, "all-position", BLUE, "white")]
COUNT_REFS = [("order 0", "unigram"), ("order 1", "markov1"),
              ("order 2", "markov2"), ("order 5", "markov5")]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "font.family": "DejaVu Sans", "savefig.dpi": 300,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def long_table() -> pd.DataFrame:
    return pd.read_csv(RES / "all_metrics_long.csv")


def seed_vals(df, cond, readout, ev, metric="eval_mcc"):
    s = df[(df.condition == cond) & (df.readout == readout)
           & (df.eval_name == ev) & df.seed.isin(SEEDS)]
    return s.groupby("seed")[metric].first().reindex(SEEDS).values


def mean_sd(ax, x, vals, color, marker="o", ms=5, label=None, mfc=None):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return
    face = {} if mfc is None else {"mfc": mfc}
    ax.errorbar([x], [v.mean()], yerr=[v.std(ddof=1) if len(v) > 1 else 0],
                fmt=marker, color=color, ms=ms, capsize=2.5, lw=1.2,
                zorder=3, label=label, **face)


def mean_ci(ax, x, m, lo, hi, color, marker="o", ms=5, mfc=None):
    """Mean with the interval [lo, hi] of a contrast table."""
    face = {} if mfc is None else {"mfc": mfc}
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt=marker, color=color,
                ms=ms, capsize=2.5, lw=1.2, zorder=3, **face)


def points(ax, x, vals, jitter=0.0):
    v = np.asarray(vals, float)
    ax.scatter(np.full(len(v), x) * (1 + jitter), v, s=5, color=LIGHT,
               zorder=2)


def band(ax, vals, color, label):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return
    m, s = v.mean(), v.std(ddof=1)
    ax.axhspan(m - s, m + s, color=color, alpha=0.12, lw=0, zorder=0)
    ax.axhline(m, color=color, lw=0.8, ls="--", zorder=1, label=label)


def objective_color(cond: str) -> str:
    """One identity colour per objective across figures 2-4."""
    if "random_init" in cond:
        return ORANGE
    if "endpoint" in cond:
        return GREY
    if "allpos" in cond:
        return BLUE
    return AQUA                      # dense-tail with 2, 4 or 8 targets


def objective_style(cond: str) -> tuple:
    """(colour, marker) of an objective; eight evenly spaced targets are the
    violet square."""
    if "spread8" in cond:
        return VIOLET, "s"
    return objective_color(cond), "o"


def lift_table() -> pd.DataFrame:
    """Supervised targets and lift in holdout MCC over the untrained register
    (subset30, paired over seeds, 95 % t-interval) of each condition."""
    t = pd.read_csv(RES / "conditions_table.csv").set_index("condition")
    k = "lift_holdout_promoter_mcc"
    return t[["targets_mean", k, f"{k}_ci95_low", f"{k}_ci95_high"]].rename(
        columns={"targets_mean": "targets", k: "lift",
                 f"{k}_ci95_low": "low", f"{k}_ci95_high": "high"})


def nll_terminal() -> tuple:
    """Terminal next-base loss on the 10,240 held-out 64/128-bp windows:
    one row per QSSM checkpoint, and the count references on the same
    targets."""
    n = pd.read_csv(RES / "nll_large.csv")
    n = n[n.window_set == "L64_128"]
    ref = n[n.family == "count"].set_index("condition")["nll_terminal"]
    return (n[n.family == "qssm"],
            {lab: float(ref[c]) for lab, c in COUNT_REFS})


def damping_means() -> pd.DataFrame:
    """Per-seed amplitude-damping rate of each qubit, averaged over A, C, G
    and T (rows: condition, seed; columns: qubit 1..4)."""
    d = pd.read_csv(RES / "damping_rates.csv")
    d = d[d.seed.isin(SEEDS)]
    return d.pivot(index=["condition", "seed"], columns="qubit",
                   values="gamma_mean_acgt")


def kernel_relative(cond: str) -> np.ndarray:
    """[seeds, 251] readout sensitivity to a substitution at distance d from
    the last base read, divided by the sensitivity at d = 0 of the seed."""
    k = pd.read_csv(RES / "memory_kernel.csv")
    s = k[k.condition == cond].set_index("seed").reindex(SEEDS)
    return (s[[f"sens_d{d}" for d in range(251)]].values
            / s[["sens_d0"]].values)


def composition_reference() -> pd.Series:
    """MCC of logistic regression on GC fraction and CpG observed/expected
    ratio of the whole sequence (standardized features), per evaluation
    set."""
    c = pd.read_csv(RES / "composition_floor.csv")
    c = c[(c.kind == "composition") & (c.features == "gc_cpg")
          & (c.window == "all")]
    return c.set_index("eval_name")["lr_mcc"]


def noise_lifts() -> pd.DataFrame:
    """Lift in holdout MCC over the untrained register under the same noise
    and shot setting (subset30, paired over seeds, 95 % t-interval)."""
    n = pd.read_csv(RES / "noise_reference.csv", low_memory=False)
    n = n[(n.level == "contrast") & (n.metric == "eval_mcc")
          & (n.readout == "subset30") & (n.eval_set == "holdout_promoter")]
    return n.assign(shots=n.shots.astype(str)).set_index(
        ["condition", "setting", "shots"])[
        ["mean_diff", "ci95_low", "ci95_high"]]


def by_objective(ax, values):
    """Objectives on a categorical axis; 0.25 M tokens filled, 2.5 M tokens
    open and side by side where both exist. values(condition) -> seeds."""
    for i, (c, _) in enumerate(OBJECTIVES):
        col, mk = objective_style(c)
        both = c in TENFOLD
        x = i - 0.17 if both else i
        v = values(c)
        points(ax, x, v)
        mean_sd(ax, x, v, col, marker=mk)
        if both:
            w = values(c + B10)
            points(ax, i + 0.17, w)
            mean_sd(ax, i + 0.17, w, col, marker=mk, mfc="white")
    ax.set_xticks(range(len(OBJECTIVES)))
    ax.set_xticklabels([lab for _, lab in OBJECTIVES])
    ax.set_xlim(-0.6, len(OBJECTIVES) - 0.4)
    ax.set_xlabel("Supervised positions per training window")


def budget_handles() -> list:
    return [Line2D([], [], color=GREY, marker="o", ls="", ms=5,
                   label="0.25 M tokens"),
            Line2D([], [], color=GREY, marker="o", ls="", ms=5, mfc="white",
                   label="2.5 M tokens")]


def fig2_supervision():
    df = long_table()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    fig.subplots_adjust(wspace=0.36, hspace=0.62)

    # a: lift over the untrained register against supervised targets
    ax = axes[0, 0]
    lt = lift_table()
    ax.axhline(0, color=DARK, lw=0.6, zorder=1)
    # condition, label, displacement along x (dense-tail-8 and spread-8 have
    # the same number of targets), label anchor (lift, low or high end of
    # the interval), label offset in points, alignment
    right = ((6, 0), "left", "center")
    left = ((-6, 0), "right", "center")
    cells = [("qssm_endpoint", "1", 1.0, "lift", *right),
             ("qssm_dense2", "2", 1.0, "lift", *right),
             ("qssm_dense4", "4", 1.0, "lift", *right),
             ("qssm_dense8", "8", 0.90, "lift", *left),
             ("qssm_spread8", "8 spread", 1.04, "high", (5, 6), "center",
              "bottom"),
             ("qssm_allpos", "all", 1.0, "lift", *right),
             ("qssm_endpoint" + B10, "1", 1.0, "lift", *right),
             ("qssm_dense8" + B10, "8", 1.0, "lift", *left),
             ("qssm_allpos" + B10, "all", 1.0, "lift", *left),
             ("qssm_dense8_w256" + B10, "8 (256)", 1.0, "low", (0, -3),
              "center", "top")]
    for c, lab, shift, anchor, off, ha, va in cells:
        r = lt.loc[c]
        col, mk = objective_style(c)
        x = r.targets * shift
        mean_ci(ax, x, r.lift, r.low, r.high, col, marker=mk,
                mfc="white" if c.endswith(B10) else None)
        ax.annotate(lab, (x, r[anchor]), xytext=off,
                    textcoords="offset points", ha=ha, va=va, fontsize=6,
                    color=col)
    ax.set_xscale("log")
    ax.set_xlim(1.5e3, 4.5e6)
    ax.set_xlabel("Supervised targets during pretraining")
    ax.set_ylabel("Lift in holdout MCC\nover untrained register")
    ax.legend(handles=budget_handles(), frameon=False, fontsize=6,
              loc="upper left")
    ax.set_title("a", loc="left", fontweight="bold")

    # b, c: transfer by objective and budget
    for ax, (ev, title), letter in zip((axes[0, 1], axes[1, 0]), EVALS[:2],
                                       "bc"):
        band(ax, seed_vals(df, "qssm_random_init", "subset30", ev), ORANGE,
             "untrained register")
        by_objective(ax, lambda c, ev=ev: seed_vals(df, c, "subset30", ev))
        ax.set_ylabel("MCC")
        ax.set_title(title, fontsize=7)
        ax.set_title(letter, loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=6, loc="lower right")

    # d: terminal next-base loss on the large held-out window set
    ax = axes[1, 1]
    nq, refs = nll_terminal()
    for lab, y in refs.items():
        ax.axhline(y, color=GREY, lw=0.6, ls=":", zorder=1)
        ax.annotate(lab, (1, y), xycoords=("axes fraction", "data"),
                    xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=6, color=DARK)
    by_objective(ax, lambda c: nq[nq.condition == c].set_index("seed")[
        "nll_terminal"].reindex(SEEDS).values)
    ax.set_ylabel("Next-base loss (nats per base)")
    ax.set_title("d", loc="left", fontweight="bold")
    save(fig, "fig2_supervision_density")


def fig3_mechanism():
    df = long_table()
    ev = "holdout_promoter"
    fig = plt.figure(figsize=(6.7, 5.0))
    rows = fig.add_gridspec(2, 1, hspace=0.62)
    top = rows[0].subgridspec(1, 2, width_ratios=[1.3, 1.15], wspace=0.36)
    bottom = rows[1].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.62)

    # a: ablations of the untrained and of the trained register
    ax = fig.add_subplot(top[0])
    full = pd.read_csv(RES / "readout_full.csv")
    groups = [("untrained\nregister", ORANGE, full,
               [("qssm_random_init", "full"), ("init_norzz", "no couplings"),
                ("init_zerogamma", "no damping")]),
              ("dense-tail-8\n0.25 M", AQUA, df,
               [("qssm_dense8", "full"),
                ("qssm_dense8_staticgamma", "static damping"),
                ("qssm_dense8_norzz", "no couplings"),
                ("qssm_dense8_zerogamma", "no damping")]),
              ("all-position\n0.25 M", BLUE, df,
               [("qssm_allpos", "full"),
                ("qssm_allpos_norzz", "no couplings")])]
    x0, ticks, labels = 0.0, [], []
    for name, col, table, cats in groups:
        xs = x0 + np.arange(len(cats))
        for x, (c, lab) in zip(xs, cats):
            v = seed_vals(table, c, "subset30", ev)
            points(ax, x, v)
            mean_sd(ax, x, v, col)
            ticks.append(x)
            labels.append(lab)
        ax.text(xs.mean(), 0.985, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=DARK)
        x0 = xs[-1] + 1.7
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=6.5)
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    ax.set_ylim(-0.08, 0.62)
    ax.set_ylabel("MCC, promoter holdout")
    ax.set_title("a", loc="left", fontweight="bold")

    # b: learned amplitude-damping rate of each qubit
    ax = fig.add_subplot(top[1])
    gam = damping_means()
    qubits = np.array(gam.columns)
    for j, (c, lab, col, mfc) in enumerate(REGISTERS):
        g = gam.loc[c].reindex(SEEDS)
        x = qubits + (j - 1.5) * 0.17
        for xi, qb in zip(x, qubits):
            points(ax, xi, g[qb].values)
        face = {} if mfc is None else {"mfc": mfc}
        ax.plot(x, g.mean().values, color=col, marker="o", ms=4.5, lw=0.8,
                zorder=3, label=lab if mfc is None else f"{lab}, 2.5 M",
                **face)
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 1.5)
    ax.set_xticks(qubits)
    ax.set_xlabel("Qubit")
    ax.set_ylabel("Amplitude-damping rate")
    ax.legend(frameon=False, fontsize=5.5, loc="lower right",
              handlelength=1.2, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("b", loc="left", fontweight="bold")

    # c: memory kernel of the same registers
    ax = fig.add_subplot(bottom[0])
    d = np.arange(251)
    for c, lab, col, mfc in REGISTERS:
        rel = kernel_relative(c)
        m, s = rel.mean(0), rel.std(0, ddof=1)
        ax.fill_between(d, m - s, m + s, color=col, alpha=0.15, lw=0,
                        zorder=1)
        ax.plot(d, m, color=col, lw=1.2, zorder=3)
        ax.annotate(lab.replace(" ", "\n"), (d[-1], m[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6,
                    color=DARK, linespacing=1.0)
    # linear from 0 to 1, one unit per doubling beyond
    ax.set_xscale("symlog", base=2, linthresh=1, linscale=0.5)
    ax.set_xticks([0, 1, 2, 4, 8, 16, 32, 64, 128, 250])
    ax.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64", "128",
                        "250"])
    ax.minorticks_off()
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Distance of the substitution from the 3\u2032 end (bases)")
    ax.set_ylabel("Readout sensitivity\nrelative to the last base")
    ax.set_title("c", loc="left", fontweight="bold")

    # d: register size
    ax = fig.add_subplot(bottom[1])
    sc = RES / "scaling_long.csv"
    if sc.exists():
        s = pd.read_csv(sc)
        # two measurement settings at every size: all same-basis parities
        s = s[(s.eval_set == ev) & (s.readout_kind == "parities")]
        for cond, col, lab, dx in (("allpos", BLUE, "all-position", 0.0),
                                   ("allpos_norzz", DARK,
                                    "without couplings", 0.25),
                                   ("random_init", ORANGE,
                                    "untrained register", -0.25)):
            sub = s[s.variant == cond]
            if sub.empty:
                continue
            for q, gq in sub.groupby("q"):
                points(ax, q + dx, gq.eval_mcc.values)
            g = sub.groupby("q")["eval_mcc"].agg(["mean", "std"])
            ax.errorbar(g.index + dx, g["mean"], yerr=g["std"], color=col,
                        marker="o", ms=3.5, capsize=2, label=lab, zorder=3)
        ax.set_xticks([4, 6])
        ax.set_xlim(3.3, 6.7)
        ax.set_ylim(top=0.6)
        ax.legend(frameon=False, fontsize=5.5, loc="upper right",
                  handlelength=1.2)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("MCC, promoter holdout\n(6,000-sequence probes)")
    ax.set_title("d", loc="left", fontweight="bold")
    save(fig, "fig3_mechanism")


def fig4_references():
    df = long_table()
    comp = composition_reference()
    # condition, readout, tick label, colour, marker, marker face
    cats = [("qssm_random_init", "subset30", "untrained register", ORANGE, "o",
             None),
            ("qssm_endpoint", "subset30", "QSSM endpoint", GREY, "o", None),
            ("qssm_allpos", "subset30", "QSSM all-position", BLUE, "o", None),
            ("qssm_allpos" + B10, "subset30", "QSSM all-position, 2.5 M", BLUE, "o",
             "white"),
            ("gru8_allpos", "projected64", "GRU-8 all-position", DARK, "o", None),
            ("gru64_allpos", "projected64", "GRU-64 all-position", DARK, "o", None),
            ("gru64_dense8" + B10, "projected64", "GRU-64 dense-tail-8, 2.5 M",
             DARK, "o", "white"),
            ("gc_cpg_lr", None, "composition regression", DARK, "D", None),
            ("kmer_lr", None, "k-mer regression", DARK, "o", None),
            ("cnn", "supervised", "CNN", DARK, "o", None)]
    fig, axes = plt.subplots(1, 4, figsize=(6.7, 3.0), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, (ev, title) in zip(axes, EVALS):
        for i, (c, r, lab, col, mk, mfc) in enumerate(cats):
            if c == "gc_cpg_lr":
                v = np.array([comp[ev]])
            else:
                sub = df[(df.condition == c) & (df.eval_name == ev)]
                if r is not None:
                    sub = sub[sub.readout == r]
                v = (sub[sub.seed.isin(SEEDS)].groupby("seed")["eval_mcc"]
                     .first().values if c != "kmer_lr" else
                     sub["eval_mcc"].values)
            if len(v) == 0:
                continue
            points(ax, i, v)
            mean_sd(ax, i, v, col, marker=mk, ms=4.5 if mk == "D" else 5,
                    mfc=mfc)
        ax.axvline(6.5, color=LIGHT, lw=0.6, ls="--")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c[2] for c in cats], rotation=90, fontsize=6)
        ax.set_xlim(-0.7, len(cats) - 0.3)
        ax.set_title(title, fontsize=7)
    axes[0].set_ylabel("MCC")
    save(fig, "fig4_reference_frame")


def fig5_noise(condition: str = "qssm_allpos"):
    nm = pd.read_csv(RES / "noise_metrics.csv")
    nm["shots"] = nm["shots"].astype(str)
    # noise strengths of each setting, from jobs_noise.NOISE_SETTINGS
    for k in ("p1", "p2", "p_phi", "readout"):
        nm[f"nz_{k}"] = nm.setting.map(
            lambda s, k=k: NOISE_SETTINGS.get(s, {}).get(k, 0.0))
    nm = nm[nm.condition == condition]
    nl = pd.read_csv(RES / "noise_vs_length.csv")
    nl = nl[nl.condition == condition]
    ev = "holdout_promoter"
    ex = nm[(nm.shots == "exact") & (nm.eval_set == ev)
            & (nm.readout == "subset30")]

    def seed_mcc(setting, policy):
        e = ex[(ex.setting == setting) & (ex.policy == policy)]
        return e.groupby("seed").eval_mcc.first().reindex(SEEDS).values

    base = np.nanmean(seed_mcc("none", "A_noiseless_probe"))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9),
                             gridspec_kw={"width_ratios": [1.0, 1.55]})
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    axes = axes.ravel()
    ylim = (-0.08, 0.7)

    # a: each error type alone, probe kept from noiseless training
    ax = axes[0]
    ax.axhline(base, color=GREY, lw=0.6, ls=":")
    factors = [("p1", "1q depolarizing", BLUE, -0.06),
               ("p2", "2q depolarizing", ORANGE, -0.02),
               ("p_phi", "dephasing", AQUA, 0.02),
               ("readout", "readout flip", VIOLET, 0.06)]
    for f, lab, col, dx in factors:
        sub = ex[(ex.policy == "A_noiseless_probe") & (ex[f"nz_{f}"] > 0)]
        others = [f"nz_{g}" for g, *_ in factors if g != f]
        sub = sub[(sub[others] == 0).all(axis=1)]
        if sub.empty:
            continue
        g = (sub.groupby([f"nz_{f}", "seed"]).eval_mcc.first()
             .groupby(level=0).agg(["mean", "std"]))
        x = g.index.values * 10 ** dx      # small offsets on the log axis
        ax.errorbar(x, g["mean"], yerr=g["std"], color=col, marker="o",
                    ms=3, capsize=2, label=lab)
    ax.set_xscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlabel("Error probability")
    ax.set_ylabel("MCC, promoter holdout")
    ax.set_title("a", loc="left", fontweight="bold")
    ax.text(0.08, 1.02, "probe from noiseless training", fontsize=7,
            transform=ax.transAxes, va="bottom")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left", handlelength=1.2)

    # b: strongest settings, probe kept or refitted on noisy features
    ax = axes[1]
    ax.axhline(base, color=GREY, lw=0.6, ls=":")
    scen = [("none", "none"), ("p2_0.01", "2q\n$10^{-2}$"),
            ("pphi_0.05", "deph.\n$5{\\times}10^{-2}$"),
            ("moderate", "moderate"), ("current", "current")]
    for i, (st, lab) in enumerate(scen):
        for pol, dx, col, mk in (("A_noiseless_probe", -0.17, GREY, "o"),
                                 ("B_refit_probe", 0.17, BLUE, "s")):
            v = seed_mcc(st, pol)
            if np.all(np.isnan(v)):
                continue
            ax.scatter(np.full(len(v), i + dx), v, s=5, color=LIGHT,
                       zorder=2)
            mean_sd(ax, i + dx, v, col, marker=mk, ms=4)
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([lab for _, lab in scen], fontsize=6)
    ax.set_xlim(-0.6, len(scen) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_title("b", loc="left", fontweight="bold")
    ax.legend(handles=[Line2D([], [], color=GREY, marker="o", ls="",
                              ms=4, label="noiseless probe"),
                       Line2D([], [], color=BLUE, marker="s", ls="",
                              ms=4, label="probe refitted on noisy data")],
              frameon=False, fontsize=6, loc="upper right")

    # c: deviation of the noisy terminal state versus sequence length
    ax = axes[2]
    for name, col, lab in (("p2_0.01", ORANGE, "2q depol. $10^{-2}$"),
                           ("pphi_0.05", AQUA,
                            "dephasing $5{\\times}10^{-2}$"),
                           ("current", DARK, "current combined")):
        g = nl[nl.setting == name].groupby("L")["trace_distance_mean"].agg(
            ["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], color=col, marker="o",
                    ms=3, capsize=2, label=lab)
    ax.set_xscale("log", base=2)
    ls_ = sorted(nl.L.unique())
    ax.set_xticks(ls_)
    ax.set_xticklabels([str(v) for v in ls_], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Trace distance to\nnoiseless state")
    ax.set_title("c", loc="left", fontweight="bold")
    ax.set_ylim(0.15, None)
    ax.legend(frameon=False, fontsize=5.5, loc="lower center", handlelength=1.2)

    # d: lift of the 2.5 M-token encoders over the untrained register, each
    # evaluated under the same noise and shot setting
    ax = axes[3]
    lifts = noise_lifts()
    ax.axhline(0, color=DARK, lw=0.6, zorder=1)
    # setting, shots, and the two rows of the tick label
    variants = [("none", "exact", "noiseless,", "exact"),
                ("none", "1024", "noiseless,", "1,024 shots"),
                ("p2_0.001", "exact", "2q $10^{-3}$,", "exact"),
                ("moderate", "exact", "moderate,", "exact"),
                ("moderate", "1024", "moderate,", "1,024 shots")]
    for c, lab, col, dx in (("qssm_dense8" + B10, "dense-tail-8, 2.5 M",
                             AQUA, -0.15),
                            ("qssm_allpos" + B10, "all-position, 2.5 M",
                             BLUE, 0.15)):
        for i, (st, sh, *_) in enumerate(variants):
            r = lifts.loc[(c, st, sh)]
            mean_ci(ax, i + dx, r.mean_diff, r.ci95_low, r.ci95_high, col,
                    mfc="white")
        ax.plot([], [], "o", color=col, mfc="white", ms=5, label=lab)
    ax.set_xticks(range(len(variants)))
    # both rows are placed by their baselines, so the exponent of one label
    # does not displace it against its neighbours
    ax.tick_params(axis="x", pad=8)
    ax.set_xticklabels([v[2] for v in variants], fontsize=6, va="baseline")
    for i, v in enumerate(variants):
        ax.annotate(v[3], (i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -18.2), textcoords="offset points",
                    ha="center", va="baseline", fontsize=6)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_ylim(-0.02, 0.31)
    ax.set_ylabel("Lift in holdout MCC\nover untrained register")
    ax.set_title("d", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2)
    save(fig, "fig5_noise")


if __name__ == "__main__":
    for f in (fig2_supervision, fig3_mechanism, fig4_references, fig5_noise):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            import traceback
            print("skip", f.__name__, type(e).__name__, e)
            traceback.print_exc()
