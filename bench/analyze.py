"""Raw logs -> summary tables + figures (Benchmarking p.31, p.35). Never edit the outputs by hand.

    python bench/analyze.py                         # every device under results/raw/
    python bench/analyze.py --meter meter.csv       # add the inline meter (columns t_unix,watts)

Writes results/summary.csv (one row per device x variant: median over repetitions + IQR),
results/accuracy.csv (mAP, drop vs FP32, quality gate) and results/figures/*.png.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from meta import RAW, REPO  # noqa: E402
from power import load_meter_csv  # noqa: E402

OUT, FIG = REPO / "results", REPO / "results" / "figures"
GATE_POINTS = 1.0  # max mAP@0.5:0.95 drop vs FP32, in points (p.31 "e.g. > 1 % drop"): see docs/protocol.md
# Reference categorical palette (dataviz skill), fixed order: colour follows the variant, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "lines.linewidth": 2,
        "savefig.dpi": 160,
    }
)


def window_mean(df: pd.DataFrame, col: str, t0: float, t1: float):
    if col not in df:
        return None
    w = df[(df.t_unix >= t0) & (df.t_unix <= t1)][col]
    return float(w.mean()) if len(w) else None


def meter_mean(meter: pd.DataFrame | None, t0: float, t1: float):
    return None if meter is None else window_mean(meter, "watts", t0, t1)


def idle_power(device_dir: Path, meter: pd.DataFrame | None) -> dict:
    """Latest idle run of this device: P_idle from the PMIC and/or the meter."""
    runs = sorted((device_dir / "idle").glob("*/meta.json"))
    if not runs:
        return {}
    m = json.loads(runs[-1].read_text())
    sys_df = pd.read_csv(runs[-1].parent / "system.csv")
    return {"pmic_w": window_mean(sys_df, "pmic_w", m["t_start"], m["t_end"]), "meter_w": meter_mean(meter, m["t_start"], m["t_end"])}


def per_run(run_dir: Path, idle: dict, meter: pd.DataFrame | None) -> tuple[dict, pd.Series]:
    m = json.loads((run_dir / "meta.json").read_text())
    frames = pd.read_csv(run_dir / "frames.csv")
    steady = frames[~frames.warmup]
    sys_df = pd.read_csv(run_dir / "system.csv")
    t0, t1, n = m["t_steady"], m["t_end"], m["steady_frames"]
    flags = sorted(set(sys_df.throttled.dropna().astype(str)) - {"0x0"}) if "throttled" in sys_df else []
    row = {
        "device": m["config"]["device"],
        "variant": m["variant"],
        "rep": m["rep"],
        "frames": n,
        "p50_ms": steady.total_ms.quantile(0.50),
        "p95_ms": steady.total_ms.quantile(0.95),
        "p99_ms": steady.total_ms.quantile(0.99),
        "pre_p50_ms": steady.pre_ms.median(),
        "infer_p50_ms": steady.infer_ms.median(),
        "post_p50_ms": steady.post_ms.median(),
        "fps": n / (t1 - t0),
        "load_s": m["load_s"],
        "cpu_pct": window_mean(sys_df, "cpu_pct", t0, t1),
        "rss_peak_mb": float(sys_df.rss_mb.max()),
        "temp_max_c": float(sys_df.temp_c.max()) if sys_df.temp_c.notna().any() else None,
        "throttled": ",".join(flags) or None,
    }
    for src, active in {"pmic": window_mean(sys_df, "pmic_w", t0, t1), "meter": meter_mean(meter, t0, t1)}.items():
        if active is not None:
            row[f"{src}_w"] = active
            if idle.get(f"{src}_w") is not None:  # E/inference = (P - P_idle) * T / N (p.25)
                row[f"{src}_mj_per_inf"] = (active - idle[f"{src}_w"]) * (t1 - t0) / n * 1e3
    return row, steady.total_ms


def iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in runs.columns if c not in ("device", "variant", "rep") and pd.api.types.is_numeric_dtype(runs[c])]
    g = runs.groupby(["device", "variant"])
    out = g[num].median()
    out["reps"] = g.size()
    out["p95_iqr_ms"] = g.p95_ms.agg(iqr)
    out["fps_iqr"] = g.fps.agg(iqr)
    out["throttled_any"] = g.throttled.agg(lambda s: s.notna().any())
    return out.reset_index()


def reference(variant: str) -> str:
    """The 'PC' model a variant is compared to: same arch, weights and input size, unpruned, FP32 ONNX."""
    arch, tag, size = variant.split("-")[:3]
    unpruned = re.sub(r"_p\d+(ft)?$", "", tag)
    return f"{arch}-{unpruned}-{size}-fp32-onnx"


def accuracy_table() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) | {"device": p.parent.parent.name} for p in RAW.glob("*/accuracy/*.json")]
    if not rows:
        return pd.DataFrame()
    acc = pd.DataFrame(rows)[["device", "variant", "map50_95", "map50", "precision", "recall", "f1", "n_images"]]
    ref = acc.groupby("variant").map50_95.max()
    acc["reference"] = acc.variant.map(reference)
    acc["drop_points"] = [round(100 * (ref.get(r, np.nan) - m), 2) for r, m in zip(acc.reference, acc.map50_95, strict=True)]
    acc["gate_pass"] = acc.drop_points <= GATE_POINTS
    return acc.sort_values(["device", "variant"])


def chunks(items: list, n: int = len(SERIES)):
    return [items[i : i + n] for i in range(0, len(items), n)] or [[]]


def title(ax, text: str, label: str | None):
    ax.set_title(text + (f"\n{label}" if label else ""), loc="left", color=INK, fontsize=10)


def fig_cdf(device: str, samples: dict, label: str | None):
    for k, names in enumerate(chunks(sorted(samples))):
        fig, ax = plt.subplots(figsize=(7, 4))
        for color, name in zip(SERIES, names, strict=False):  # names <= 8 per chunk
            x = np.sort(samples[name])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, label=name)
        ax.axhline(0.95, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set(xlabel="pipeline latency, frame in RAM -> boxes (ms)", ylabel="fraction of frames (dashed: p95)", xscale="log")
        title(ax, f"{device}: latency CDF (all repetitions pooled)", label)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG / f"latency_cdf_{device}{f'_{k}' if k else ''}.png")
        plt.close(fig)


def fig_box(device: str, samples: dict, label: str | None):
    names = sorted(samples, key=lambda n: np.median(samples[n]))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.4))
    ax.boxplot(
        [samples[n] for n in names],
        vert=False,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": SERIES[0], "edgecolor": SERIES[0], "alpha": 0.35},
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
    )
    ax.set_yticks(range(1, len(names) + 1), names, fontsize=7)
    ax.set(xlabel="pipeline latency (ms), whiskers 1.5 IQR, outliers hidden", xscale="log")
    ax.grid(axis="y", visible=False)
    title(ax, f"{device}: latency per variant", label)
    fig.tight_layout()
    fig.savefig(FIG / f"latency_box_{device}.png")
    plt.close(fig)


def fig_stages(device: str, summary: pd.DataFrame, label: str | None):
    s = summary[summary.device == device].sort_values("p50_ms")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(s) + 1.8))
    left = np.zeros(len(s))
    stages = [("pre_p50_ms", "pre-processing"), ("infer_p50_ms", "inference"), ("post_p50_ms", "post-processing")]
    for color, (col, name) in zip(SERIES, stages, strict=False):
        ax.barh(s.variant, s[col], left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.6, label=name)
        left += s[col].to_numpy()
    ax.set(xlabel="median time per stage (ms)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    title(ax, f"{device}: where the time goes", label)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.12 if label else 1.04))
    fig.tight_layout()
    fig.savefig(FIG / f"stages_{device}.png")
    plt.close(fig)


def fig_pareto(device: str, summary: pd.DataFrame, acc: pd.DataFrame, label: str | None):
    """Latency x accuracy for one device. A CPU variant's accuracy is the same file everywhere,
    so a device without its own accuracy runs uses the VM's (said in the title)."""
    s = summary[summary.device == device][["variant", "p95_ms"]]
    own = (acc.device == device).any()
    a = acc[acc.device == device] if own else acc
    d = s.merge(a.groupby("variant").map50_95.first().reset_index(), on="variant").sort_values("p95_ms")
    if d.empty:
        return
    front = d[d.map50_95 >= d.map50_95.cummax()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(front.p95_ms, front.map50_95, color=MUTED, linewidth=1, drawstyle="steps-post", zorder=1)
    ax.scatter(d.p95_ms, d.map50_95, s=40, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=2)
    for _, r in d.iterrows():
        ax.annotate(r.variant, (r.p95_ms, r.map50_95), textcoords="offset points", xytext=(5, 3), fontsize=7, color=INK)
    ax.set(xlabel="p95 pipeline latency (ms, median of repetitions)", ylabel="mAP@0.5:0.95", xscale="log")
    title(ax, f"{device}: accuracy vs latency, grey = Pareto front{'' if own else ' (accuracy measured on another device)'}", label)
    fig.tight_layout()
    fig.savefig(FIG / f"pareto_{device}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meter", type=Path, help="inline meter log, columns t_unix,watts")
    ap.add_argument("--device", help="only this device")
    args = ap.parse_args()
    meter = load_meter_csv(args.meter) if args.meter else None
    FIG.mkdir(parents=True, exist_ok=True)

    rows, samples, labels = [], {}, {}
    for device_dir in sorted(p for p in RAW.iterdir() if p.is_dir() and (not args.device or p.name == args.device)):
        idle = idle_power(device_dir, meter)
        for meta in sorted(device_dir.glob("*/*/meta.json")):
            if meta.parent.parent.name in ("idle", "accuracy"):
                continue
            row, total = per_run(meta.parent, idle, meter)
            rows.append(row)
            samples.setdefault(row["device"], {}).setdefault(row["variant"], []).extend(total)
            labels[row["device"]] = json.loads(meta.read_text())["config"].get("label")
    acc = accuracy_table()
    if not acc.empty:
        acc.to_csv(OUT / "accuracy.csv", index=False)
        print(acc.drop(columns="reference").to_string(index=False))
    if not rows:
        print("no runs yet under results/raw/")
        return
    summary = summarize(pd.DataFrame(rows))
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.3f")
    print(summary[["device", "variant", "reps", "p50_ms", "p95_ms", "p95_iqr_ms", "fps"]].to_string(index=False, float_format="%.1f"))
    for device, by_variant in samples.items():
        fig_cdf(device, by_variant, labels[device])
        fig_box(device, by_variant, labels[device])
        fig_stages(device, summary, labels[device])
        if not acc.empty:
            fig_pareto(device, summary, acc, labels[device])
    print(f"figures -> {FIG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
