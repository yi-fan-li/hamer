"""
Compare two eval CSV files side by side.

Usage:
    python plot_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

MODELS = [
    ("results/eval_regression_dinov3_base.csv", "DINOv3-B (fine-tuned)",  "#4C72B0"),
    ("results/eval_regression_dino4l.csv",      "DINOv3-B (4L head)",     "#DD8452"),
]

OUT = "results/comparison_plot.png"

# Summary metrics to plot (one panel each)
SUMMARY_METRICS = [
    ("mode_kpl2",      "mode_kpl2 ↓"),
    ("kpAvg_pck_0.05", "PCK@0.05 ↑"),
    ("kpAvg_pck_0.1",  "PCK@0.10 ↑"),
    ("kpAvg_pck_0.15", "PCK@0.15 ↑"),
]

DATASETS = [
    "NEWDAYS-TEST-ALL",
    "NEWDAYS-TEST-VIS",
    "NEWDAYS-TEST-OCC",
    "EPICK-TEST-ALL",
    "EPICK-TEST-VIS",
    "EPICK-TEST-OCC",
]

DATASET_LABELS = [d.replace("-TEST-", "\n") for d in DATASETS]


def load(path, label):
    df = pd.read_csv(path)
    df["model"] = label
    return df


def pivot(df, metric):
    sub = df[df["metric_name"] == metric].sort_values("timestamp")
    sub = sub.drop_duplicates(subset=["dataset"], keep="last")
    sub = sub[["dataset", "metric_value"]].set_index("dataset")
    return sub["metric_value"]


def main():
    dfs = [(load(path, label), label, color) for path, label, color in MODELS]
    n_models = len(dfs)

    n_metrics = len(SUMMARY_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    fig.suptitle("Model comparison", fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(DATASETS))
    width = 0.25
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for ax, (metric_key, metric_label) in zip(axes, SUMMARY_METRICS):
        all_bars = []
        for (df, label, color), offset in zip(dfs, offsets):
            vals = pivot(df, metric_key).reindex(DATASETS)
            bars = ax.bar(x + offset, vals, width, label=label, color=color)
            all_bars.append(bars)

        ax.set_title(metric_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_LABELS, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        if "pck" in metric_key:
            ax.set_ylim(0, 1.0)

        # annotate bar tops
        for bars in all_bars:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT}")

    # Also print a compact table
    print("\n--- Summary table ---")
    rows = []
    for metric_key, metric_label in SUMMARY_METRICS:
        for ds in DATASETS:
            row = {"metric": metric_label, "dataset": ds}
            for df, label, _ in dfs:
                v = df[(df["dataset"] == ds) & (df["metric_name"] == metric_key)]["metric_value"]
                row[label] = v.values[0] if len(v) else float("nan")
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["metric", "dataset"])
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
