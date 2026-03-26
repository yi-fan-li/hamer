"""
Compare two eval CSV files side by side.

Usage:
    python plot_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

CSV_A = "results/eval_regression_dinov3_base.csv"
LABEL_A = "DINOv3-B (unfrozen)"

CSV_B = "results/eval_regression.csv"
LABEL_B = "DINOv3-B (frozen, 4L head)"

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

COLOR_A = "#4C72B0"
COLOR_B = "#DD8452"


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
    df_a = load(CSV_A, LABEL_A)
    df_b = load(CSV_B, LABEL_B)

    n_metrics = len(SUMMARY_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    fig.suptitle(f"{LABEL_A}  vs  {LABEL_B}", fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(DATASETS))
    width = 0.35

    for ax, (metric_key, metric_label) in zip(axes, SUMMARY_METRICS):
        vals_a = pivot(df_a, metric_key).reindex(DATASETS)
        vals_b = pivot(df_b, metric_key).reindex(DATASETS)

        bars_a = ax.bar(x - width / 2, vals_a, width, label=LABEL_A, color=COLOR_A)
        bars_b = ax.bar(x + width / 2, vals_b, width, label=LABEL_B, color=COLOR_B)

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
        for bar in [*bars_a, *bars_b]:
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
            va = df_a[df_a["dataset"] == ds][df_a["metric_name"] == metric_key]["metric_value"]
            vb = df_b[df_b["dataset"] == ds][df_b["metric_name"] == metric_key]["metric_value"]
            va = va.values[0] if len(va) else float("nan")
            vb = vb.values[0] if len(vb) else float("nan")
            rows.append({"metric": metric_label, "dataset": ds, LABEL_A: va, LABEL_B: vb})
    table = pd.DataFrame(rows).set_index(["metric", "dataset"])
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
