"""
Plot per-keypoint PCK accuracy from eval_regression.csv (HOT3D eval).

Produces a grouped bar chart: one bar group per keypoint,
three bars per group (PCK@0.05, PCK@0.1, PCK@0.15).
Also prints a summary table and saves the figure to a PNG.

Usage:
    python plot_hot3d_eval.py
    python plot_hot3d_eval.py --csv results/eval_regression.csv --out results/hot3d_pck.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# OpenPose hand keypoint names (21 joints, HaMeR ordering)
KP_NAMES = [
    "Wrist",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP","Middle_PIP","Middle_DIP","Middle_Tip",
    "Ring_MCP",  "Ring_PIP",  "Ring_DIP",  "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip",
]

THRESHOLDS = [0.05, 0.1, 0.15]
COLORS     = ["#e07b54", "#5b8db8", "#6abf69"]   # orange, blue, green


def load_pck(csv_path: str):
    df = pd.read_csv(csv_path)
    # Filter to HOT3D-VAL rows only (file may contain multiple datasets)
    df = df[df["dataset"] == "HOT3D-VAL"]

    pck = {}   # threshold → np.array of shape (21,)
    for thr in THRESHOLDS:
        thr_str = str(thr)
        vals = []
        for kp in range(21):
            row = df[df["metric_name"] == f"kp{kp}_pck_{thr_str}"]
            vals.append(float(row["metric_value"].iloc[0]) if len(row) else float("nan"))
        pck[thr] = np.array(vals)

    kpl2_row = df[df["metric_name"] == "mode_kpl2"]
    kpl2 = float(kpl2_row["metric_value"].iloc[0]) if len(kpl2_row) else float("nan")

    avg = {thr: float(df[df["metric_name"] == f"kpAvg_pck_{thr}"]["metric_value"].iloc[0])
           for thr in THRESHOLDS}

    return pck, avg, kpl2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/eval_regression.csv")
    parser.add_argument("--out", default="results/hot3d_pck.png")
    args = parser.parse_args()

    pck, avg, kpl2 = load_pck(args.csv)

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f"\n{'Keypoint':<14}  {'PCK@0.05':>9}  {'PCK@0.10':>9}  {'PCK@0.15':>9}")
    print("-" * 48)
    for i, name in enumerate(KP_NAMES):
        print(f"{name:<14}  {pck[0.05][i]:>9.1%}  {pck[0.1][i]:>9.1%}  {pck[0.15][i]:>9.1%}")
    print("-" * 48)
    print(f"{'Average':<14}  {avg[0.05]:>9.1%}  {avg[0.1]:>9.1%}  {avg[0.15]:>9.1%}")
    print(f"\nmode_kpl2 = {kpl2:.4f}\n")

    # -----------------------------------------------------------------------
    # Grouped bar chart
    # -----------------------------------------------------------------------
    n_kp    = 21
    x       = np.arange(n_kp)
    width   = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(18, 6))

    for i, (thr, offset, color) in enumerate(zip(THRESHOLDS, offsets, COLORS)):
        bars = ax.bar(x + offset, pck[thr] * 100, width,
                      label=f"PCK@{thr}  (avg {avg[thr]:.1%})",
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.4)

    # Finger group dividers
    finger_boundaries = [0.5, 4.5, 8.5, 12.5, 16.5, 20.5]
    finger_labels_x   = [0, 2.5, 6.5, 10.5, 14.5, 18.5]
    finger_labels     = ["Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky"]
    for bnd in finger_boundaries[1:-1]:
        ax.axvline(bnd, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 105)
    ax.set_ylabel("PCK (%)", fontsize=11)
    ax.set_title("HaMeR (DINOv3-B) — Per-Keypoint PCK on HOT3D-VAL\n"
                 f"mode_kpl2 = {kpl2:.4f}  |  n = 3289 samples",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Finger group labels at the top
    for lx, lbl in zip(finger_labels_x, finger_labels):
        ax.text(lx, 102, lbl, ha="center", va="bottom", fontsize=8,
                color="dimgray", fontstyle="italic")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
