"""
Hand keypoint accuracy map for HOT3D eval results.

Plots the 21 keypoints at their anatomical positions on a hand diagram,
color-coded by PCK accuracy. One subplot per threshold (0.05 / 0.1 / 0.15).

Usage:
    python plot_hot3d_hand.py
    python plot_hot3d_hand.py --csv results/eval_regression.csv --out results/hot3d_hand.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Hand layout — right hand, palm-side view, fingers pointing up.
# Coordinates in [0,1] x [0,1]  (x: pinky-left → thumb-right, y: wrist-bottom → tips-top)
# ---------------------------------------------------------------------------
KP_POSITIONS = np.array([
    # 0  Wrist
    (0.50, 0.06),
    # 1-4  Thumb  (fans out to the right)
    (0.72, 0.20),   # Thumb_MCP
    (0.83, 0.32),   # Thumb_PIP
    (0.89, 0.44),   # Thumb_DIP
    (0.92, 0.56),   # Thumb_Tip
    # 5-8  Index
    (0.63, 0.40),   # Index_MCP
    (0.64, 0.55),   # Index_PIP
    (0.64, 0.67),   # Index_DIP
    (0.64, 0.78),   # Index_Tip
    # 9-12  Middle
    (0.51, 0.42),   # Middle_MCP
    (0.51, 0.58),   # Middle_PIP
    (0.51, 0.71),   # Middle_DIP
    (0.51, 0.83),   # Middle_Tip
    # 13-16  Ring
    (0.38, 0.40),   # Ring_MCP
    (0.37, 0.55),   # Ring_PIP
    (0.36, 0.67),   # Ring_DIP
    (0.35, 0.78),   # Ring_Tip
    # 17-20  Pinky
    (0.26, 0.35),   # Pinky_MCP
    (0.24, 0.48),   # Pinky_PIP
    (0.22, 0.58),   # Pinky_DIP
    (0.21, 0.67),   # Pinky_Tip
])

KP_NAMES = [
    "Wrist",
    "Thumb\nMCP", "Thumb\nPIP", "Thumb\nDIP", "Thumb\nTip",
    "Idx\nMCP",   "Idx\nPIP",  "Idx\nDIP",  "Idx\nTip",
    "Mid\nMCP",   "Mid\nPIP",  "Mid\nDIP",  "Mid\nTip",
    "Ring\nMCP",  "Ring\nPIP", "Ring\nDIP", "Ring\nTip",
    "Pky\nMCP",   "Pky\nPIP",  "Pky\nDIP",  "Pky\nTip",
]

# Bone connections [from, to]
BONES = [
    # Thumb chain
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index chain
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle chain
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring chain
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky chain
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckle bar
    (5, 9), (9, 13), (13, 17),
]

THRESHOLDS  = [0.05, 0.1, 0.15]
THRESHOLD_LABELS = ["PCK @ 0.05", "PCK @ 0.10", "PCK @ 0.15"]


def load_pck(csv_path: str):
    df  = pd.read_csv(csv_path)
    df  = df[df["dataset"] == "HOT3D-VAL"]
    pck = {}
    for thr in THRESHOLDS:
        vals = []
        for kp in range(21):
            row = df[df["metric_name"] == f"kp{kp}_pck_{thr}"]
            vals.append(float(row["metric_value"].iloc[0]) if len(row) else float("nan"))
        pck[thr] = np.array(vals)
    return pck


def draw_hand(ax, pck_values, title, cmap, norm, dot_scale=1800):
    """Draw the hand skeleton + keypoints coloured by pck_values on ax."""
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.05, 1.00)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Draw bones in light grey
    for a, b in BONES:
        xa, ya = KP_POSITIONS[a]
        xb, yb = KP_POSITIONS[b]
        ax.plot([xa, xb], [ya, yb], color="#cccccc", linewidth=2.5, zorder=1)

    # Draw keypoints
    sc = ax.scatter(
        KP_POSITIONS[:, 0], KP_POSITIONS[:, 1],
        c=pck_values,
        cmap=cmap, norm=norm,
        s=dot_scale, zorder=3,
        edgecolors="white", linewidths=1.5,
    )

    # Keypoint value labels (percentage inside each dot)
    for i, (x, y) in enumerate(KP_POSITIONS):
        pct = pck_values[i]
        # Choose white or black text depending on dot brightness
        rgb = cmap(norm(pct))[:3]
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt_color = "white" if lum < 0.55 else "black"
        ax.text(x, y, f"{pct:.0%}", ha="center", va="center",
                fontsize=7, color=txt_color, fontweight="bold", zorder=4)

    return sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/eval_regression_Hot3d_hamer_base.csv")
    parser.add_argument("--out", default="results/hot3d_hand.png")
    args = parser.parse_args()

    pck  = load_pck(args.csv)
    cmap = cm.RdYlGn          # red=low, yellow=mid, green=high
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle(
        "HaMeR (Vit) — Per-Keypoint PCK on HOT3D-VAL\n"
        "Colour: red = low accuracy → green = high accuracy",
        fontsize=13, y=1.01,
    )

    for ax, thr, label in zip(axes, THRESHOLDS, THRESHOLD_LABELS):
        sc = draw_hand(ax, pck[thr], label, cmap, norm)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("PCK", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    )

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
