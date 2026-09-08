#!/usr/bin/env python3
"""
Reproduce the TFTV Llama 3.1 normalized-utility plots.

Input CSVs:
  - llama_addition_raw.csv
  - llama_negation_raw.csv

The utility score is:
    0.5 * (MMLU / base_MMLU + GSM8K / base_GSM8K)

For Llama 3.1:
    base_MMLU  = 68.26
    base_GSM8K = 77.10

Task Vectors and CWS are shown as hollow gray markers and annotated
with † because they require fine-tuning.

Outputs:
  - six PNG figures
  - six PDF figures
  - normalized addition/negation CSVs
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

BASE_MMLU = 68.26
BASE_GSM8K = 77.10

# Colors chosen to match the paper's TFTV-vs-steering tradeoff figure.
PURPLE = "#c790c7"   # TFTV
BLUE = "#8aa4d4"     # inference steering
SALMON = "#d69494"   # attention-style baseline / Steer2Edit here
DARK_GRAY = "#686868"
MID_GRAY = "#8a8a8a"
LIGHT_GRID = "#dddddd"

METHOD_SPECS = {
    "Base": {
        "color": DARK_GRAY, "marker": "x",
        "label": "Base", "filled": True
    },
    "Steering": {
        "color": BLUE, "marker": "+",
        "label": "Steering", "filled": True
    },
    "Steer2Edit": {
        "color": SALMON, "marker": "o",
        "label": "Steer2Edit", "filled": True
    },
    "TFTV": {
        "color": PURPLE, "marker": "v",
        "label": "TFTV", "filled": True
    },
    "Task Vectors": {
        "color": DARK_GRAY, "marker": "s",
        "label": "Task Vectors†", "filled": False
    },
    "CWS": {
        "color": DARK_GRAY, "marker": "D",
        "label": "CWS†", "filled": False
    },
}

METHOD_ORDER = [
    "Base", "Steering", "Steer2Edit",
    "TFTV", "Task Vectors", "CWS"
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def add_normalized_utility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MMLU / Base"] = df["MMLU"] / BASE_MMLU
    df["GSM8K / Base"] = df["GSM8K"] / BASE_GSM8K
    df["Utility Score"] = 0.5 * (
        df["MMLU / Base"] + df["GSM8K / Base"]
    )
    return df


def plot_trait(df: pd.DataFrame, trait: str, experiment: str):
    sub = df[df["Trait"] == trait].copy()

    fig, ax = plt.subplots(figsize=(4.35, 3.9))

    for method in METHOD_ORDER:
        row = sub[sub["Method"] == method].iloc[0]
        spec = METHOD_SPECS[method]

        if spec["filled"]:
            ax.plot(
                row["Trait Score"],
                row["Utility Score"],
                linestyle="None",
                marker=spec["marker"],
                markersize=7.5,
                markeredgewidth=1.2,
                markeredgecolor=spec["color"],
                markerfacecolor=spec["color"],
                color=spec["color"],
                label=spec["label"],
                zorder=3,
            )
        else:
            ax.plot(
                row["Trait Score"],
                row["Utility Score"],
                linestyle="None",
                marker=spec["marker"],
                markersize=7.5,
                markeredgewidth=1.3,
                markeredgecolor=spec["color"],
                markerfacecolor="none",
                color=spec["color"],
                label=spec["label"],
                zorder=3,
            )

    # Base-utility reference line.
    ax.axhline(
        1.0,
        color=MID_GRAY,
        linewidth=0.9,
        linestyle="--",
        zorder=1,
    )

    ax.set_xlabel("Trait score")
    ax.set_ylabel("Utility score")
    ax.set_title("Sycophancy" if trait == "Sycophantic" else trait)

    # Paper-like grid/frame.
    ax.grid(True, color=LIGHT_GRID, linewidth=0.7, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(MID_GRAY)
        spine.set_linewidth(0.9)
    ax.tick_params(color=MID_GRAY, width=0.8, length=3)

    # Direction cue.
    direction = (
        "higher trait score →"
        if experiment == "Addition"
        else "← lower trait score"
    )
    ax.text(
        0.98, 0.03, direction,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7.5,
        color=DARK_GRAY,
    )

    # Tight limits with small padding.
    xmin, xmax = sub["Trait Score"].min(), sub["Trait Score"].max()
    xpad = max(3.5, 0.08 * max(1, xmax - xmin))
    ax.set_xlim(xmin - xpad, xmax + xpad)

    ymin, ymax = sub["Utility Score"].min(), sub["Utility Score"].max()
    ypad = max(0.012, 0.13 * max(0.01, ymax - ymin))
    ax.set_ylim(ymin - ypad, ymax + ypad)

    # Legend above the plot.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=3,
        frameon=True,
        edgecolor=DARK_GRAY,
        handlelength=1.2,
        columnspacing=1.0,
        handletextpad=0.4,
        borderpad=0.45,
    )

    fig.text(
        0.5, 0.005,
        "Utility = ½(MMLU/base + GSM8K/base).  † Requires fine-tuning.",
        ha="center", va="bottom",
        fontsize=7.2,
        color=DARK_GRAY,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.90])

    slug = trait.lower().replace(" ", "_")
    stem = f"llama_{experiment.lower()}_{slug}_reference_style"

    fig.savefig(OUT / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    add = pd.read_csv(HERE / "llama_addition_raw.csv")
    neg = pd.read_csv(HERE / "llama_negation_raw.csv")

    add = add_normalized_utility(add)
    neg = add_normalized_utility(neg)

    # Save the derived values so they can be checked independently.
    add.to_csv(OUT / "llama_addition_normalized_utility.csv", index=False)
    neg.to_csv(OUT / "llama_negation_normalized_utility.csv", index=False)

    for trait in ["Evil", "Hallucinating", "Sycophantic"]:
        plot_trait(add, trait, "Addition")
        plot_trait(neg, trait, "Negation")

    print(f"Wrote outputs to: {OUT}")


if __name__ == "__main__":
    main()
