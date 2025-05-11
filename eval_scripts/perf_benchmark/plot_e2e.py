#!/usr/bin/env python

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="path to output image file")
parser.add_argument("-i", "--input", help="path to input data file")
parser.add_argument("-t", "--title", help="title of the plot")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def hatch_bar(ax, df):
    bars = ax.patches
    hatches = "\\/-+o."
    hatch_repeats = 3
    all_hatches = []
    for h in hatches:
        all_hatches.extend([h * hatch_repeats] * len(df))
    for bar, hatch in zip(bars, all_hatches):
        bar.set_hatch(hatch)


def show_abs_bar_label(ax, barh, abs_df):
    for rect, val in zip(barh, abs_df.values):
        h = rect.get_height()
        w = rect.get_x() - 0.02
        val_str = "%g" % (val)

        h = h * 1.2

        ax.text(w, h, val_str)


def plot_overhead(df):
    systrace = df[df["method"] == "systrace"]["overhead"]
    systrace_std = df[df["method"] == "systrace"]["std"]
    monkey_patch = df[df["method"] == "monkey-patch"]["overhead"]
    monkey_patch_std = df[df["method"] == "monkey-patch"]["std"]
    selective = df[df["method"] == "selective"]["overhead"]
    selective_std = df[df["method"] == "selective"]["std"]
    print(systrace)

    figure, ax = plt.subplots(figsize=(10, 4))
    ind = np.arange(len(systrace))
    width = 0.2
    systrace_bars = ax.bar(
        ind,
        systrace.values,
        width - 0.03,
        yerr=systrace_std.values,
        capsize=5,
        bottom=0,
        label="settrace",
        color="#a6dc80",
        zorder=2,
    )
    monkey_bars = ax.bar(
        ind + width,
        monkey_patch.values,
        width - 0.03,
        yerr=monkey_patch_std.values,
        capsize=5,
        bottom=0,
        label="mpatch",
        color="#98c8df",
        zorder=2,
    )
    selective_bars = ax.bar(
        ind + 2 * width,
        selective.values,
        width - 0.03,
        yerr=selective_std.values,
        capsize=5,
        bottom=0,
        label="selective",
        color="#ffb665",
        zorder=2,
    )

    # display raw value as the bar label
    show_abs_bar_label(ax, systrace_bars, systrace)
    show_abs_bar_label(ax, monkey_bars, monkey_patch)
    show_abs_bar_label(ax, selective_bars, selective)

    ax.set_title(args.title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(systrace.index.values, rotation=0)
    ax.set_ylim(0.1, 2500)
    # ax.set_ylim(0.1, 200)
    ax.set_yscale("log")
    ax.axhline(y=1, linestyle="--", color="black", lw=0.8, zorder=3)
    hatch_bar(ax, systrace)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        edgecolor="black",
        fontsize=9,
        ncol=4,
        columnspacing=0.5,
    )
    ax.set_ylabel("Overhead")
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.input:
        sys.stderr.write("Must specify input data file\n")
        sys.exit(1)
    df = pd.read_csv(args.input, index_col=0)
    print("parsed data")
    print(df)
    plot_overhead(df)
