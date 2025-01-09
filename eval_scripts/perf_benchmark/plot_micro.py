#!/usr/bin/env python

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# matplotlib.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="path to output image file")
parser.add_argument("-i", "--input", help="path to input data file")
parser.add_argument("-t", "--title", help="title of the plot")


def plot(args):
    # parse data from CSV
    df = pd.read_csv(args.input)

    # preprocess func_names to remove the prefix
    df["API"] = df["API"].apply(lambda x: x.split(".")[-1])

    figure, ax = plt.subplots(figsize=(4.5, 2))

    sns.barplot(
        x="Wrapper Time Overhead Ratio",
        y="API",
        data=df,
        orient="h",
        ax=ax,
        palette="Set2",
    )
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_ylabel("")
    ax.set_xlabel("Ratio")
    ax.set_title(args.title)

    max_width = max([p.get_width() for p in ax.patches])
    for p in ax.patches:
        if p.get_width() < 2 / 3 * max_width:
            ax.text(
                p.get_width() + 5,
                p.get_y() + p.get_height() / 2,
                "%.2f" % p.get_width(),
                ha="left",
                va="center",
            )

    plt.tight_layout()
    if args.output:
        ax.margins(0, 0)
        plt.savefig(args.output, bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.input:
        sys.stderr.write("Must specify input data file\n")
        sys.exit(1)
    plot(args)
