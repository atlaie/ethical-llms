import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import pylab
from natsort import natsorted

# matplotlib.use('module://backend_interagg')
matplotlib.use("Qt5Agg")
# try:
#     matplotlib.use('module://backend_interagg')
# except:
#     matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from viz_data.get_UDHR_inputs import get_identities_dicts, get_rights_lists

color_names = [
    "dusty purple",
    "orange",
    "dark tan",
    "pink",
    "baby blue",
    "olive",
    "sea blue",
    "dusty red",
    "faded green",
    "amber",
    "windows blue",
]
colors = sns.xkcd_palette(color_names)

i_names = np.array(list(get_identities_dicts().keys()))
r_names = np.array([r for r in get_rights_lists()[0]])
pre_name = "../viz_data/UDHR_sara/"
post_name = "_1"


def get_results_list():
    results_mfq = natsorted(
        glob.iglob(
            f"../viz_data/UDHR_sara/responses{post_name}/UDHR_sara_results_l_*.csv"
        )
    )
    results_mfq = [fn for fn in results_mfq if "_l_" in fn]
    print(results_mfq)
    layers = list(range(len(results_mfq)))

    scores_before = np.zeros((1, len(r_names), len(i_names)))
    scores_after = np.zeros((len(layers), len(r_names), len(i_names)))

    for ll in layers:
        print(f"Layer {ll}")
        file_n = results_mfq[ll]
        iter = int(file_n.split("_rep_")[-1].split(".csv")[0])
        dat = pd.read_csv(file_n, delimiter="\t")
        dat["completions"] = np.clip(dat["completions"].values.astype(int), 0, 1)
        for ii, i_name in enumerate(i_names):
            for rr, r_name in enumerate(r_names):
                scores_before[0, rr, ii] = np.mean(
                    dat[
                        (dat.is_modified == False)
                        & (dat.right == r_name)
                        & (dat.id_class == i_name)
                    ]["completions"].values.astype(int)
                )
                scores_after[ll, rr, ii] = np.mean(
                    dat[
                        (dat.is_modified == True)
                        & (dat.right == r_name)
                        & (dat.id_class == i_name)
                    ]["completions"].values.astype(int)
                )

    print(scores_before.shape, np.mean(scores_before))
    print(scores_after.shape, np.mean(scores_after))
    return scores_before, scores_after


def get_names2plot(names):
    names2plot = []
    for r in names:
        r = r.split(" ")
        names2plot.append(
            "\n".join([" ".join(r[i : i + 10]) for i in range(0, len(r), 10)])
        )
    return names2plot


def get_pic(scores_before, scores_after):
    NUM_COLORS = scores_after.shape[0]

    cm = pylab.get_cmap("gist_rainbow")
    # cm = pylab.get_cmap('Set3')
    colors = cm(np.linspace(0, 1, NUM_COLORS))

    fig, axes = plt.subplots(1, 2, figsize=(20, 50.5))
    bar_height = 0.1
    offset = 0.2

    # plot by r_names
    ax = axes[0]
    names2plot = get_names2plot(r_names)
    ii = 0
    i_name = i_names[ii]
    ax.barh(
        y=names2plot,
        width=list(scores_before[0, :, ii]),
        color="black",
        edgecolor="black",
        alpha=0.2,
        height=bar_height,
        label="Unsteered",
    )
    for ll in range(scores_after.shape[0]):
        ax.barh(
            y=np.arange(len(names2plot)) - bar_height * (ll + 1),
            width=list(scores_after[ll, :, ii]),
            color=colors[ll],
            edgecolor=colors[ll],
            alpha=0.2,
            height=bar_height,
            label=f"Layer {ll}",
        )
    ax.legend()
    for ii in range(1, scores_after.shape[2]):
        i_name = i_names[ii]
        ax.barh(
            y=names2plot,
            width=list(scores_before[0, :, ii]),
            color="black",
            edgecolor="black",
            alpha=0.2,
            height=bar_height,
            label="Unsteered",
        )
        for ll in range(scores_after.shape[0]):
            ax.barh(
                y=np.arange(len(names2plot)) - bar_height * (ll + 1),
                width=list(scores_after[ll, :, ii]),
                color=colors[ll],
                edgecolor=colors[ll],
                alpha=0.2,
                height=bar_height,
                label=f"Layer {ll}",
            )

    # plot by i_names
    ax = axes[1]
    names2plot = get_names2plot(i_names)
    for rr in range(scores_after.shape[1]):
        r_name = r_names[rr]
        ax.barh(
            y=names2plot,
            width=list(scores_before[0, rr, :]),
            color="black",
            edgecolor="black",
            alpha=0.2,
            height=bar_height,
            label="Unsteered",
        )
        for ll in range(scores_after.shape[0]):
            ax.barh(
                y=np.arange(len(names2plot)) - bar_height * (ll + 1),
                width=list(scores_after[ll, rr, :]),
                color=colors[ll],
                edgecolor=colors[ll],
                alpha=0.2,
                height=bar_height,
                label=f"Layer {ll}",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(pre_name, f"rights{post_name}.png"))
    # plt.show()


def get_pic_by_rights(scores_before_s, scores_after_s):
    NUM_COLORS = scores_after_s.shape[0]

    cm = pylab.get_cmap("tab20c")
    # cm = pylab.get_cmap('gist_rainbow')
    # cm = pylab.get_cmap('Set3')
    colors = cm(np.linspace(0, 1, NUM_COLORS))

    for s in [10, 20, 30, 40]:
        scores_before = scores_before_s[:, s - 10 : s, :]
        scores_after = scores_after_s[:, s - 10 : s, :]
        names2plot = get_names2plot(r_names[s - 10 : s])
        fig, axes = plt.subplots(1, 1, figsize=(10, len(names2plot) * 10))
        bar_height = 0.05
        offset = 0.2

        # plot by r_names
        ax = axes
        ii = 0
        i_name = i_names[ii]
        ax.plot(
            scores_before[0, :, ii],
            np.arange(len(names2plot)),
            label="Unsteered",  # color=colors[ii],
            marker="o",
            markersize=15,
            markerfacecolor="None",
            markeredgecolor="black",
            linestyle="None",
        )

        for ll in range(scores_after.shape[0]):
            ax.plot(
                scores_after[ll, :, ii],
                np.arange(len(names2plot)) - bar_height * (ll + 1),
                label=f"SARA on Layer {ll}",  # color=colors[ii],
                marker="o",
                markersize=15,
                markerfacecolor="None",
                markeredgecolor=colors[ll],
                linestyle="None",
            )
        ax.legend()
        for ii in range(1, scores_after.shape[2]):
            i_name = i_names[ii]
            ax.plot(
                scores_before[0, :, ii],
                np.arange(len(names2plot)),
                label="Unsteered",  # color=colors[ii],
                marker="o",
                markersize=15,
                markerfacecolor="None",
                markeredgecolor="black",
                linestyle="None",
            )
            for ll in range(scores_after.shape[0]):
                ax.plot(
                    scores_after[ll, :, ii],
                    np.arange(len(names2plot)) - bar_height * (ll + 1),
                    label=f"SARA on Layer {ll}",  # color=colors[ii],
                    marker="o",
                    markersize=15,
                    markerfacecolor="None",
                    markeredgecolor=colors[ll],
                    linestyle="None",
                )
        ax.set_yticks(np.arange(len(names2plot)))
        ax.set_yticklabels(names2plot)
        ax.grid(which="both")
        ax.xaxis.grid(False)
        ax.set_xlabel("Accuracy")

        plt.tight_layout()
        plt.savefig(os.path.join(pre_name, f"rights{post_name}_{s}.png"))
        # plt.show()


def get_pic_by_identity(scores_before, scores_after):
    NUM_COLORS = scores_after.shape[0]

    cm = pylab.get_cmap("tab20c")
    # cm = pylab.get_cmap('gist_rainbow')
    # cm = pylab.get_cmap('Set3')
    colors = cm(np.linspace(0, 1, NUM_COLORS))

    names2plot = get_names2plot(i_names)
    fig, axes = plt.subplots(1, 1, figsize=(10, len(names2plot) * 5))
    bar_height = 0.05

    # plot by i_names
    ax = axes
    rr = 0
    r_name = r_names[rr]
    ax.plot(
        scores_before[0, rr, :],
        np.arange(len(names2plot)),
        label="Unsteered",  # color=colors[ii],
        marker="o",
        markersize=15,
        markerfacecolor="None",
        markeredgecolor="black",
        linestyle="None",
    )

    for ll in range(scores_after.shape[0]):
        ax.plot(
            scores_after[ll, rr, :],
            np.arange(len(names2plot)) - bar_height * (ll + 1),
            label=f"SARA on Layer {ll}",  # color=colors[ii],
            marker="o",
            markersize=15,
            markerfacecolor="None",
            markeredgecolor=colors[ll],
            linestyle="None",
        )
    ax.legend()
    for rr in range(1, scores_after.shape[1]):
        r_name = r_names[rr]
        ax.plot(
            scores_before[0, rr, :],
            np.arange(len(names2plot)),
            label="Unsteered",  # color=colors[ii],
            marker="o",
            markersize=15,
            markerfacecolor="None",
            markeredgecolor="black",
            linestyle="None",
        )
        for ll in range(scores_after.shape[0]):
            ax.plot(
                scores_after[ll, rr, :],
                np.arange(len(names2plot)) - bar_height * (ll + 1),
                label=f"SARA on Layer {ll}",  # color=colors[ii],
                marker="o",
                markersize=15,
                markerfacecolor="None",
                markeredgecolor=colors[ll],
                linestyle="None",
            )
    ax.set_yticks(np.arange(len(names2plot)))
    ax.set_yticklabels(names2plot)
    ax.grid(which="both")
    ax.xaxis.grid(False)
    ax.set_xlabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(pre_name, f"identity{post_name}.png"), dpi=1200)
    # plt.show()


if __name__ == "__main__":
    scores_before, scores_after = get_results_list()
    # get_pic(scores_before, scores_after)
    get_pic_by_rights(scores_before, scores_after)
    # get_pic_by_identity(scores_before, scores_after)
