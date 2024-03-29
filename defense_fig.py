"""Misc additional figures for defense presentation"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas
import random

from constants import SIGMA, SURFRAD, P_ATM, N_SPECIES, LI_TABLE1, LBL_LABELS, \
    BANDS_V, BANDS_L
from fraction import planck_lambda
from figures import training_data
from corr26b import fit_linear
from fig3 import ijhmt_to_tau


COLORS = {
    "cornflowerblue": "#6495ED",
    "orange": "#F37748",
    "yellow": "#FFCA3A",
    "darkyellow": "#E0A500",
    "green": "#1A603A",
    "dustypurple": "#464D77",
    "dustygreen": "#77af9c",
}


def bb_spectra():
    # electromagnetic spectrum figure
    wvl_sun = np.geomspace(0.15, 15, 100)  # micron
    bb_sun = planck_lambda(wvl_sun, t=5800)  # / (SIGMA * np.power(5670, 4))
    wvl_earth = np.geomspace(3, 100, 100)  # micron
    bb_earth = planck_lambda(wvl_earth, t=288)

    # scale down bb_sun by distance between earth and sun
    r_earth = 6378000  # m
    r_earthsun = 1.495e11  # m
    d_sun = 1.39e9  # m

    factor = np.power(d_sun, 2) / (4 * np.power(r_earthsun, 2))
    bb_sun = bb_sun * factor

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(wvl_sun / 1000000, bb_sun * wvl_sun / 4, lw=2, c=COLORS["darkyellow"])
    ax.plot(wvl_earth / 1000000, bb_earth * wvl_earth, lw=2, c=COLORS["orange"])
    ax.set_xscale("log")
    ax.set_xlabel("$\lambda$ [m]")
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(True)
    ax.set_yticks([])
    s = "Blackbody\nradiation\nat 5800 K\n(distance 1 a.u.)"
    ax.text(0.15, 0.6, s, transform=ax.transAxes, ha="right", va="center")
    s = "Blackbody\nradiation\nat 288 K"
    ax.text(0.8, 0.5, s, transform=ax.transAxes, ha="left", va="center")
    ax.set_ylim(bottom=0)
    ax.axvline(0.000004, ls="--", c="0.5")
    # ylabel = "W/m$^2$/$\mu$m"
    # ax.set_ylabel(ylabel)
    # plt.show()
    filename = os.path.join(folder, "bb_spectra.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def uc_demographic():
    # bar plot of demographic data from UC
    filename = os.path.join("data", "archive", "campus_values_230125.csv")
    df = pd.read_csv(filename)

    campus_labels = {
        "Berkeley": "Berkeley",
        "Davis": "Davis",
        "Irvine": "Irvine",
        "LosAngeles": "Los Angeles",
        "Merced": "Merced",
        "Riverside": "Riverside",
        "SanDiego": "San\nDiego",
        "SantaBarbara": "Santa\nBarbara",
        "SantaCruz": "Santa\nCruz"
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(axis="y", alpha=0.3)
    x = np.arange(df.shape[0])
    width = 0.2
    ax.bar(
        x - (width * 1.5), df.undergrads, width=width,
        color=COLORS["cornflowerblue"], label="Undergrad"
    )
    ax.bar(
        x - (width * 0.5), df.grads + df.health, width=width,
        color=COLORS["dustygreen"], label="Grad"
    )
    ax.bar(
        x + (width * 0.5), df.faculty, width=width,
        color=COLORS["orange"], label="Faculty"
    )
    ax.bar(
        x + (width * 1.5), df.staff, width=width,
        color=COLORS["dustypurple"], label="Staff"
    )
    ax.set_xticks(x)
    labels = [campus_labels[s] for s in df.campus.to_list()]
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(
        ncol=4, frameon=False, bbox_to_anchor=(1.0, 1.01), loc="lower right",
        fontsize=10
    )
    ax.set_axisbelow(True)
    filename = os.path.join(folder, "uc_campus_demographic.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def surfrad_map():
    states = geopandas.read_file('data/shp_files/usa-states-census-2014.shp')
    states = states.set_crs("epsg:4326")  # WSG84

    # states.to_crs("EPSG:2163", inplace=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    states.boundary.plot(ax=ax, color="0.4")
    for s in SURFRAD:
        lat = SURFRAD[s]["lat"]
        lon = SURFRAD[s]["lon"]
        print(lat, lon)
        if s == "BOU":
            s = "TBL"
        ax.scatter(lon, lat, c="blue", zorder=10)
        ax.text(lon + 1.5, lat, s=s, fontsize=15, va="bottom", ha="left",
                bbox=dict(facecolor="1.0", edgecolor="blue", pad=3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # plt.show()
    filename = os.path.join("figures", "defense", "map.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def fit1and2():
    # small subset to demonstrate impact of data processing
    df = training_data()

    df1 = df.loc[df.clr_num > 4000].sample(100)
    df2 = df.loc[df.clr_num < 3500].sample(100)

    ms = 15
    xmax = 35
    x = np.geomspace(0.001, xmax, 40)  # hPa

    # compare df1 and df2
    pdf = df2.copy()
    image_name = "fit2.png"

    # same plot format
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(
        pdf.pw_hpa, pdf.y, marker="o", s=ms,
        alpha=0.3, c="0.3", ec="0.5", lw=0.5, zorder=0
    )
    c1, c2 = fit_linear(pdf)
    y = c1 + c2 * np.sqrt(x * 100 / P_ATM)  # emissivity
    ax.plot(x, y, lw=1.5, ls="--", c="0.0", zorder=2)
    title = f"{c1}+{c2}"+r"$\sqrt{p_w}$"
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0.5, 1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    filename = os.path.join("figures", "defense", image_name)
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def band_visuals():
    # make broadband vs wideband vs spectral figure:
    wvl_earth = np.linspace(3, 50, 70)  # micron
    bb_earth = planck_lambda(wvl_earth, t=288)
    emissivity = 0.7

    # generate this one to keep consistent
    noise = [random.uniform(0.85, 1.15) for _ in range(len(wvl_earth))]

    band = "wide"
    for band in ["broad", "wide", "spectral"]:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(wvl_earth, bb_earth, c="0.1")
        # ax.text(0.25, 0.8, "Blackbody\nradiation", transform=ax.transAxes)
        if band == "broad":
            ax.plot(wvl_earth, bb_earth * emissivity * noise,
                    c=COLORS["cornflowerblue"])
            ax.fill_between(
                wvl_earth, 0, bb_earth * emissivity,
                fc=COLORS["cornflowerblue"], alpha=0.5, ec=None
            )
        elif band == "spectral":
            ax.stem(wvl_earth, bb_earth * emissivity * noise,
                    markerfmt=" ", linefmt=COLORS["cornflowerblue"])
        elif band == "wide":
            ax.plot(wvl_earth, bb_earth * emissivity * noise,
                    c=COLORS["cornflowerblue"])
            idx1, idx2 = (15, 35)  # index to mark bands
            ax.axvline(wvl_earth[idx1], ls="--", c="0.5")
            ax.axvline(wvl_earth[idx2], ls="--", c="0.5")
            ax.fill_between(
                wvl_earth[0:idx1 + 1], 0,
                bb_earth[0:idx1+1] * emissivity,
                fc=COLORS["cornflowerblue"], alpha=0.7,
                ec=COLORS["cornflowerblue"], hatch="//",
            )
            ax.fill_between(
                wvl_earth[idx1:idx2+1], 0,
                bb_earth[idx1:idx2+1] * emissivity,
                fc=COLORS["cornflowerblue"], alpha=0.7,
                ec=COLORS["cornflowerblue"], hatch="--"
            )
            ax.fill_between(
                wvl_earth[idx2:-1], 0,
                bb_earth[idx2:-1] * emissivity,
                fc=COLORS["cornflowerblue"], alpha=0.7,
                ec=COLORS["cornflowerblue"], hatch="\\"
            )

        ax.set_xlabel("wavelength ($\lambda$)", fontsize=12)
        ax.set_ylabel("intensity", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_xlim(wvl_earth[0], wvl_earth[-1])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        filename = os.path.join(folder, f"band_{band}.png")
        fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True)
    return None


def wideband_contribution():
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    species = list(LI_TABLE1.keys())[:-1]

    tau = ijhmt_to_tau()
    x = tau.index.to_numpy()

    fs = 10
    tick_fs = fs / 1.3

    fig, axes = plt.subplots(
        3, 7, sharex=True, figsize=(8, 4), sharey="row",
        height_ratios=[1, 4, 4]
    )
    plt.subplots_adjust(wspace=0.0)
    # plot e, t, d_opt per band
    for j in np.arange(1, 8):
        ax_t = axes[1, j - 1]  # transmissivity (middle)
        ax_d = axes[2, j - 1]  # optical depth  (bottom)

        # ax_e.set_title(f"b{j}", loc="center")
        t_ij = ijhmt_to_tau(f"lc2019_esky_ij_b{j}.csv")
        y_t = np.ones(len(x))
        y_d = np.zeros(len(x))
        for i in range(N_SPECIES):
            y = t_ij[species[i]].to_numpy()
            ax_t.fill_between(
                x, y_t, y_t * y, fc=cmaplist[i], label=LBL_LABELS[species[i]]
            )
            y_t = y_t * y

            dopt = -1 * np.log(y)
            ax_d.fill_between(
                x, y_d, y_d + dopt, fc=cmaplist[i], label=LBL_LABELS[species[i]]
            )
            y_d = y_d + dopt

        ax_t.grid(alpha=0.3)
        ax_t.set_axisbelow(True)

        ax_d.grid(alpha=0.3)
        ax_d.set_axisbelow(True)
        ax_d.set_xlabel("$p_w$ x 100")

        # set up x-axis ticks and labels
        ax_d.set_xlim(x[0], x[-1])
        xticks = [0.005, 0.010, 0.015, 0.020]
        x_labels = [f"{i * 100:.1f}" for i in xticks]
        ax_d.set_xticks(xticks, labels=x_labels, fontsize=tick_fs/1.1)

    # handle titles
    band_features = [
        "H$_2$O absorbing", "window", "CO$_2$ absorbing", "window",
        "H$_2$O absorbing", "CO$_2$ absorbing", "window"
    ]
    unit = r"cm$^{-1}$"
    unit_l = r"$\mu$m"
    for j in np.arange(1, 8):
        ax = axes[0, j-1]
        ax.tick_params(
            labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        b = f"b{j}"
        s1, s2 = BANDS_V[b]
        l1, l2 = BANDS_L[b]
        if b == "b1":
            s1 = 0
            l2 = 1000
        title = f"({b})\n{s1}-{s2} {unit}\n{l1}-{l2} {unit_l}\n" \
                f"({band_features[j-1]})"
        ax.text(
            0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes,
            fontsize=fs / 1.3
        )

    # set y-axis ticks and labels
    axes[1, 0].set_ylabel(r"$\tau_{ij}$", fontsize=fs)
    axes[1, 0].set_yticks(np.linspace(0.7, 1.1, 5))
    axes[1, 0].tick_params(axis="y", labelsize=tick_fs)

    axes[2, 0].set_ylabel(r"$\delta_{ij}$", fontsize=fs)
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 0].set_yticks(np.linspace(0, 0.4, 5))
    axes[2, 0].tick_params(axis="y", labelsize=tick_fs)

    # legend
    axes[2, -1].legend(ncol=2, fontsize=tick_fs, labelspacing=0.2)
    fig.savefig(os.path.join(folder, "wideband_contribution.png"),
                bbox_inches="tight", dpi=300)
    plt.close()
    return None


if __name__ == "__main__":
    print()
    folder = os.path.join("figures", "defense")
    if not os.path.exists(folder):
        os.makedirs(folder)