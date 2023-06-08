"""Produce figures for paper."""


import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

from constants import ELEVATIONS, SEVEN_COLORS
from corr26b import create_training_set, reduce_to_equal_pts_per_site


# set defaults
mpl.rcParams['font.family'] = "serif"
mpl.rcParams["font.serif"] = "Arial"
mpl.rcParams['mathtext.fontset'] = "cm"

FILTER_PCT_CLR = 0.05  # percent below which to drop
FILTER_NPTS_CLR = 0.20  # percentile below which to drop


COLORS = {
    "cornflowerblue": "#6495ED",
    "persianindigo": "#391463"
}


def pressure_temperature_per_site():
    # variation on t0/p0 original, without season distinction
    overlay_profile = False
    filter_ta = False
    alpha_background = 0.2 if overlay_profile else 1.0
    pm_p_mb = 20  # plus minus pressure (mb)
    ms = 15  # marker size

    if overlay_profile:
        filename = os.path.join("data", "afgl_midlatitude_summer.csv")
        af_sum = pd.read_csv(filename)
        filename = os.path.join("data", "afgl_midlatitude_winter.csv")
        af_win = pd.read_csv(filename)

    df = create_training_set(
        year=[2010, 2015, 2020], temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR, drive="server4"
    )
    df = reduce_to_equal_pts_per_site(df)

    if filter_ta:
        df = df.loc[abs(df.t_a - df.afgl_t0) <= 2].copy()

    # filter per season Winter
    # pdf = df.sample(2000, random_state=22)
    month_season = {}
    for i in range(1, 13):
        if i in [12, 1, 2]:
            month_season[i] = "winter"
        elif i in [3, 4, 5]:
            month_season[i] = "spring"
        elif i in [6, 7, 8]:
            month_season[i] = "summer"
        elif i in [9, 10, 11]:
            month_season[i] = "fall"
    df["month"] = df.index.month
    df["season"] = df["month"].map(month_season)

    pdf = df.loc[df.season == "winter"].sample(250, random_state=22)
    tmp = df.loc[df.season == "summer"].sample(250, random_state=22)
    pdf = pd.concat([pdf, tmp])
    tmp = df.loc[(df.season == "spring")].sample(125, random_state=22)
    pdf = pd.concat([pdf, tmp])
    tmp = df.loc[(df.season == "fall")].sample(125, random_state=22)
    pdf = pd.concat([pdf, tmp])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = pdf.loc[pdf.site == s]
        afgl_t = group.afgl_t0.values[0]
        afgl_p = group.afgl_p0.values[0]

        x = np.linspace(group.t_a.min(), group.t_a.max(), 2)
        ax.fill_between(
            x, afgl_p - pm_p_mb, afgl_p + pm_p_mb,
            fc=SEVEN_COLORS[i], alpha=0.1 * alpha_background, zorder=0)
        ax.axhline(afgl_p, c=SEVEN_COLORS[i], label=s, zorder=1,
                   alpha=alpha_background)
        ax.axvline(afgl_t, c=SEVEN_COLORS[i], zorder=1,
                   alpha=alpha_background)

        ax.scatter(
            group.loc[group.season == "fall"].t_a,
            group.loc[group.season == "fall"].pa_hpa, marker="o", s=ms,
            alpha=0.8 * alpha_background * 0.5,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)
        ax.scatter(
            group.loc[group.season == "spring"].t_a,
            group.loc[group.season == "spring"].pa_hpa, marker="o", s=ms,
            alpha=0.8 * alpha_background * 0.5,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)

        ax.scatter(
            group.loc[group.season == "summer"].t_a,
            group.loc[group.season == "summer"].pa_hpa, marker="^", s=ms,
            alpha=0.8 * alpha_background,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)
        ax.scatter(
            group.loc[group.season == "winter"].t_a,
            group.loc[group.season == "winter"].pa_hpa, marker="s", s=ms,
            alpha=0.8 * alpha_background,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)
        i += 1
    ymin, ymax = ax.get_ylim()
    if overlay_profile:
        alt_x = np.linspace(0, 2, 10)  # km
        y_sum_p = np.interp(alt_x, af_sum.alt_km.values, af_sum.pres_mb.values)
        x_sum_t = np.interp(alt_x, af_sum.alt_km.values, af_sum.temp_k.values)
        ax.plot(x_sum_t, y_sum_p, ls="--", c="0.3", label="AFGL\nsummer")
        y_sum_p = np.interp(alt_x, af_win.alt_km.values, af_win.pres_mb.values)
        x_sum_t = np.interp(alt_x, af_win.alt_km.values, af_win.temp_k.values)
        ax.plot(x_sum_t, y_sum_p, ls="-.", c="0.3", label="AFGL\nwinter")
        ax.set_ylim(ymin, ymax)

    ax.plot([], [], color="1.0", ls="none", label=" ")
    ax.plot([], [], marker="^", color="0.2", ls="none", label="summer")
    ax.plot([], [], marker="s", color="0.2", ls="none", label="winter")
    lgd = ax.legend(ncol=5, bbox_to_anchor=(0.5, 1.01), loc="lower center")
    for lh in lgd.legend_handles:
        lh.set_alpha(1)
    ax.set_xlabel("T$_a$ [K]")
    ax.set_ylabel("P [mb]")
    ax.invert_yaxis()
    plt.tight_layout()
    filename = os.path.join("figures", f"pressure_temperature_per_site.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    print()
    pressure_temperature_per_site()
