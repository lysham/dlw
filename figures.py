"""Produce figures for paper."""


import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

from constants import ELEVATIONS, SEVEN_COLORS, P_ATM, SIGMA
from corr26b import create_training_set, reduce_to_equal_pts_per_site, \
    add_solar_time


# set defaults
mpl.rcParams['font.family'] = "serif"
mpl.rcParams["font.serif"] = "Arial"
mpl.rcParams['mathtext.fontset'] = "cm"

FILTER_PCT_CLR = 0.05  # percent below which to drop
FILTER_NPTS_CLR = 0.20  # percentile below which to drop


COLORS = {
    "cornflowerblue": "#6495ED",
    "persianindigo": "#391463",
    "gunmetal": "#16262E",
    "persianred": "#BC3838",
    "viridian": "#BC3838",
    "coquelicot": "#FF4000",  # very saturated
}

C1_CONST = 0.6
C2_CONST = 1.6528  # (2010-15 data, equal #/site, 5%, 20th)
C3_CONST = 0.15


def training_data(create=False):
    """Function returns the training dataset.
    Option to generate and save the dataframe if training dataset parameters
    are adapted.

    Parameters
    ----------
    create : bool, optional
        If True, create the training dataset and save the csv.

    Returns
    -------
    df : DataFrame
    """
    filename = os.path.join("data", "training_data.csv")
    if create:
        df = create_training_set(
            year=[2010, 2011, 2012, 2014, 2015],
            temperature=False, cs_only=True,
            filter_pct_clr=FILTER_PCT_CLR,
            filter_npts_clr=FILTER_NPTS_CLR, drive="server4"
        )
        df = reduce_to_equal_pts_per_site(df)  # min_pts = 200
        df['correction'] = C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename)
    return df


def pressure_temperature_per_site():
    # variation on t0/p0 original, showing winter/summer, other
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

    df = training_data()

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


def emissivity_vs_pw_data():
    df = training_data()  # import data
    df = reduce_to_equal_pts_per_site(df, min_pts=200)
    ms = 15

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = df.loc[df.site == s]
        ax.scatter(
            group.pw_hpa, group.y, marker="o", s=ms,
            alpha=0.8, c=SEVEN_COLORS[i], ec="0.5", lw=0.5, zorder=10
        )
        ax.scatter([], [], marker="o", s=3*ms, alpha=1, c=SEVEN_COLORS[i],
                   ec="0.5", lw=0.5,  label=s)  # dummy for legend
        i += 1
    xmin, xmax = (0, 40)
    x = np.geomspace(0.00001, xmax, 40)
    y = C1_CONST + C2_CONST * np.sqrt(x * 100 / P_ATM)
    label = r"$c_1 + c_2 \sqrt{p_w}$"
    ax.plot(x, y, c="0.3", lw=1.5, ls="--", label=label, zorder=10)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    ax.legend(ncol=4, bbox_to_anchor=(0.99, 0.05), loc="lower right")
    plt.tight_layout()
    filename = os.path.join("figures", f"emissivity_vs_pw_data.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def altitude_correction():
    # histogram per site of lw_err with and without altitude correction
    # dataframe should match exactly that of emissivity vs pw data plot
    df = training_data()  # import data
    df = reduce_to_equal_pts_per_site(df, min_pts=200)

    df["e"] = C1_CONST + (C2_CONST * df.x)
    df["e_corr"] = df.e + C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
    df["lw_pred"] = df.e * SIGMA * np.power(df.t_a, 4)
    df["lw_err"] = df.lw_pred - df.dw_ir
    df["lw_pred_corr"] = df.e_corr * SIGMA * np.power(df.t_a, 4)
    df["lw_err_corr"] = df.lw_pred_corr - df.dw_ir

    xmin, xmax = (-30, 30)
    bins = np.arange(xmin, xmax + 1, 2.5)

    fig, axes = plt.subplots(7, 1, figsize=(6, 10), sharex=True)
    i = 0
    lbl = r"$c_1 + c_2 \sqrt{p_w}$"
    lbl_ = r"$c_1 + c_2 \sqrt{p_w} + c_3 (\exp{^{-z/H}} - 1)$"
    for s, alt in ELEVATIONS:
        ax = axes[i]
        ax.grid(axis="x", alpha=0.3)
        pdf = df.loc[df.site == s]
        ax.hist(pdf.lw_err, bins=bins, alpha=0.3, color="0.3",
                label=lbl)
        ax.hist(pdf.lw_err_corr, bins=bins, alpha=0.4,
                color=COLORS["persianindigo"], ec="0.3", label=lbl_)
        ax.set_title(f"{s}", loc="left", fontsize=12)
        ax.text(0.01, 0.93, s=f"(z = {alt:,} m)", va="top", ha="left",
                fontsize="medium", transform=ax.transAxes, color="0.2")
        i += 1
        ax.set_axisbelow(True)
        ax.set_ylim(0, 45)
    axes[-1].set_xlabel("LW error [W/m$^2$]")
    axes[0].legend(ncol=2, bbox_to_anchor=(1.0, 1.01), loc="lower right")
    ax.set_xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()
    filename = os.path.join("figures", f"altitude_correction.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def compare():
    # plot comparisons of selected corrletions
    # WIP to express measurement uncertainty in result
    x = np.geomspace(0.00001, 35, 40)
    y = C1_CONST + C2_CONST * np.sqrt(x * 100 / P_ATM)  # emissivity
    y_lbl = 0.6376 + 1.6026 * np.sqrt(x * 100 / P_ATM)  # emissivity
    t = 288  # standard temperature for scaling measurement error
    dlw = y * SIGMA * np.power(t, 4)  # approximate measured dlw
    yerr = 5 / (SIGMA * np.power(t, 4))  # +/-5 W/m^2 error
    yerr5 = (0.05 * dlw) / (SIGMA * np.power(t, 4))  # 5% error
    y_mendoza = 0.624 * np.power(x, 0.083)
    y_brunt = 0.605 + 0.048 * np.sqrt(x)
    y_li = 0.6173 + 1.6940 * np.power(x * 100 / P_ATM, 0.5035)
    y_17 = 0.5980 + 1.8140 * np.sqrt(x * 100 / P_ATM)

    sqrt_pw = "$\sqrt{p_w}$"
    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    ax.plot(x, y, lw=1.5, ls="--", c="0.2",
            label=f"{C1_CONST}+{C2_CONST}{sqrt_pw}")
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.5)
    ax.fill_between(x, y - yerr5, y + yerr5, fc="0.7", alpha=0.4)
    ax.plot(x, y_lbl, c="0.3", ls=":", lw=2,
            label="LBL fit: $0.6376+1.6026\sqrt{p_w}$")
    ax.plot(x, y_mendoza, c=COLORS["persianred"],
            alpha=0.8, label="Mendoza 2017: $0.624P_w^{0.083}$")
    ax.plot(x, y_brunt, c=COLORS["cornflowerblue"],
            alpha=0.8, label="Brunt 1932: $0.605+0.048\sqrt{P_w}$")
    ax.plot(x, y_li, c=COLORS["persianindigo"], ls=":", lw=2,
            alpha=0.8, label="Li 2019: $0.6173+1.6940{p_w}^{0.5035}$")
    ax.plot(x, y_17, c=COLORS["coquelicot"], ls=":", lw=2,
            alpha=0.8, label="Li 2017: $0.598+1.814\sqrt{p_w}$")
    # ax.set_xscale("log")
    ax.set_xlim(0.2, 20)
    ax.set_ylim(0.5, 0.9)
    ax.legend()
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    ax.set_axisbelow(True)
    filename = os.path.join("figures", f"compare.png")  # _log
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    print()
    # pressure_temperature_per_site()
    # emissivity_vs_pw_data()
    # altitude_correction()

    # TODO solar time correction plot
    df = training_data()
    df = reduce_to_equal_pts_per_site(df, min_pts=200)
    ms = 15
    df['y'] = df.y - df.correction

    xmin, xmax = (0.2, 20)
    x = np.geomspace(xmin+0.00001, xmax, 40)
    y = C1_CONST + C2_CONST * np.sqrt(x * 100 / P_ATM)
    y_lbl = 0.6376 + 1.6026 * np.sqrt(x * 100 / P_ATM)  # emissivity
    y_mendoza = 0.624 * np.power(x, 0.083)
    y_brunt = 0.605 + 0.048 * np.sqrt(x)
    # y_brunt = 0.52 + 0.065 * np.sqrt(x)
    y_li = 0.6173 + 1.6940 * np.power(x * 100 / P_ATM, 0.5035)
    y_17 = 0.5980 + 1.8140 * np.sqrt(x * 100 / P_ATM)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = df.loc[df.site == s]
        ax.scatter(
            group.pw_hpa, group.y, marker="o", s=ms,
            alpha=0.2, c=SEVEN_COLORS[i], ec="0.5", lw=0.5, zorder=10
        )
        ax.scatter([], [], marker="o", s=3*ms, alpha=1, c=SEVEN_COLORS[i],
                   ec="0.5", lw=0.5,  label=s)  # dummy for legend
        i += 1
    label = r"$c_1 + c_2 \sqrt{p_w}$"
    ax.plot(x, y, c="0.3", lw=3, ls="--", label=label, zorder=10)
    ax.plot(x, y_lbl, c="0.2", ls="-", lw=1.5, zorder=10,
            label="LBL fit: $0.6376+1.6026\sqrt{p_w}$")
    ax.plot(x, y_mendoza, c=COLORS["persianred"], zorder=10, lw=1.5,
            alpha=0.8, label="Mendoza 2017: $0.624P_w^{0.083}$")
    ax.plot(x, y_brunt, c=COLORS["cornflowerblue"], zorder=10, lw=1.5,
            alpha=0.8, label="Brunt 1932: $0.605+0.048\sqrt{P_w}$")
    ax.plot(x, y_li, c=COLORS["persianindigo"], ls=":", lw=2, zorder=10,
            alpha=0.8, label="Li 2019: $0.6173+1.6940{p_w}^{0.5035}$")
    ax.plot(x, y_17, c=COLORS["coquelicot"], ls=":", lw=2, zorder=10,
            alpha=0.8, label="Li 2017: $0.598+1.814\sqrt{p_w}$")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    ax.legend(ncol=1, bbox_to_anchor=(1.01, 0.05), loc="lower left")
    plt.tight_layout()
    filename = os.path.join("figures", f"compare_with_data.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)


