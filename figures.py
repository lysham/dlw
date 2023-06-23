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
    "persianindigo": "#391463"
}

C1_CONST = 0.6
C2_CONST = 1.56
C3_CONST = 0.15


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


def emissivity_vs_pw_data():
    df = create_training_set(
        year=[2010, 2011, 2012, 2014, 2015],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR, drive="server4"
    )
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
    df = create_training_set(
        year=[2010, 2011, 2012, 2014, 2015],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR, drive="server4"
    )
    df = reduce_to_equal_pts_per_site(df, min_pts=200)

    df["e"] = C1_CONST + (C2_CONST * df.x)
    df["e_corr"] = df.e + C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
    df["lw_pred"] = df.e * SIGMA * np.power(df.t_a, 4)
    df["lw_err"] = df.lw_pred - df.dw_ir
    df["lw_pred_corr"] = df.e_corr * SIGMA * np.power(df.t_a, 4)
    df["lw_err_corr"] = df.lw_pred_corr - df.dw_ir

    fig, axes = plt.subplots(7, 1, figsize=(6, 10), sharex=True)
    i = 0
    lbl = r"$c_1 + c_2 \sqrt{p_w}$"
    lbl_ = r"$c_1 + c_2 \sqrt{p_w} + c_3 (\exp{^{-z/H}} - 1)$"
    for s, alt in ELEVATIONS:
        ax = axes[i]
        ax.grid(axis="x", alpha=0.3)
        pdf = df.loc[df.site == s]
        ax.hist(pdf.lw_err, bins=20, alpha=0.3, color="0.3",
                label=lbl)
        ax.hist(pdf.lw_err_corr, bins=20, alpha=0.4,
                color=COLORS["persianindigo"], ec="0.3", label=lbl_)
        ax.set_title(f"{s}", loc="left", fontsize=12)
        ax.text(0.01, 0.93, s=f"(z = {alt:,} m)", va="top", ha="left",
                fontsize="medium", transform=ax.transAxes, color="0.2")
        i += 1
        ax.set_axisbelow(True)
        ax.set_ylim(0, 45)
    axes[-1].set_xlabel("LW error [W/m$^2$]")
    axes[0].legend(ncol=2, bbox_to_anchor=(1.0, 1.01), loc="lower right")
    ax.set_xlim(-30, 30)
    plt.tight_layout()
    filename = os.path.join("figures", f"altitude_correction.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    print()
    # pressure_temperature_per_site()
    # emissivity_vs_pw_data()
    # altitude_correction()

    # TODO solar time correction plot

    site = "GWC"
    ms = 10  # marker size
    df_ref = create_training_set(
        year=[2010 + i for i in range(12)], sites=[site],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR,
        drive="server4"
    )

    cnorm = mpl.colors.Normalize(vmin=280, vmax=310)
    cmap = mpl.cm.coolwarm

    fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(12):
        df = df_ref.loc[df_ref.index.year == 2010 + i]
        df = df.sample(n=1000)
        ax = axes[0 + i//3, 0 + i % 3]
        ax.grid(alpha=0.3)
        cb = ax.scatter(
            df.pw_hpa, df.y, c=df.t_a, cmap=cmap, norm=cnorm,
            s=ms, alpha=0.5
        )
        ax.set_title(f"{2010 + i}", loc="left", fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel("emissivity [-]")
        if i // 3 == 3:
            ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylim(0.5, 1)
    ax.set_xlim(0, 25)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(cb, cax=cbar_ax, extend="both", label="T [K]")
    cbar.solids.set(alpha=1)
    # fig.suptitle(f"{site}", fontsize=14)
    filename = os.path.join("figures", f"data_12yr_e_vs_pw_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    cnorm = mpl.colors.Normalize(vmin=1, vmax=12)
    cmap = mpl.cm.twilight
    fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(12):
        df = df_ref.loc[df_ref.index.year == 2010 + i]
        df = df.sample(n=1000)
        ax = axes[0 + i//3, 0 + i % 3]
        ax.grid(alpha=0.3)
        cb = ax.scatter(
            df.rh, df.dw_ir, c=df.index.month, cmap=cmap, norm=cnorm,
            s=ms, alpha=0.5
        )
        ax.set_title(f"{2010 + i}", loc="left", fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel("DLW [W/m$^2$]")
        if i // 3 == 3:
            ax.set_xlabel("RH [%]")
    ax.set_ylim(100, 550)
    ax.set_xlim(0, 100)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(cb, cax=cbar_ax, label="month")
    cbar.solids.set(alpha=1)
    # fig.suptitle(f"{site}", fontsize=14)
    filename = os.path.join("figures", f"data_12yr_dlw_vs_rh_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)


