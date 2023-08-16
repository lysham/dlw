"""Produce figures for paper."""


import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from constants import ELEVATIONS, SEVEN_COLORS, P_ATM, SIGMA, SURFRAD, \
    N_SPECIES, LI_TABLE1, LBL_LABELS, BANDS_V
from corr26b import create_training_set, reduce_to_equal_pts_per_site, \
    add_solar_time, fit_linear
from fig3 import shakespeare, ijhmt_to_tau, ijhmt_to_individual_e


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
    "viridian": "#5B9279",
    "coquelicot": "#FF4000",  # very saturated
    "barnred": "#6f1a07",
    "giantsorange": "#f46036",
}

# (2010-15 data, equal #/site, 5%, 20th, 1,000 pts set aside for validation)
C1_CONST = 0.6
C2_CONST = 1.652
C3_CONST = 0.15


def training_data(create=False, import_full_train=False, import_val=False):
    """Function returns the training dataset of which there are three options
    to return: training, tra, val. Training is the full training dataset,
    val is the 1000 sample validation, tra is the remaining training set
    reduced to equal # of samples per site. Essentially tra+val~=training.

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
    if create:
        df = create_training_set(
            year=[2010, 2011, 2012, 2013, 2014, 2015],
            temperature=False, cs_only=True,
            filter_pct_clr=FILTER_PCT_CLR,
            filter_npts_clr=FILTER_NPTS_CLR, drive="server4"
        )
        df = reduce_to_equal_pts_per_site(df)  # min_pts = 200
        df['correction'] = C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
        filename = os.path.join("data", "training_data.csv")
        df.to_csv(filename)
        create_tra_val_sets(df)
    elif import_full_train:
        filename = os.path.join("data", "training_data.csv")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    elif import_val:
        filename = os.path.join("data", "val.csv")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        filename = os.path.join("data", "tra.csv")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df


def create_tra_val_sets(df):
    # from training data, create training and validation sets
    # using train_test_split to create idx split
    # df = training_data(import_full_train=True)  # equal # samples per site
    # split training_data into train and validation (10,000)
    tra_idx, val_idx = train_test_split(
        np.arange(df.shape[0]), test_size=10000, random_state=43)

    filename = os.path.join("data", "val.csv")
    df.iloc[val_idx].to_csv(filename)

    filename = os.path.join("data", "tra.csv")
    df = reduce_to_equal_pts_per_site(df.iloc[tra_idx])
    df.to_csv(filename)
    return None


def pressure_temperature_per_site(server4=True):
    # variation on t0/p0 original, showing winter/summer, other
    overlay_profile = False
    filter_ta = False
    alpha_background = 0.2 if overlay_profile else 1.0
    pm_p_mb = 20  # plus minus pressure (mb)
    ms = 8  # marker size
    ec = "0.3"  # marker edge color
    fs = 11  # fontsize

    if overlay_profile:
        filename = os.path.join("data", "afgl_midlatitude_summer.csv")
        af_sum = pd.read_csv(filename)
        filename = os.path.join("data", "afgl_midlatitude_winter.csv")
        af_win = pd.read_csv(filename)

    df = training_data(import_full_train=True)

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

    fig, ax = plt.subplots(figsize=(5, 5.2))
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = pdf.loc[pdf.site == s]
        afgl_t = group.afgl_t0.values[0]
        afgl_p = group.afgl_p0.values[0]

        x = np.linspace(group.t_a.min(), group.t_a.max(), 2)
        ax.fill_between(
            x, group.pa_hpa.min() * np.ones(len(x)), group.pa_hpa.max()* np.ones(len(x)),
            fc=SEVEN_COLORS[i], alpha=0.1 * alpha_background, zorder=0)
        # ax.fill_between(
        #     x, afgl_p - pm_p_mb, afgl_p + pm_p_mb,
        #     fc=SEVEN_COLORS[i], alpha=0.1 * alpha_background, zorder=0)
        lbl_s = "TBL" if s == "BOU" else s
        # run once as light color outside the box and dark inside
        ax.plot(x, afgl_p * np.ones(len(x)), c=SEVEN_COLORS[i],
                label=lbl_s, zorder=1, alpha=alpha_background)
        t_line = np.linspace(group.pa_hpa.min(), group.pa_hpa.max(), 2)
        ax.plot(afgl_t * np.ones(len(t_line)), t_line, c=SEVEN_COLORS[i],
                zorder=1, alpha=alpha_background)
        ax.axhline(afgl_p, c=SEVEN_COLORS[i], zorder=0,
                   alpha=0.2 * alpha_background)
        ax.axvline(afgl_t, c=SEVEN_COLORS[i], zorder=0,
                   alpha=0.2 * alpha_background)

        ax.scatter(
            group.loc[group.season == "fall"].t_a,
            group.loc[group.season == "fall"].pa_hpa, marker="o", s=ms,
            alpha=0.8 * alpha_background * 0.5,
            c=SEVEN_COLORS[i], ec=ec, zorder=10)
        ax.scatter(
            group.loc[group.season == "spring"].t_a,
            group.loc[group.season == "spring"].pa_hpa, marker="o", s=ms,
            alpha=0.8 * alpha_background * 0.5,
            c=SEVEN_COLORS[i], ec=ec, zorder=10)

        ax.scatter(
            group.loc[group.season == "summer"].t_a,
            group.loc[group.season == "summer"].pa_hpa, marker="^", s=ms,
            alpha=0.8 * alpha_background,
            c=SEVEN_COLORS[i], ec=ec, zorder=10)
        ax.scatter(
            group.loc[group.season == "winter"].t_a,
            group.loc[group.season == "winter"].pa_hpa, marker="s", s=ms,
            alpha=0.8 * alpha_background,
            c=SEVEN_COLORS[i], ec=ec, zorder=10)
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
    lgd = ax.legend(
        ncol=5, bbox_to_anchor=(0.5, 1.01),
        labelspacing=0.25, borderpad=0.3, handlelength=1.5,
        loc="lower center", fontsize=fs
    )
    # handles = lgd.legend_handles if server4 else lgd.legendHandles
    for lh in lgd.legend_handles:
        lh.set_alpha(1)
    ax.set_xlabel("T$_a$ [K]", fontsize=fs)
    ax.set_ylabel("P$_a$ [mb]", fontsize=fs)
    ax.invert_yaxis()
    plt.tight_layout()
    # plt.show()
    filename = os.path.join("figures", f"pressure_temperature_per_site.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def altitude_correction():
    fs = 10  # fontsize

    # histogram per site of lw_err with and without altitude correction
    # dataframe should match exactly that of emissivity vs pw data plot
    df = training_data()  # import data
    df = reduce_to_equal_pts_per_site(df, min_pts=150, random_state=14)

    df["e"] = C1_CONST + (C2_CONST * df.x)
    df["e_corr"] = df.e + C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
    df["lw_pred"] = df.e * SIGMA * np.power(df.t_a, 4)
    df["lw_err"] = df.lw_pred - df.dw_ir
    df["lw_pred_corr"] = df.e_corr * SIGMA * np.power(df.t_a, 4)
    df["lw_err_corr"] = df.lw_pred_corr - df.dw_ir

    xmin, xmax = (-30, 30)
    bins = np.arange(xmin, xmax + 1, 2.5)

    fig, axes = plt.subplots(4, 2, figsize=(6, 4), sharey=True, sharex=True)
    plt.subplots_adjust(hspace=0.2, wspace=0.05)
    lbl = r"$\varepsilon_{\rm{sky,c}}(p_w)$"
    lbl_ = r"$\varepsilon_{\rm{sky,c}}(p_w) + c_3 (\exp{^{-z/H}} - 1)$"
    i = 0  # plot counter
    j = 0  # site counter
    for i in range(len(ELEVATIONS) + 1):  # hacky
        ax = axes[i // 2, i % 2]  # zig zag downward
        s, alt = ELEVATIONS[j]
        if i == 1:
            # format empty subplot at upper right, legend only
            ax.hist([], color="0.3", alpha=0.3, label=lbl)
            ax.hist([], ec="0.3", alpha=0.3, color=COLORS["persianindigo"],
                    label=lbl_)
            ax.legend(frameon=False, bbox_to_anchor=(0.5, 0.5),
                      loc="center", fontsize=fs)
            ax.tick_params(labelbottom=False, labelleft=False, bottom=False,
                           left=False)
            ax.spines.right.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.spines.bottom.set_visible(False)
        else:
            ax.grid(axis="x", alpha=0.3)
            pdf = df.loc[df.site == s]
            ax.hist(pdf.lw_err, bins=bins, alpha=0.3, color="0.3",
                    label=lbl)
            ax.hist(pdf.lw_err_corr, bins=bins, alpha=0.4,
                    color=COLORS["persianindigo"], ec="0.3", label=lbl_)
            if s == "BOU":
                s = "TBL"
            note = f"{s} ({alt:,} m)"
            # ax.set_title(title, loc="left")
            ax.text(
                0.01, 0.93, s=note, va="top", ha="left",
                fontsize=fs, transform=ax.transAxes, color="0.0"
            )
            ax.set_axisbelow(True)
            ax.set_ylim(0, 35)
            ax.set_xlim(xmin, xmax)
            j += 1
        i += 1

    xlabel = r"$L_{d,\rm{pred}} - L_{d,\rm{meas}}$ [W/m$^2$]"
    axes[3, 0].set_xlabel(xlabel, fontsize=fs)
    axes[3, 1].set_xlabel(xlabel, fontsize=fs)
    filename = os.path.join("figures", f"altitude_correction.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def evaluate_sr2021(x, site_param=None, p_param=P_ATM):
    # return transmissivity (tau) values evaluated at lowest elevation site
    tau = np.zeros(len(x))
    pw = (x * P_ATM)  # Pa, partial pressure of water vapor
    w = 0.62198 * pw / (p_param - pw)
    q = w / (1 + w)  # kg/kg

    # not optimized
    if isinstance(site_param, str):
        lat1 = SURFRAD[site_param]["lat"]
        lon1 = SURFRAD[site_param]["lon"]
        h1, spline = shakespeare(lat1, lon1)
        he_p0 = (h1 / np.cos(40.3 * np.pi / 180)) * np.power((p_param / P_ATM), 1.8)
        for i in range((len(x))):
            d_opt = spline.ev(q[i], he_p0).item()
            tau[i] = np.exp(-1 * d_opt)
    elif hasattr(site_param, "__iter__"):
        for i in range(len(x)):
            lat1 = SURFRAD[site_param[i]]["lat"]
            lon1 = SURFRAD[site_param[i]]["lon"]
            h1, spline = shakespeare(lat1, lon1)
            he_p0 = (h1 / np.cos(40.3 * np.pi / 180)) * np.power((p_param / P_ATM), 1.8)
            d_opt = spline.ev(q[i], he_p0).item()
            tau[i] = np.exp(-1 * d_opt)
    else:
        site = "GWC"
        lat1 = SURFRAD[site]["lat"]
        lon1 = SURFRAD[site]["lon"]
        h1, spline = shakespeare(lat1, lon1)
        he_p0 = (h1 / np.cos(40.3 * np.pi / 180))  * np.power((p_param / P_ATM), 1.8)
        for i in range((len(x))):
            d_opt = spline.ev(q[i], he_p0).item()
            tau[i] = np.exp(-1 * d_opt)

    return tau


def compare_combined():
    fs = 12  # fontsize

    df = training_data()
    pdf = df.copy()
    pdf = reduce_to_equal_pts_per_site(pdf, min_pts=150, random_state=14)
    df = reduce_to_equal_pts_per_site(df, min_pts=100, random_state=14)
    df['y'] = df.y - df.correction  # bring all sample to sea level
    ms = 10  # marker size for data samples
    filename = os.path.join("figures", f"compare_combined.png")

    figsize = (7.5, 10)
    # set axis bounds of both figures
    xmin, xmax = (0.01, 35.5)  # hpa
    ymin, ymax = (0.5, 1.0)

    # define fitted correlation
    x = np.geomspace(xmin+0.00001, xmax, 100)  # hPa
    x = x * 100 / P_ATM  # normalized
    y = C1_CONST + C2_CONST * np.sqrt(x)  # emissivity

    tau = evaluate_sr2021(x)
    e_tau_p0 = 1 - tau

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharey=True, sharex=True)
    ax = axes[0]
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = pdf.loc[pdf.site == s]
        ax.scatter(
            group.pw_hpa * 100 / P_ATM, group.y, marker="o", s=ms,
            alpha=0.8, c=SEVEN_COLORS[i], ec="0.5", lw=0.5, zorder=10
        )
        lbl_s = "TBL" if s == "BOU" else s
        ax.scatter([], [], marker="o", s=3*ms, alpha=1, c=SEVEN_COLORS[i],
                   ec="0.5", lw=0.5,  label=lbl_s)  # dummy for legend
        i += 1
    ax.plot(x, y, c="0.0", lw=2, ls="-", label="(proposed)", zorder=0)
    ax.set_ylabel("emissivity [-]", fontsize=fs/1.2)
    ax.legend(ncol=2, bbox_to_anchor=(0.99, 0.02), loc="lower right")
    ax.set_title("(a)", loc="left", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs/1.2)

    ax = axes[1]
    axins = inset_axes(ax, width="50%", height="42%", loc=4, borderpad=1.6)
    ax.set_xlim(0, 0.035)
    ax.set_ylim(ymin, ymax)
    axins.set_xlim(xmin * 100 / P_ATM, 0.01)
    axins.set_ylim(0.6, 0.8)
    ax, axins = _add_common_features(
        ax, axins, x, y, e_tau_p0, with_data=True, combined=True, fs=fs)
    ax.scatter(
        (df.pw_hpa * 100) / P_ATM, df.y, marker="o", s=ms,
        alpha=0.3, c="0.3", ec="0.5", lw=0.5, zorder=0
    )
    axins.scatter(
        (df.pw_hpa * 100) / P_ATM, df.y, marker="o", s=ms,
        alpha=0.3, c="0.3", ec="0.5", lw=0.5, zorder=0
    )
    ax.set_title("(b)", loc="left", fontsize=fs)

    ax = axes[2]
    axins = inset_axes(ax, width="50%", height="42%", loc=4, borderpad=1.6)
    ax.set_xlim(0, 0.035)
    ax.set_ylim(ymin, ymax)
    axins.set_xlim(xmin * 100 / P_ATM, 0.01)
    axins.set_ylim(0.6, 0.8)
    ax, axins = _add_common_features(
        ax, axins, x, y, e_tau_p0, with_data=False, combined=True, fs=fs)
    ax.set_title("(c)", loc="left", fontsize=fs)

    # make space for combined legend below
    plt.subplots_adjust(wspace=0.15)  # bottom=0.2
    ax.legend(
        ncol=3, bbox_to_anchor=(0.5, -0.005), loc="lower center",
        borderaxespad=0, bbox_transform=fig.transFigure, fontsize=fs/1.2)
    # y=-0.035 for two subplot
    fig.savefig(filename, bbox_inches="tight", dpi=600)
    return None


def _add_common_features(ax, axins, x, y, e_tau_p0, with_data=True, combined=False, fs=12):
    # Helper function for compare.
    # #Adds select correlations, axis labels, and grid

    if combined:
        tick_fs = fs / 1.2  # tick fontsize
    else:
        tick_fs = fs

    # dlw = y * SIGMA * np.power(t, 4)  # approximate measured dlw
    # # yerr5 = (0.05 * dlw) / (SIGMA * np.power(t, 4))  # 5% error
    y_mendoza = 1.108 * np.power(x, 0.083)
    y_brunt = 0.605 + 1.528 * np.sqrt(x)
    y_li = 0.617 + 1.694 * np.sqrt(x)
    y_berdahl = 0.564 + 1.878 * np.sqrt(x)

    # find valid x-values for MVGS (Mendoza, Victor..., 2017)
    mvgs_min = 0.2  # hPa
    mvgs_max = 17  # hPa
    mvgs_idx = (x >= mvgs_min * 100 / P_ATM) & (x <= mvgs_max * 100 / P_ATM)

    # fit: -./:, lw=1, gray, change ls only
    # lbl: -, lw=1, colors
    # main_fit: -, lw=1.5, black

    # t = 288  # standard temperature for scaling measurement error
    t = 260 + (40 / len(x)) * np.arange(len(x))
    y_err = 10 / (SIGMA * np.power(t, 4))  # +/-5 W/m^2 error
    if with_data:
        label = f"${C1_CONST:.03f}+{C2_CONST:.03f}$" + "$\sqrt{p_w}$ (proposed)"
        ax.plot(x, y, lw=2, ls="-", c="0.0", zorder=2, label=label)
    else:  # on no data plot, add legend entry for proposed fit
        if combined:
            label = f"${C1_CONST:.03f}+{C2_CONST:.03f}$" + "$\sqrt{p_w}$ (proposed)"
            ax.plot([], [], lw=2, ls="-", c="0.0", zorder=2, label=label)
        label = r"$\pm$10 W/m$^2$"
        if combined:
            ax.fill_between(x, y - y_err, y + y_err, fc="0.6", alpha=0.5,
                            zorder=2)
        else:
            ax.fill_between(x, y - y_err, y + y_err, fc="0.6", alpha=0.5,
                             zorder=2, label=label)

    # Brunt-type models
    ax.plot(x, y_li, lw=2, ls="-", c=COLORS["persianred"], zorder=8,
            label="$0.617+1.694\sqrt{p_w}$ (LC2019)")
    ax.plot(x, y_brunt, lw=2, ls="--", c="0.5", zorder=5,
            label="$0.605+1.528\sqrt{p_w}$ (S1965)")
    ax.plot(x, y_berdahl, c="0.5", ls=":", lw=2, zorder=5,
            label="$0.564+1.878\sqrt{p_w}$ (B1984)")
    # other forms
    # plot faded line for mendoza below valid line
    ax.plot(
        x, y_mendoza, lw=1, ls="-", c=COLORS["viridian"], alpha=0.3, zorder=3)
    ax.plot(
        x[mvgs_idx], y_mendoza[mvgs_idx], lw=2, ls="-", c=COLORS["viridian"],
        zorder=8, label="$1.108p_w^{0.083}$ (MVGS2017)")

    # tau model
    ax.plot(x, e_tau_p0, lw=2, ls="-", c=COLORS["cornflowerblue"], zorder=5,
            label=r"$1-e^{-\delta(p_w,H_e)}$ (SR2021)")

    # inset
    if with_data:
        axins.plot(x, y, lw=2, ls="-", c="0.0", zorder=2)
    else:
        axins.fill_between(
            x, y - y_err, y + y_err, fc="0.6", alpha=0.5, zorder=2)
    axins.plot(x, y_mendoza, lw=2, ls="-", c=COLORS["viridian"], alpha=0.3)
    axins.plot(x[mvgs_idx], y_mendoza[mvgs_idx], lw=1, ls="-", c=COLORS["viridian"])
    axins.plot(x, y_li, lw=2, ls="-", c=COLORS["persianred"])
    axins.plot(x, y_brunt, lw=2, ls="--", c="0.5")
    axins.plot(x, y_berdahl, c="0.5", ls=":", lw=2)
    axins.plot(x, e_tau_p0, lw=2, ls="-", c=COLORS["cornflowerblue"])
    axins.grid(alpha=0.3)
    axins.set_axisbelow(True)
    _, connects = ax.indicate_inset_zoom(axins, edgecolor="#969696")
    connects[0].set_visible(True)  # bottom left
    connects[1].set_visible(False)  # top left
    connects[2].set_visible(False)  # bottom right
    connects[3].set_visible(True)  # top right

    # misc
    ax.grid(alpha=0.3)
    if not combined or not with_data:
        ax.set_xlabel("$p_w$ [-]", fontsize=tick_fs)
    # if not combined or with_data:
    ax.set_ylabel("emissivity [-]", fontsize=tick_fs)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", labelsize=tick_fs)
    axins.tick_params(axis="both", labelsize=tick_fs)
    return ax, axins


def pw2rh(pw, t=294.2):
    # non-dimensional pw to relative humidity
    p_sat = 610.94 * np.exp(17.625*(t - 273.15)/(t - 30.11))
    rh = 100 * (pw * P_ATM) / p_sat
    return rh


def rh2pw(rh, t=294.2):
    # inverse of pw2rh()
    p_sat = 610.94 * np.exp(17.625 * (t - 273.15) / (t - 30.11))
    pw = (rh / 100) * p_sat * (1/P_ATM)
    return pw


def print_results_table():
    # print validation results
    df = training_data(import_val=True)
    df["y_corr"] = df.y - df.correction

    x = df.x.to_numpy().reshape(-1, 1)
    y = df.y.to_numpy().reshape(-1, 1)  # non-adjusted y
    y_corr = df.y_corr.to_numpy().reshape(-1, 1)  # elevation-adjusted

    print("Fit")
    _print_results_metrics(y_corr, C1_CONST + C2_CONST * x)  #
    # print("\nS1965, LC2019, B2021")
    _print_results_metrics(y, 0.605 + 1.528 * x)  # S1965
    _print_results_metrics(y, 0.617 + 1.694 * x)  # LC2019
    _print_results_metrics(y, 0.564 + 1.878 * x)  # B2021

    # Note: MVGS2017 and SR2021 take pw as input instead of sqrt(pw)
    print("\nMVGS2017, SR2021")
    x2 = np.power(x, 2)
    _print_results_metrics(y, 1.108 * np.power(x2, 0.083))  # MVGS2017
    # tau = evaluate_sr2021(x2, site_param=df.site.to_numpy())
    tau = evaluate_sr2021(x2)  # assume GWC H
    e_sr2021 = 1 - tau
    _print_results_metrics(y, e_sr2021.reshape(-1, 1))
    return None


def _print_results_metrics(actual, model):
    # Helper function for print_results_table()
    rmse = np.sqrt(mean_squared_error(actual, model))
    r2 = r2_score(actual, model)
    mbe = np.nanmean((model - actual), axis=0)
    print(f"RMSE: {rmse.round(5)} | MBE: {mbe.round(5)} | R2: {r2.round(5)}")
    return None


def solar_time(create_csv=False):
    fs = 11

    if create_csv:
        df = create_training_set(
            year=[2010, 2011, 2012],
            temperature=True, cs_only=True,
            filter_pct_clr=FILTER_PCT_CLR,
            filter_npts_clr=FILTER_NPTS_CLR,
            filter_solar_time=False,
            drive="server4"
        )
        df = df.loc[(df.site != "DRA") & (df.site != "BOU")]
        df = df.sample(1000, random_state=30)
        # df = reduce_to_equal_pts_per_site(df, min_pts=200)
        df['correction'] = C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
        df['e'] = C1_CONST + C2_CONST * df.x
        tmp = np.log(df.pw_hpa * 100 / 610.94)
        df["tdp"] = 273.15 + ((243.04 * tmp) / (17.625 - tmp))
        df["dtdp"] = df.t_a - df.tdp
        filename = os.path.join("data", "specific_figure", "solar_time.csv")
        df.to_csv(filename)
    else:
        filename = os.path.join("data", "specific_figure", "solar_time.csv")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    # df["y"] = df.y - 0.6
    # fit_df = pd.DataFrame(dict(x=df.x.to_numpy(), y=df.y.to_numpy()))
    c1, c2 = fit_linear(df)

    # pdf = df.loc[df.site == "SXF"].copy()
    df['e'] = c1 + c2 * df.x
    df["lw_pred"] = (df.e + df.correction) * SIGMA * np.power(df.t_a, 4)
    df["lw_err"] = df.lw_pred - df.dw_ir  # error
    df["time"] = df.index.hour + (df.index.minute / 60)

    # boxplot by hour
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    ax = axes[0]
    data = []
    for s in np.arange(6, 19):
        data.append(df.loc[df.index.hour == s, "lw_err"].to_numpy())
    ax.grid(alpha=0.3)
    ax.axhline(0, c="0.7", ls="-")
    ax.boxplot(
        data, labels=np.arange(6, 19), patch_artist=True,
        boxprops={'fill': True, 'facecolor': 'white', 'alpha': 0.9},
        medianprops={'color': "black"},
        showfliers=False, zorder=10
    )
    ylabel = r"$L_{d,\rm{pred}} - L_{d,\rm{meas}}$ [W/m$^2$]"
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_axisbelow(True)
    ax.set_ylim(-30, 30)
    ax.set_title("(a)", loc="left")

    ax = axes[1]
    data = []
    for s in np.arange(6, 19):
        data.append(df.loc[df.index.hour == s, "dtdp"].to_numpy())
    ax.grid(alpha=0.3)
    ax.boxplot(
        data, labels=np.arange(6, 19), patch_artist=True,
        boxprops={'fill': True, 'facecolor': 'white'},
        medianprops={'color': "black"},
        showfliers=False, zorder=10
    )
    ax.set_ylabel(r"$T_{a} - T_{\rm{dp}}$ [K]", fontsize=fs)
    ax.set_xlabel("Solar hour of day", fontsize=fs)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 30)
    ax.set_title("(b)", loc="left")

    # add data behind
    pdf = df.copy()
    pdf = pdf.loc[(pdf.time >= 6) & (pdf.time <= 18)]
    # x-axis 1st tick is 6 am
    axes[0].scatter(pdf.time - 5, pdf.lw_err, c="0.8",
                    s=10, alpha=0.3, zorder=0)
    axes[1].scatter(pdf.time - 5, pdf.dtdp, c="0.8",
                    s=10, alpha=0.3, zorder=0)

    filename = os.path.join("figures", "solar_time_boxplot.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    # DRAFT - switch boxplot to fill between with 25-75th percentile
    # hours = np.arange(6, 19)
    # rh = np.zeros((3, len(hours)))
    # dtdp = np.zeros((3, len(hours)))
    # i = 0
    # for s in np.arange(6, 19):
    #     group = df.loc[df.index.hour == s].copy()
    #     rh[0, i] = group.rh.median()
    #     rh[1, i] = group.rh.quantile(0.25)
    #     rh[2, i] = group.rh.quantile(0.75)
    #     dtdp[0, i] = group.dtdp.median()
    #     dtdp[1, i] = group.dtdp.quantile(0.25)
    #     dtdp[2, i] = group.dtdp.quantile(0.75)
    #     i += 1
    # fig, ax = plt.subplots()
    # ax.grid(alpha=0.3)
    # ax.plot(hours, rh[0, :])
    # ax.fill_between(hours, rh[1, :], rh[2, :], alpha=0.5)
    # ax2 = ax.twinx()
    # ax2.plot(hours, dtdp[0, :])
    # ax2.fill_between(hours, dtdp[1, :], dtdp[2, :], alpha=0.5)
    # plt.show()
    return None


def convergence():
    df = training_data()
    df["y"] = df.y - df.correction
    # test = df.loc[df.index.year == 2013].copy()  # make test set
    # test = test.sample(n=1000, random_state=23)
    # df = df.loc[df.index.year != 2013].copy()
    test = training_data(import_val=True)
    de_p = test.correction.to_numpy()  # altitude correction for test set

    sizes = np.geomspace(100, 100000, 20)
    n_iter = 1000  # per sample size
    c1_vals = np.zeros((len(sizes), n_iter))
    c2_vals = np.zeros((len(sizes), n_iter))
    r2s = np.zeros((len(sizes), n_iter))
    for i in range(len(sizes)):
        for j in range(n_iter):
            train = df.sample(n=int(sizes[i]), replace=False)
            fit_df = pd.DataFrame(dict(x=train.x, y=train.y))
            c1, c2 = fit_linear(fit_df, print_out=False)
            c1_vals[i, j] = c1
            c2_vals[i, j] = c2

            # evaluate on test
            pred_e = c1 + (c2 * test.x)
            r2s[i, j] = r2_score(test.y.to_numpy(), pred_e.to_numpy() + de_p)

    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    ax = axes[0]
    ax.set_xscale("log")
    ax.fill_between(
        sizes, c1_vals.min(axis=1),
        c1_vals.max(axis=1), alpha=0.3, fc=COLORS["cornflowerblue"])
    ax.fill_between(
        sizes, np.quantile(c1_vals, 0.1, axis=1),
        np.quantile(c1_vals, 0.9, axis=1),
        alpha=0.5, fc=COLORS["cornflowerblue"])
    ax.plot(sizes, c1_vals.mean(axis=1), c=COLORS["cornflowerblue"])
    ax.set_ylabel("$c_1$")
    ax.set_yticks(np.linspace(0.58, 0.63, 6))
    ax.set_ylim(0.58, 0.63)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_xscale("log")
    ax.fill_between(
        sizes, c2_vals.min(axis=1),
        c2_vals.max(axis=1), alpha=0.3, fc=COLORS["cornflowerblue"])
    ax.fill_between(
        sizes, np.quantile(c2_vals, 0.1, axis=1),
        np.quantile(c2_vals, 0.9, axis=1),
        alpha=0.5, fc=COLORS["cornflowerblue"])
    ax.plot(sizes, c2_vals.mean(axis=1), c=COLORS["cornflowerblue"])
    ax.set_ylabel("$c_2$")
    ax.set_yticks(np.linspace(1.3, 1.8, 6))
    ax.set_ylim(1.3, 1.8)
    ax.yaxis.set_major_formatter('{x:.02f}')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.set_xscale("log")
    ax.fill_between(
        sizes, r2s.min(axis=1),
        r2s.max(axis=1), alpha=0.3, fc=COLORS["cornflowerblue"])
    ax.fill_between(
        sizes, np.quantile(r2s, 0.1, axis=1),
        np.quantile(r2s, 0.9, axis=1),
        alpha=0.5, fc=COLORS["cornflowerblue"])
    ax.plot(sizes, r2s.mean(axis=1), c=COLORS["cornflowerblue"])
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R$^2$")
    ax.set_yticks(np.linspace(0.86, 0.90, 5))
    ax.set_ylim(0.86, 0.90)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(sizes[0], sizes[-1])

    filename = os.path.join("figures", "convergence.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def error_map_fixed_c3():
    # error maps for fixed c3
    df = training_data()
    df['correction'] = np.exp(-1 * df.elev / 8500) - 1
    # train, test = train_test_split(df, test_size=0.2, random_state=35)
    # test = df.sample(10000, random_state=35)
    test = df.copy()
    title_idx = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    c1_x = np.linspace(0.5, 0.7, 50)  # 100
    c2_x = np.linspace(1, 3, 50)  # 200
    c3_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    vmin, vmax = (0, 0.225)
    # cnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=True)
    ii = 0
    for c3 in c3_values:
        z = np.zeros((len(c1_x), len(c2_x)))
        for i in range(len(c1_x)):
            for j in range(len(c2_x)):
                pred_y = c1_x[i] + c2_x[j] * test.x
                correction = c3 * test.correction
                z[i, j] = np.sqrt(
                    mean_squared_error(test.y, pred_y + correction))
        xi, yi = np.unravel_index(z.argmin(), z.shape)
        print(c3, xi, yi, f"{z[xi, yi]:.5f}", f"{z.mean():.5f}")

        ax = axes[ii // 3, ii % 3]
        cb = ax.contourf(
            c2_x, c1_x, z, cmap=mpl.cm.coolwarm,
            vmin=vmin, vmax=vmax
        )
        ax.scatter(c2_x[yi], c1_x[xi], c="k", marker="^")
        # annotation (x, y, s)
        text = f"({c1_x[xi]:.3f}, {c2_x[yi]:.3f}) \nRMSE: {z.min():.4f}"
        ax.text(c2_x[yi] - 0.1, c1_x[xi] + 0.01, text)

        title = f"{title_idx[ii]} $c_3={c3}$"
        ax.set_title(title, loc="left")

        ii += 1
    axes[1, 0].set_xlabel("$c_2$")
    axes[1, 1].set_xlabel("$c_2$")
    axes[1, 2].set_xlabel("$c_2$")
    axes[0, 0].set_ylabel("$c_1$")
    axes[1, 0].set_ylabel("$c_1$")
    # ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
    # ax.xaxis.set_major_formatter('{x:.02f}')
    # ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(6))
    # ax.yaxis.set_major_formatter('{x:.02f}')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.94, 0.1, 0.05, 0.8])
    fig.colorbar(cb, cax=cbar_ax, label="RMSE [-]")
    filename = os.path.join("figures", "error_map_fixed_c3.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def data_processing_table(create_csv=False):
    # create specific files
    if create_csv:
        # Dataset reflecting true training set, without full filters
        df = create_training_set(
            year=[2010, 2012, 2015],
            temperature=False, cs_only=False,
            filter_pct_clr=0.0, filter_npts_clr=0.0, drive="server4"
        )
        # df = reduce_to_equal_pts_per_site(df)  # min_pts = 200
        df['correction'] = C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
        filename = os.path.join(
            "data", "specific_figure", "data_processing_table_base.csv")
        df.to_csv(filename)

        # training dataset with DRA 2017
        df = create_training_set(
            year=[2010, 2012, 2017],
            temperature=False, cs_only=False,
            filter_pct_clr=0.0, filter_npts_clr=0.0, drive="server4"
        )
        # df = reduce_to_equal_pts_per_site(df)  # min_pts = 200
        df['correction'] = C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
        filename = os.path.join(
            "data", "specific_figure", "data_processing_table_2017.csv")
        df.to_csv(filename)

    # import test/validation set and import baseline total set
    test = training_data(import_val=True)  # evaluate on validation set
    test = test.loc[test.index.year == 2013].sample(1000, random_state=23)

    filename = os.path.join(
        "data", "specific_figure", "data_processing_table_base.csv")
    # # import _2017 file to include DRA 2017 data in training set
    # filename = os.path.join(
    #     "data", "specific_figure", "data_processing_table_2017.csv")

    # training data set has not had filters applied
    base = pd.read_csv(filename, index_col=0, parse_dates=True)

    # change one of these at a time and evaluate
    clear_sky = "both"  # ["both", "li", "reno"]
    elev_correction = True
    # switch table to table_2017
    solar_time_filter = True
    apply_pct_clr = True  # percent clear
    apply_npts_clr = True  # number of samples
    remove_site_bias = False
    # if "both" and all true for above, then baseline is evaluated

    if apply_pct_clr:
        base = base.loc[base.clr_pct >= FILTER_PCT_CLR]
    if apply_npts_clr:
        base = base.loc[base.clr_num >= FILTER_NPTS_CLR]
    if clear_sky == "both":
        base = base.loc[base.csv2]
    elif clear_sky == "li":
        base = base.loc[base.cs_period]
    elif clear_sky == "reno":
        base = base.loc[base.reno_cs]
    if solar_time_filter:
        base = base.loc[base.index.hour >= 8]
    # use an equal number of points, apply correction
    if remove_site_bias:
        df = reduce_to_equal_pts_per_site(base, min_pts=10000, random_state=23)
    else:  # use straightforward sample
        df = base.sample(70000, random_state=23)  # 7x test size
    if elev_correction:
        df["y"] = df.y - df.correction

    fit_df = pd.DataFrame(dict(x=df.x.to_numpy(), y=df.y.to_numpy()))
    c1, c2 = fit_linear(fit_df, print_out=True)

    # print(df.loc[(df.site == "DRA") & (df.index.year == 2017)].shape[0])
    # print(df.shape)
    # evaluate on test set
    print("\nperformance on validation set:")
    model = c1 + c2 * test.x.to_numpy()
    if elev_correction:
        model = model + test.correction.to_numpy()
    actual = test.y.to_numpy()
    _print_results_metrics(actual, model)
    print("npts=", len(actual))
    return None


def clear_sky_filter(create_csv=False):
    # show clear sky filter difference
    s = "GWC"
    filename = os.path.join(
        "data", "specific_figure", f"{s.lower()}_clearness.csv")
    if create_csv:
        df = create_training_set(
            year=[2010, 2011, 2012, 2013, 2014, 2015], sites=s,
            filter_pct_clr=0.0, filter_npts_clr=0.0, filter_solar_time=False,
            temperature=False, cs_only=False, drive="server4")
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        df = df.loc[df.index.year < 2012]  # could increase to more samples
    df["csv2"] = df.csv2.astype("bool")
    x = df.resample("D")["csv2"].mean()
    y = df.resample("D")["csv2"].count()

    tmp_clr = df["csv2"].resample("D").count()
    thresh = np.quantile(
        tmp_clr.loc[tmp_clr > 0].to_numpy(), 0.2
    )

    toss_date = dt.date(2010, 10, 19)
    keep_date = dt.date(2010, 6, 20)
    # highlight examples used
    toss_x = x.loc[x.index.date == toss_date].item()
    toss_y = y.loc[y.index.date == toss_date].item()
    keep_x = x.loc[x.index.date == keep_date].item()
    keep_y = y.loc[y.index.date == keep_date].item()

    # print out # and % for both examples
    fig = plt.figure(figsize=(8.5, 3.5), layout="constrained")
    subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 2])
    # scatter plot
    ax = subfigs[0].subplots()
    ax.set_aspect(1/900)  # unit in y is *aspect* times displayed unit in x
    ax.grid(True, alpha=0.7)
    ax.scatter(x, y, marker=".", alpha=0.5, c="0.5")
    ax.scatter(toss_x, toss_y, marker="o", alpha=0.9, c=COLORS["persianred"])
    ax.scatter(keep_x, keep_y, marker="o", alpha=0.9, c=COLORS["persianred"])
    ax.axvline(
        0.05, ls="--", c=COLORS["persianred"],
        label="Daily percent threshold (5%)"
    )
    ax.axhline(
        thresh, c=COLORS["persianred"], label="Daily sample threshold (20th)"
    )
    ax.set_title("(a)", loc="left")
    # ax.set_title(f"{s}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Daily clear sky fraction")
    ax.set_ylabel("Daily clear sky samples")
    ax.set_ylim(0, 900)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=1, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))

    ax1, ax2 = subfigs[1].subplots(1, 2, sharey=True)
    # tossed example
    keep_cols = ["GHI_m", "DNI_m", "GHI_c", "DNI_c", "cs_period", "reno_cs"]
    pdf = df.loc[df.index.date == toss_date][keep_cols].copy()
    pdf = pdf.resample("60S", label="right", closed="right").median()
    ax1 = _clear_sky_filter(ax1, pdf, toss_date)
    ax1.set_xlabel("Solar hour of day")
    ax1.legend(
        frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        borderpad=0.3, labelspacing=0.3, columnspacing=0.8
    )
    title = "(b) " + toss_date.strftime("%Y-%m-%d") + " (exclude)"
    ax1.set_title(title, loc="left")

    # kept example
    keep_cols = ["GHI_m", "DNI_m", "GHI_c", "DNI_c", "cs_period", "reno_cs"]
    pdf = df.loc[df.index.date == keep_date][keep_cols].copy()
    pdf = pdf.resample("60S", label="right", closed="right").median()
    ax2 = _clear_sky_filter(ax2, pdf, keep_date)
    ax2.set_xlabel("Solar hour of day")
    title = "(c) " + keep_date.strftime("%Y-%m-%d") + " (include)"
    ax2.set_title(title, loc="left")
    filename = os.path.join("figures", "clear_sky_filter.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def _clear_sky_filter(ax, pdf, plot_date):
    # pdf["time"] = pdf.index.hour + (pdf.index.minute / 60)
    # pdf.set_index("time", inplace=True)
    ax.plot(pdf.index, pdf.GHI_m, c=COLORS["persianindigo"], label="GHI")
    ax.plot(pdf.index, pdf.DNI_m, c=COLORS["viridian"], label="DNI")
    ax.plot(pdf.index, pdf.GHI_c, c=COLORS["persianindigo"], ls="--", label="GHI$_c$")
    ax.plot(pdf.index, pdf.DNI_c, c=COLORS["viridian"], ls="--", label="DNI$_c$")
    ax.fill_between(
        pdf.index, 0, pdf.GHI_m, where=pdf.cs_period, fc="0.7", alpha=0.4,
        label="CS1"
    )
    ax.fill_between(
        pdf.index, 0, pdf.GHI_m, where=pdf.reno_cs, fc="0.9", alpha=0.4,
        hatch="//", label="CS2", ec="0.3"
    )
    # ax.xaxis.set_major_locator(mpl.dates.HourLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H"))
    ax.set_xlim(pdf.index[0], pdf.index[-1])
    ax.set_ylim(0, 1200)
    start = dt.datetime(plot_date.year, plot_date.month, plot_date.day, 5, 0)
    end = dt.datetime(plot_date.year, plot_date.month, plot_date.day, 19, 0)
    ax.set_xlim(start, end)
    return ax


def broadband_contribution():
    # emissivity and transmissivity with RH reference axes
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    species = list(LI_TABLE1.keys())[:-1]

    tau = ijhmt_to_tau()  # use default file (lc2019)
    eps = ijhmt_to_individual_e()
    x = tau.index.to_numpy()

    fs = 12  # fontsize
    tick_fs = fs / 1.3  # tick fontsize

    # set margins, size of figure
    fig_x0 = 0.05
    fig_y0 = 0.2
    fig_width = 0.265
    fig_height = 0.8
    wspace = 1 - (3 * fig_width) - (3 * fig_x0)
    if wspace < 0:
        print("warning: overlapping figures")

    fig = plt.figure(figsize=(7, 2.8))
    ax0 = fig.add_axes((fig_x0, fig_y0, fig_width, fig_height))
    ax1 = fig.add_axes(
        (fig_x0 + fig_width + wspace, fig_y0, fig_width, fig_height))
    ax2 = fig.add_axes(
        (fig_x0 + 2*fig_width + 2*wspace, fig_y0, fig_width, fig_height)
    )
    j = 0
    y_e = np.zeros(len(x))  # emissivity
    y_t = np.ones(len(x))  # transmissivity
    y_d = np.zeros(len(x))  # optical depth
    for gas in species:
        y = eps[gas].to_numpy()
        ax0.fill_between(x, y_e, y_e + y, label=LBL_LABELS[gas], fc=cmaplist[j])
        y_e = y_e + y
        y = tau[gas].to_numpy()  # tau
        ax1.fill_between(x, y_t, y_t * y, label=LBL_LABELS[gas], fc=cmaplist[j])
        y_t = y_t * y
        y = -1 * np.log(y)  # d_opt
        ax2.fill_between(x, y_d, y_d + y, label=LBL_LABELS[gas], fc=cmaplist[j])
        y_d = y_d + y
        j += 1
    # set axis limits, labels, grid
    i = 0
    titles = [r"(a) $\varepsilon_{i}$", r"(b) $\tau_{i}$",
              r"(c) $\delta_{i}$"]
    for ax in [ax0, ax1, ax2]:
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("$p_w$ x 100", fontsize=tick_fs)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(titles[i], loc="left", fontsize=fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

        # set up x-axis ticks and labels
        ax.set_xlim(x[0], x[-1])
        xticks = [0.005, 0.010, 0.015, 0.020, 0.025]
        x_labels = [f"{i * 100:.1f}" for i in xticks]
        ax.set_xticks(xticks, labels=x_labels, fontsize=tick_fs)

        i += 1
    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 2.2)
    ax1.legend(ncol=1, loc="upper right", fontsize=tick_fs, labelspacing=0.2)

    # add secondary axes for relative humidity reference
    rh_ticks = np.array([0, 20, 40, 60, 80, 100])

    t = 288.0  # axis below first plot
    ax3 = fig.add_axes((fig_x0, 0.03, fig_width, 0.0))
    ax3.yaxis.set_visible(False)  # hide the yaxis
    equiv_pw = np.array([rh2pw(i, t) for i in rh_ticks])
    idx = (equiv_pw > x[0]) & (equiv_pw < x[-1])
    ax3.set_xlim(x[0], x[-1])
    ax3.set_xlim(pw2rh(x[0], t), pw2rh(x[-1], t))
    ax3.set_xticks(rh_ticks[idx], labels=rh_ticks[idx], fontsize=tick_fs)
    ax3.set_xlabel(f"RH [%] at {t} K", fontsize=tick_fs)

    t = 294.2  # axis below first plot
    ax4 = fig.add_axes((fig_x0 + fig_width + wspace, 0.03, fig_width, 0.0))
    ax4.yaxis.set_visible(False)  # hide the yaxis
    equiv_pw = np.array([rh2pw(i, t) for i in rh_ticks])
    idx = (equiv_pw > x[0]) & (equiv_pw < x[-1])
    ax4.set_xlim(x[0], x[-1])
    ax4.set_xlim(pw2rh(x[0], t), pw2rh(x[-1], t))
    ax4.set_xticks(rh_ticks[idx], labels=rh_ticks[idx], fontsize=tick_fs)
    ax4.set_xlabel(f"RH [%] at {t} K", fontsize=tick_fs)

    t = 300.0  # axis below first plot
    ax5 = fig.add_axes((fig_x0 + 2 * fig_width + 2 * wspace, 0.03, fig_width, 0.0))
    ax5.yaxis.set_visible(False)  # hide the yaxis
    equiv_pw = np.array([rh2pw(i, t) for i in rh_ticks])
    idx = (equiv_pw > x[0]) & (equiv_pw < x[-1])
    ax5.set_xlim(x[0], x[-1])
    ax5.set_xlim(pw2rh(x[0], t), pw2rh(x[-1], t))
    ax5.set_xticks(rh_ticks[idx], labels=rh_ticks[idx], fontsize=tick_fs)
    ax5.set_xlabel(f"RH [%] at {t} K", fontsize=tick_fs)

    filename = os.path.join("figures", "broadband_contribution.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def spectral_band_contribution():
    # create plot of spectral band contributions
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    species = list(LI_TABLE1.keys())[:-1]

    tau = ijhmt_to_tau()
    x = tau.index.to_numpy()

    fs = 10
    tick_fs = fs / 1.3

    fig, axes = plt.subplots(
        4, 7, sharex=True, figsize=(8, 5), sharey="row",
        height_ratios=[1, 4, 4, 4]
    )
    plt.subplots_adjust(wspace=0.0)
    # plot e, t, d_opt per band
    for j in np.arange(1, 8):
        ax_e = axes[1, j - 1]  # emissivity (top)
        ax_t = axes[2, j - 1]  # transmissivity (middle)
        ax_d = axes[3, j - 1]  # optical depth  (bottom)

        # ax_e.set_title(f"b{j}", loc="center")
        e_ij = ijhmt_to_individual_e(f"lc2019_esky_ij_b{j}.csv")
        t_ij = ijhmt_to_tau(f"lc2019_esky_ij_b{j}.csv")
        y_e = np.zeros(len(x))
        y_t = np.ones(len(x))
        y_d = np.zeros(len(x))
        for i in range(N_SPECIES):
            y = e_ij[species[i]].to_numpy()
            ax_e.fill_between(
                x, y_e, y_e + y, fc=cmaplist[i], label=LBL_LABELS[species[i]]
            )
            y_e += y

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

        ax_e.grid(alpha=0.3)
        ax_e.set_axisbelow(True)

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
        if b == "b1":
            s1 = 0
        title = f"({b})\n{s1}-{s2}{unit}\n({band_features[j-1]})"
        ax.text(
            0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes,
            fontsize=fs / 1.3
        )

    # set y-axis ticks and labels
    axes[1, 0].set_ylabel(r"$\varepsilon_{ij}$", fontsize=fs)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].set_yticks(np.linspace(0, 0.4, 5))
    axes[1, 0].tick_params(axis="y", labelsize=tick_fs)

    axes[2, 0].set_ylabel(r"$\tau_{ij}$", fontsize=fs)
    axes[2, 0].set_yticks(np.linspace(0.7, 1.1, 5))
    axes[2, 0].tick_params(axis="y", labelsize=tick_fs)

    axes[3, 0].set_ylabel(r"$\delta_{ij}$", fontsize=fs)
    axes[3, 0].set_ylim(bottom=0)
    axes[3, 0].set_yticks(np.linspace(0, 0.4, 5))
    axes[3, 0].tick_params(axis="y", labelsize=tick_fs)

    # legend
    axes[1, -1].legend(ncol=2, fontsize=tick_fs, labelspacing=0.2)
    # plt.show()
    filename = os.path.join("figures", f"spectral_band_contribution.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def print_spectral_dopt_coefs():
    # print coefficients per band using spectral model
    df = ijhmt_to_tau("lc2019_esky_i.csv")
    x = df.index.to_numpy()

    species = list(LI_TABLE1.keys())[:-1]
    for j in np.arange(1, 8):
        y_ref = np.zeros(len(x))
        filename = f"lc2019_esky_ij_b{j}.csv"
        df = ijhmt_to_tau(filename)
        # define whether to fit to sqrt(pw), tanh(pw), or pw
        if j == 2:
            x_input = np.tanh(270 * x)
        elif j == 3:
            x_input = np.sqrt(x)
        else:
            x_input = x
        print("")
        for s in species:
            y = df[s].to_numpy()
            dopt = -1 * np.log(y)
            y_ref = y_ref + dopt  # cumulative product in the band
            if np.std(dopt) < 0.001:  # make constant
                print(f"b{j}", s, f"c1={np.mean(dopt).round(3)}")
            else:
                # note whether x=x or x=sqrt(x), y=y or y=-1*np.log(y)
                print(f"b{j}", s)
                fit_df = pd.DataFrame(dict(x=x_input, y=dopt))
                fit_linear(fit_df, print_out=True)
        print(f"b{j}", "TOTAL")
        if np.std(y_ref) < 0.0001:
            print(f"c1={np.mean(y_ref).round(3)}")
        else:
            fit_df = pd.DataFrame(dict(x=x_input, y=y_ref))
            fit_linear(fit_df, print_out=True)
    return None


def print_broadband_dopt_coefs():
    df = ijhmt_to_tau("lc2019_esky_i.csv")
    x = df.index.to_numpy()

    species = list(LI_TABLE1.keys())
    for s in species:
        y = df[s].to_numpy()
        dopt = -1 * np.log(y)
        if np.std(dopt) < 0.001:  # make constant
            print(s, f"c1={np.mean(dopt).round(3)}")
        else:
            # note whether x=x or x=sqrt(x), y=y or y=-1*np.log(y)
            print(s)
            fit_df = pd.DataFrame(dict(x=x, y=dopt))
            fit_linear(fit_df, print_out=True)
        print()
    return None


def tau_lc_vs_sr():
    df = ijhmt_to_tau("lc2019_esky_i.csv")  # tau, first p removed
    x = df.index.to_numpy()

    # transmissivity - plot total tau against Shakespeare
    tau_shp = evaluate_sr2021(x)

    y_fit = C1_CONST + C2_CONST * np.sqrt(x)
    y_fit = 1 - y_fit

    fig, ax = plt.subplots(figsize=(5.25, 3))
    ax.plot(x, df.total.to_numpy(), c=COLORS["persianred"], ls="-",
            label="LC2019", zorder=2)
    ax.plot(
        x, df.H2O.to_numpy() * df.CO2.to_numpy(), c=COLORS["persianred"],
        ls="--", label="LC2019 H$_2$O and CO$_2$", zorder=4
    )
    ax.plot(x, tau_shp, c=COLORS["cornflowerblue"],
            label="SR2021", zorder=5)
    fit_label = f"${C1_CONST:.03f}+{C2_CONST:.03f}$" + "$\sqrt{p_w}$"
    ax.plot(x, y_fit, lw=2, ls="-", c="0.0", zorder=0,
            label=fit_label)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 0.5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("$p_w$ [-]")
    ax.set_ylabel("transmissivity [-]")
    ax2 = ax.secondary_xaxis("top", functions=(pw2rh, rh2pw))
    ax2.set_xlabel("RH [%] at 294.2 K")

    ax.legend(ncol=2, bbox_to_anchor=(0.5, -0.2), loc="upper center")
    filename = os.path.join("figures", "tau_lc_vs_sr.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    # essentially same figure, now in d_opt
    fig, ax = plt.subplots(figsize=(5.25, 3))
    ax.plot(x, -1 * np.log(df.total.to_numpy()), c=COLORS["persianred"], ls="-",
            label="LC2019", zorder=2)
    y = df.H2O.to_numpy() * df.CO2.to_numpy()
    ax.plot(
        x, -1 * np.log(y), c=COLORS["persianred"],
        ls="--", label="LC2019 H$_2$O and CO$_2$", zorder=4
    )
    ax.plot(x, -1 * np.log(tau_shp), c=COLORS["cornflowerblue"],
            label="SR2021", zorder=5)
    fit_label = f"${C1_CONST:.03f}+{C2_CONST:.03f}$" + "$\sqrt{p_w}$"
    ax.plot(x, -1 * np.log(y_fit), lw=2, ls="-", c="0.0", zorder=0,
            label="MC2023")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.8, 2.2)
    ax.grid(alpha=0.3)
    ax.set_xlabel("$p_w$ [-]")
    ax.set_ylabel("optical depth [-]")
    ax2 = ax.secondary_xaxis("top", functions=(pw2rh, rh2pw))
    ax2.set_xlabel("RH [%] at 294.2 K")

    ax.legend(ncol=2, bbox_to_anchor=(0.5, -0.2), loc="upper center")
    filename = os.path.join("figures", "dopt_lc_vs_sr.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    # df = training_data(create=True)
    print()
    # solar_time(create_csv=False)  # boxplot
    # clear_sky_filter(create_csv=False)
    # pressure_temperature_per_site(server4=False)
    # altitude_correction()
    compare_combined()
    # print_results_table()
    # convergence()
    # error_map_fixed_c3()
    # data_processing_table(create_csv=False)
    # tau_lc_vs_sr()
    # broadband_contribution()
    # spectral_band_contribution()
    # print_spectral_dopt_coefs()
    # print_broadband_dopt_coefs()
    print()

    # ff = pd.DataFrame(dict(x=x, y=y))
    # ff.loc[(ff.x >0.5) & (ff.y < 200)]

    # filename = os.path.join("data", "afgl_midlatitude_summer.csv")
    # af_sum = pd.read_csv(filename)
    #
    # df = ijhmt_to_tau("lc2019_esky_i.csv")  # tau, first p removed
    # x = df.index.to_numpy()
    # # # transmissivity - plot total tau against Shakespeare
    # site = "GWC"
    # site_elev = SURFRAD[site]["alt"] / 1000  # km
    # p = np.interp(site_elev, af_sum.alt_km.values, af_sum.pres_mb.values)  # mb
    # tau_shp = evaluate_sr2021(x, site_param=site, p_param=p * 100)
    #
    # site = "BOU"
    # site_elev = SURFRAD[site]["alt"] / 1000  # km
    # p = np.interp(site_elev, af_sum.alt_km.values, af_sum.pres_mb.values)  # mb
    # tau_shp_bou = evaluate_sr2021(x, site_param=site, p_param=p * 100)
    #
    # y_fit = C1_CONST + C2_CONST * np.sqrt(x)
    # y_fit = 1 - y_fit
    # y_corr = C3_CONST * (np.exp(site_elev / 8.5) - 1)
    #
    # # essentially same figure, now in d_opt
    # fig, ax = plt.subplots(figsize=(5.25, 4))
    # ax.plot(x, -1 * np.log(df.total.to_numpy()), c=COLORS["persianred"], ls="-",
    #         label="LC2019", zorder=2)
    # ax.plot(x, -1 * np.log(tau_shp), c=COLORS["cornflowerblue"],
    #         label="SR2021 (GWC)", zorder=5)
    # ax.plot(x, -1 * np.log(tau_shp_bou), ls="--", c=COLORS["cornflowerblue"],
    #         label="SR2021 (BOU)", zorder=5)
    # fit_label = f"${C1_CONST:.03f}+{C2_CONST:.03f}$" + "$\sqrt{p_w}$"
    # ax.plot(x, -1 * np.log(y_fit), lw=2, ls="-", c="0.0", zorder=0,
    #         label="sea-level")
    # ax.plot(x, -1 * np.log(y_fit + y_corr), lw=2, ls="--", c="0.0", zorder=0,
    #         label="BOU")
    # ax.set_xlim(x[0], x[-1])
    # ax.set_ylim(0.8, 2.2)
    # ax.grid(alpha=0.3)
    # ax.set_xlabel("$p_w$ [-]")
    # ax.set_ylabel("optical depth [-]")
    # ax2 = ax.secondary_xaxis("top", functions=(pw2rh, rh2pw))
    # ax2.set_xlabel("RH [%] at 294.2 K")
    #
    # ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.2), loc="upper center")
    # plt.tight_layout()
    # plt.show()
    # filename = os.path.join("figures", "dopt_lc_vs_sr_tmp.png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)
    #
    # eps = ijhmt_to_individual_e()
    # x = eps.index.to_numpy()
    # y = eps.total.to_numpy()
    # y_fit = C1_CONST + C2_CONST * np.sqrt(x)
    # error = y - y_fit
    # print("average bias", error.mean().round(4))