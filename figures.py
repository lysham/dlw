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
    COLOR7_DICT
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
C2_CONST = 1.653
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


def create_tra_val_sets():
    # from training data, create training and validation sets
    # using train_test_split to create idx split
    df = training_data(import_full_train=True)  # equal # samples per site
    # split training_data into train and validation (1,000)
    tra_idx, val_idx = train_test_split(
        np.arange(df.shape[0]), test_size=1000, random_state=43)

    filename = os.path.join("data", "val.csv")
    df.iloc[val_idx].to_csv(filename)

    filename = os.path.join("data", "tra.csv")
    df = reduce_to_equal_pts_per_site(df.iloc[tra_idx])
    df.to_csv(filename)
    return None


def pressure_temperature_per_site():
    # variation on t0/p0 original, showing winter/summer, other
    overlay_profile = False
    filter_ta = False
    alpha_background = 0.2 if overlay_profile else 1.0
    pm_p_mb = 20  # plus minus pressure (mb)
    ms = 15  # marker size
    ec = "0.2"  # marker edge color

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
    lgd = ax.legend(ncol=5, bbox_to_anchor=(0.5, 1.01), loc="lower center")
    # for lh in lgd.legend_handles:  # running on linux
    for lh in lgd.legendHandles:  # running on Mac
        lh.set_alpha(1)
    ax.set_xlabel("T$_a$ [K]")
    ax.set_ylabel("P [mb]")
    ax.invert_yaxis()
    plt.tight_layout()
    # plt.show()
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
        lbl_s = "TBL" if s == "BOU" else s
        ax.scatter([], [], marker="o", s=3*ms, alpha=1, c=SEVEN_COLORS[i],
                   ec="0.5", lw=0.5,  label=lbl_s)  # dummy for legend
        i += 1
    xmin, xmax = (0, 40)
    x = np.geomspace(0.00001, xmax, 40)
    y = C1_CONST + C2_CONST * np.sqrt(x * 100 / P_ATM)
    # label = r"$c_1 + c_2 \sqrt{p_w}$"
    fit_label = f"${C1_CONST:.03f}+{C2_CONST:.03f}$" + "$\sqrt{p_w}$"
    ax.plot(x, y, c="0.3", lw=1.5, ls="--", label=fit_label, zorder=10)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    ax.legend(ncol=3, bbox_to_anchor=(0.99, 0.05), loc="lower right")
    filename = os.path.join("figures", f"emissivity_vs_pw_data.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def altitude_correction():
    # histogram per site of lw_err with and without altitude correction
    # dataframe should match exactly that of emissivity vs pw data plot
    df = training_data()  # import data
    df = reduce_to_equal_pts_per_site(df, min_pts=150, random_state=22)

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
    lbl = r"$\tilde{\varepsilon}_{\rm{sky,c}}(p_w)$"
    lbl_ = r"$\tilde{\varepsilon}_{\rm{sky,c}}(p_w) + c_3 (\exp{^{-z/H}} - 1)$"
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
            ax.legend(frameon=False, bbox_to_anchor=(0.5, 1.0),
                      loc="upper center")
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
            note = f"{s} (z={alt:,}m)"
            # ax.set_title(title, loc="left")
            ax.text(0.01, 0.93, s=note, va="top", ha="left",
                    fontsize="small", transform=ax.transAxes, color="0.0")
            ax.set_axisbelow(True)
            ax.set_ylim(0, 35)
            ax.set_xlim(xmin, xmax)
            j += 1
        i += 1
    axes[3, 0].set_xlabel("$L_d$ error [W/m$^2$]")
    axes[3, 1].set_xlabel("$L_d$ error [W/m$^2$]")
    filename = os.path.join("figures", f"altitude_correction.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def evaluate_sr2021(x, site_array=None):
    # return transmissivity (tau) values evaluated at lowest elevation site
    tau = np.zeros(len(x))
    pw = (x * P_ATM)  # Pa, partial pressure of water vapor
    w = 0.62198 * pw / (P_ATM - pw)
    q = w / (1 + w)  # kg/kg

    if site_array is None:
        site = "GWC"
        lat1 = SURFRAD[site]["lat"]
        lon1 = SURFRAD[site]["lon"]
        h1, spline = shakespeare(lat1, lon1)
        he_p0 = (h1 / np.cos(40.3 * np.pi / 180))
        for i in range((len(x))):
            d_opt = spline.ev(q[i], he_p0).item()
            tau[i] = np.exp(-1 * d_opt)
    else:  # not optimized
        for i in range(len(x)):
            site = site_array[i]
            lat1 = SURFRAD[site]["lat"]
            lon1 = SURFRAD[site]["lon"]
            h1, spline = shakespeare(lat1, lon1)
            he_p0 = (h1 / np.cos(40.3 * np.pi / 180))
            d_opt = spline.ev(q[i], he_p0).item()
            tau[i] = np.exp(-1 * d_opt)
    return tau


def compare(with_data=True):
    # plot comparisons of selected correlations with and without sample data
    if with_data:
        df = training_data()
        df = reduce_to_equal_pts_per_site(df, min_pts=100)
        df['y'] = df.y - df.correction  # bring all sample to sea level
        ms = 10  # marker size for data samples
        filename = os.path.join("figures", f"compare_with_data.png")
    else:
        t = 288  # standard temperature for scaling measurement error
        yerr = 5 / (SIGMA * np.power(t, 4))  # +/-5 W/m^2 error
        filename = os.path.join("figures", f"compare.png")
    t = 288  # standard temperature for scaling measurement error
    yerr = 5 / (SIGMA * np.power(t, 4))  # +/-5 W/m^2 error
    figsize = (7, 4)
    # set axis bounds of both figures
    xmin, xmax = (0.01, 35.5)  # hpa
    ymin, ymax = (0.5, 1.0)

    # define fitted correlation
    x = np.geomspace(xmin+0.00001, xmax, 100)  # hPa
    x = x * 100 / P_ATM  # normalized
    y = C1_CONST + C2_CONST * np.sqrt(x)  # emissivity

    tau = evaluate_sr2021(x)
    e_tau_p0 = 1 - tau

    fig, ax = plt.subplots(figsize=figsize)
    axins = inset_axes(ax, width="50%", height="42%", loc=4, borderpad=1.8)
    ax.set_xlim(0, 0.035)
    ax.set_ylim(ymin, ymax)
    axins.set_xlim(xmin * 100 / P_ATM, 0.01)
    axins.set_ylim(0.6, 0.8)
    ax, axins = _add_common_features(ax, axins, x, y, e_tau_p0)
    if with_data:  # with data
        ax.scatter(
            (df.pw_hpa * 100) / P_ATM, df.y, marker="o", s=ms,
            alpha=0.3, c="0.3", ec="0.5", lw=0.5, zorder=0
        )
        axins.scatter(
            (df.pw_hpa * 100) / P_ATM, df.y, marker="o", s=ms,
            alpha=0.3, c="0.3", ec="0.5", lw=0.5, zorder=0
        )
    # ax.fill_between(x, y - yerr, y + yerr, alpha=0.5, label="+/- 5 W/m$^2$")

    ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.15), loc="upper center")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def _add_common_features(ax, axins, x, y, e_tau_p0):
    # Helper function for compare.
    # #Adds select correlations, axis labels, and grid

    # dlw = y * SIGMA * np.power(t, 4)  # approximate measured dlw
    # # yerr5 = (0.05 * dlw) / (SIGMA * np.power(t, 4))  # 5% error
    y_mendoza = 1.108 * np.power(x, 0.083)
    y_brunt = 0.605 + 1.528 * np.sqrt(x)
    y_li = 0.619 + 1.665 * np.sqrt(x)
    y_berdahl = 0.564 + 1.878 * np.sqrt(x)

    # fit: -./:, lw=1, gray, change ls only
    # lbl: -, lw=1, colors
    # main_fit: -, lw=1.5, black

    fit_label = f"${C1_CONST:.03f}+{C2_CONST:.03f}$" + "$\sqrt{p_w}$"
    ax.plot(x, y, lw=2, ls="-", c="0.0", zorder=2,
            label=fit_label)
    # LBL models
    ax.plot(x, y_mendoza, lw=1, ls="-", c=COLORS["viridian"], zorder=8,
            label="$1.108p_w^{0.083}$ (MVGS2017)")
    ax.plot(x, y_li, lw=1, ls="-", c=COLORS["persianred"], zorder=8,
            label="$0.619+1.665\sqrt{p_w}$ (LC2019)")
    # empirical for comparison
    ax.plot(x, y_brunt, lw=1, ls="--", c="0.5", zorder=5,
            label="$0.605+1.528\sqrt{p_w}$ (S1965)")
    ax.plot(x, y_berdahl, c="0.5", ls=":", lw=1, zorder=5,
            label="$0.564+1.878\sqrt{p_w}$ (B1984)")
    # tau model
    ax.plot(x, e_tau_p0, lw=1, ls="-", c=COLORS["cornflowerblue"], zorder=5,
            label=r"$1-e^{-d_{\rm opt}(p_w,H_e)}$ (SR2021)")

    # inset
    axins.plot(x, y, lw=2, ls="-", c="0.0", zorder=2)
    axins.plot(x, y_mendoza, lw=1, ls="-", c=COLORS["viridian"])
    axins.plot(x, y_li, lw=1, ls="-", c=COLORS["persianred"])
    axins.plot(x, y_brunt, lw=1, ls="--", c="0.5")
    axins.plot(x, y_berdahl, c="0.5", ls=":", lw=1)
    axins.plot(x, e_tau_p0, lw=1, ls="-", c=COLORS["cornflowerblue"])
    axins.grid(alpha=0.3)
    axins.set_axisbelow(True)
    _, connects = ax.indicate_inset_zoom(axins, edgecolor="#969696")
    connects[0].set_visible(True)  # bottom left
    connects[1].set_visible(False)  # top left
    connects[2].set_visible(False)  # bottom right
    connects[3].set_visible(True)  # top right

    # misc
    ax.grid(alpha=0.3)
    ax.set_xlabel("$p_w$ [-]")
    ax.set_ylabel("emissivity [-]")
    ax.set_axisbelow(True)
    return ax, axins


def tau_lc_vs_sr():
    df = ijhmt_to_tau("fig3_esky_i.csv")  # tau, first p removed
    x = df.index.to_numpy()

    # transmissivity - plot total tau against Shakespeare
    site = "GWC"
    lat1 = SURFRAD[site]["lat"]
    lon1 = SURFRAD[site]["lon"]
    h1, spline = shakespeare(lat1, lon1)
    pw = x * P_ATM  # Pa
    w = 0.62198 * pw / (P_ATM - pw)
    q = w / (1 + w)
    p_rep = P_ATM * np.exp(-1 * SURFRAD[site]["alt"] / 8500)
    p_ratio = p_rep / P_ATM
    he = (h1 / np.cos(40.3 * np.pi / 180)) * np.power(p_ratio, 1.8)
    d_opt = spline.ev(q, he)
    tau_shp = np.exp(-1 * d_opt)

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
    return None


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
    df = pd.read_csv(os.path.join("data", "val.csv"))
    df["y_corr"] = df.y - df.correction

    x = df.x.to_numpy().reshape(-1, 1)
    y = df.y.to_numpy().reshape(-1, 1)  # non-adjusted y
    y_corr = df.y_corr.to_numpy().reshape(-1, 1)  # elevation-adjusted

    print("Fit")
    _print_metrics(y_corr, C1_CONST + C2_CONST * x)  #
    print("\nS1965, LC2019, B2021")
    _print_metrics(y, 0.605 + 1.528 * x)  # S1965
    _print_metrics(y, 0.619 + 1.665 * x)  # LC2019
    _print_metrics(y, 0.564 + 1.878 * x)  # B2021

    # Note: MVGS2017 and SR2021 take pw as input instead of sqrt(pw)
    print("\nMVGS2017, SR2021")
    x2 = np.power(x, 2)
    _print_metrics(y, 1.108 * np.power(x2, 0.083))  # MVGS2017
    tau = evaluate_sr2021(x2, site_array=df.site.to_numpy())
    e_sr2021 = 1 - tau
    _print_metrics(y, e_sr2021)
    return None


def _print_metrics(actual, model):
    rmse = np.sqrt(mean_squared_error(actual, model))
    r2 = r2_score(actual, model)
    mbe = np.nanmean((model - actual), axis=0)
    print(f"RMSE: {rmse.round(5)} | MBE: {mbe[0].round(5)} | R2: {r2.round(5)}")
    return None


def solar_time(create_csv=False):
    if create_csv:
        df = create_training_set(
            year=[2010, 2011, 2012],
            temperature=True, cs_only=True,
            filter_pct_clr=FILTER_PCT_CLR,
            filter_npts_clr=FILTER_NPTS_CLR,
            filter_solar_time=False,
            drive="server4"
        )
        df = reduce_to_equal_pts_per_site(df, min_pts=300)
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

    # boxplot by hour
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax = axes[0]
    data = []
    for s in np.arange(6, 19):
        data.append(df.loc[df.index.hour == s, "lw_err"].to_numpy())
    ax.grid(alpha=0.3)
    ax.axhline(0, c="0.7", ls="--")
    ax.boxplot(
        data, labels=np.arange(6, 19), patch_artist=True,
        boxprops={'fill': True, 'facecolor': 'white'},
        medianprops={'color': "black"},
        showfliers=False, zorder=10
    )
    ax.set_ylabel(r"$L_{d,\rm{pred}} - L_{d,\rm{meas}}$ [W/m$^2$]")
    ax.set_axisbelow(True)
    ax.set_ylim(-30, 20)

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
    ax.set_ylabel(r"$T_{a} - T_{dp}$ [K]")
    ax.set_xlabel("Solar hour of day")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 40)

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

    df = training_data(create=False, import_val=False, import_full_train=False)
    test = df.loc[df.index.year == 2013].copy()  # make test set
    test = test.sample(n=1000)
    df = df.loc[df.index.year != 2013].copy()

    sizes = np.geomspace(100, 100000, 20)
    n_iter = 100  # per sample size
    c1_vals = np.zeros((len(sizes), n_iter))
    c2_vals = np.zeros((len(sizes), n_iter))
    rmses = np.zeros((len(sizes), n_iter))
    r2s = np.zeros((len(sizes), n_iter))
    for i in range(len(sizes)):
        for j in range(n_iter):
            train = df.sample(n=int(sizes[i]))
            fit_df = pd.DataFrame(dict(x=train.x, y=train.y-train.correction))
            c1, c2 = fit_linear(fit_df, print_out=False)
            c1_vals[i, j] = c1
            c2_vals[i, j] = c2

            # evaluate on test
            de_p = C3_CONST * (np.exp(-1 * test.elev / 8500) - 1)
            pred_e = c1 + (c2 * test.x) + de_p
            # pred_y = SIGMA * np.power(test.t_a, 4) * pred_e
            # rmse = np.sqrt(mean_squared_error(
            #     test.dw_ir.to_numpy(), pred_y.to_numpy()))
            rmse = np.sqrt(mean_squared_error(
                test.y.to_numpy(), pred_e.to_numpy()))
            rmses[i, j] = rmse
            r2s[i, j] = r2_score(test.y.to_numpy(), pred_e.to_numpy())

    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    ax = axes[0]
    ax.set_xscale("log")
    ax.fill_between(
        sizes, c1_vals.min(axis=1),
        c1_vals.max(axis=1), alpha=0.5, fc=COLORS["cornflowerblue"])
    ax.plot(sizes, c1_vals.mean(axis=1), c=COLORS["cornflowerblue"])
    ax.set_ylabel("$c_1$")
    ax.set_yticks(np.linspace(0.58, 0.62, 5))
    ax.set_ylim(0.58, 0.62)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_xscale("log")
    ax.fill_between(
        sizes, c2_vals.min(axis=1), c2_vals.max(axis=1),
        alpha=0.5, fc=COLORS["cornflowerblue"])
    ax.plot(sizes, c2_vals.mean(axis=1), c=COLORS["cornflowerblue"])
    ax.set_ylabel("$c_2$")
    ax.set_yticks(np.linspace(1.4, 1.9, 6))
    ax.set_ylim(1.4, 1.9)
    ax.yaxis.set_major_formatter('{x:.02f}')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.set_xscale("log")
    ax.fill_between(
        sizes, r2s.min(axis=1), r2s.max(axis=1),
        alpha=0.5, fc=COLORS["cornflowerblue"]
    )
    ax.plot(sizes, r2s.mean(axis=1), c=COLORS["cornflowerblue"])
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R$^2$")
    ax.set_yticks(np.linspace(0.89, 0.93, 5))
    ax.set_ylim(0.89, 0.93)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(sizes[0], sizes[-1])
    plt.show()

    filename = os.path.join("figures", "convergence.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
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
    fig.colorbar(cb, cax=cbar_ax, label="RMSE")
    filename = os.path.join("figures", "error_map_fixed_c3.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    # df = training_data(create=True)
    # create_tra_val_sets()
    print()
    # create_tra_val_sets()
    # pressure_temperature_per_site()
    # emissivity_vs_pw_data()
    altitude_correction()
    # compare(with_data=True)
    # compare(with_data=False)
    # tau_lc_vs_sr()
    # print_results_table()
    # solar_time(create_csv=False)
    print()

    # # 2010 & 2015 data
    # df = create_training_set(
    #     year=[2010, 2015], temperature=False, cs_only=False,
    #     filter_pct_clr=0.0, filter_npts_clr=0.0, drive="server4"
    # )
    # df = reduce_to_equal_pts_per_site(df)  # min_pts = 200
    # df['correction'] = C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
    # filename = os.path.join("data", "specific_figure", "training_data.csv")
    # df.to_csv(filename)

    # # fdf = training_data(import_full_train=True)
    # df = training_data()
    # df["correction"] = 0.15 * (np.exp(-1 * df.elev / 8500) - 1)
    #
    # df["y"] = df.y - df.correction
    # fit_df = pd.DataFrame(dict(x=df.x.to_numpy(), y=df.y.to_numpy()))
    # fit_linear(fit_df, print_out=True)
    #
    # fit_df = pd.DataFrame(dict(x=df.x, y=df.y - 0.6))
    # fit_linear(fit_df, set_intercept=0.6, print_out=True)
    #
