"""Recreate Fig 3 in Li and Coimbra 2019"""

import os
import scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from main import get_pw
from corr26b import shakespeare, import_cs_compare_csv, fit_linear, \
    compute_mbe, three_c_fit, add_afgl_t0_p0, add_solar_time, \
    shakespeare_comparison, create_training_set, reduce_to_equal_pts_per_site
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from scipy.stats import pearsonr
from scipy import integrate

from fraction import planck_lambda
from constants import LI_TABLE1, P_ATM, SIGMA, N_BANDS, N_SPECIES, SURFRAD, \
    ELEVATIONS, SURF_SITE_CODES, BANDS_L


def get_pw_norm(t, rh):
    # get dimensionless pw
    pw = get_pw(t, rh)  # Pa
    return pw / P_ATM


def get_atm_p(elev):
    # get barometric pressure (elev given in [m])
    return P_ATM * np.exp(-1 * elev / 8500)


def pw2rh(t, pw):
    """Provides inverse of get_pw() i.e. eq (5) in 2017 paper

    Parameters
    ----------
    t : float
        Temperature in K
    pw : float
        Partial pressure of water vapor in Pa

    Returns
    -------
    rh : float
        Returns relative humidity [%]
    """
    exp_term = 17.625 * (t - 273.15) / (t - 30.11)
    rh = ((pw / 610.94) / np.exp(exp_term)) * 100  # %
    return rh


def get_emissivity_ij(t_a, pw_norm):
    """
    Create matrix of emissivity per species (i) and per band (j)
    i.e. species on rows and bands on columns
    
    Copied from main.py in randall repo, with some adjustments

    Parameters
    ----------
    t_a : float
        Screening level temperature.
    pw_norm : float
        Normalized partial pressure of water vapor (P_w / P0)

    Returns
    -------
    emissivity : array
        Matrix of emissivities.
    """
    # import band coefficients
    filename = os.path.join("data", "band_coef.csv")
    df = pd.read_csv(filename, dtype={"O2": float, "N2": float})

    cc = list(LI_TABLE1.keys())[:-1]  # contributing components
    # p_w = get_pw_norm(t_a, rh)
    # print(p_w, pw_norm)
    emissivity = np.zeros((N_SPECIES, N_BANDS))
    for j in range(N_BANDS):  # for each band j
        band = f"B{j + 1}"
        for i in range(N_SPECIES):  # for each species i
            species = cc[i]
            c1, c2, c3 = df.loc[df.band == band, species].values
            if band == "B2":
                e_i = c1 + (c2 * np.tanh(c3 * pw_norm))
            else:
                e_i = c1 + (c2 * np.power(pw_norm, c3))
            emissivity[i, j] = e_i
    return emissivity


def get_emissivity_i(p_w, sp="H2O"):
    c1, c2, c3 = LI_TABLE1[sp]
    e = c1 + (c2 * np.power(p_w, c3))
    return e


def plot_fig3_shakespeare_comparison():
    species = list(LI_TABLE1.keys())[:-1]

    pw_x = np.geomspace(0.1, 2.3, 20)
    e_broad = np.zeros((len(pw_x), N_SPECIES))

    e_tau = np.zeros(len(pw_x))
    e_tau_p0 = np.zeros(len(pw_x))
    site = "GWC"
    lat1 = SURFRAD[site]["lat"]
    lon1 = SURFRAD[site]["lon"]
    h1, spline = shakespeare(lat1, lon1)
    pa = 900e2  # Pa
    pw = (pw_x / 100) * P_ATM  # Pa, partial pressure of water vapor
    w = 0.62198 * pw / (pa - pw)
    q = w / (1 + w)  # kg/kg
    p_ratio = pa / P_ATM
    he = (h1 / np.cos(40.3 * np.pi / 180)) * (p_ratio ** 1.8)
    he_p0 = (h1 / np.cos(40.3 * np.pi / 180))
    for i in range((len(pw_x))):
        tau = spline.ev(q[i], he).item()
        e_tau[i] = 1 - np.exp(-1 * tau)
        tau = spline.ev(q[i], he_p0).item()
        e_tau_p0[i] = 1 - np.exp(-1 * tau)

    # # use Table 2 per band per species
    # pw_norm = (pw_x / 100)  # pw/p0
    # for i in range(len(pw_x)):
    #     e_broad[i, :] = get_emissivity_ij(t_a, pw_norm[i]).sum(axis=1)

    # use Table 1 broadband per species
    pw_norm = (pw_x / 100)  # pw/p0
    e_ttl = 0.6173 + (1.6940 * np.power(pw_norm, 0.5035))  # total
    e_ref8 = 0.598 + (1.814 * np.power(pw_norm, 0.5))  # daytime clr (22a)
    for i in range(len(species)):
        e_broad[:, i] = get_emissivity_i(pw_norm, sp=species[i])

    # special function
    tau = 1.1 + (41 * pw_norm)
    e_tau41 = 1 - np.exp(-1 * tau)

    # FIGURE
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    ax.grid(axis="y", alpha=0.3)
    prev_y = 0
    for i in range(N_SPECIES):
        ax.fill_between(pw_x, prev_y, prev_y + e_broad[:, i],
                        label=species[i], fc=cmaplist[i])
        prev_y = prev_y + e_broad[:, i]
    # ax.plot(pw_x, e_ttl, ls="-", c="teal", label="total")
    # ax.plot(pw_x, e_ref8, ls="--", c="teal", label="ref8")
    ax.plot(pw_x, e_tau, "k", label=r"(1-e$^{- \tau}$), P=P0")
    ax.plot(pw_x, e_tau_p0, "k--", label=r"(1-e$^{- \tau}$), P=900hPa")
    # ax.plot(pw_x, e_tau41, "b-", label=r"$\tau$=1.1+41p$_w$")
    ax.set_xlim(pw_x[0], pw_x[-1])
    # ax.set_ylim(0, 1)
    ax.set_ylim(0.5, 0.9)
    # ax.set_title(r"height scale from BOU")
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel("pw x 100")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_axisbelow(True)
    ax.legend(frameon=True, ncol=5, loc="lower right")
    plt.tight_layout()
    plt.show()
    filename = os.path.join("figures", "fig3.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def import_ijhmt_df(filename):
    # first column gives Pw/Patm
    # second column gives contribution by H2O
    # subsequent columns give the sum of previous columns plus constituent
    # pOverlap represents total emissivity for the given pw
    filename = os.path.join("data", "ijhmt_2019_data", filename)
    colnames = ['pw', 'H2O', 'pCO2', 'pO3', 'paerosols',
                'pN2O', 'pCH4', 'pO2', 'pN2', 'poverlaps']
    df = pd.read_csv(filename, names=colnames, header=0)
    return df


def plot_fig3():
    # graph fig3 from data Mengying provided
    df = import_ijhmt_df("fig3_esky_i.csv")
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    fig, ax = plt.subplots()
    x = df.pw.to_numpy()
    species = list(LI_TABLE1.keys())[:-1]
    species = species[::-1]
    j = 0
    for i in species:
        if i == "H2O":
            y = i
        else:
            y = f"p{i}"
        ax.fill_between(x, 0, df[y].to_numpy(), label=i, fc=cmaplist[-(j + 1)])
        j += 1
    ax.set_ylim(bottom=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("pw")
    ax.set_ylabel(r"$\varepsilon$")
    plt.show()
    return None


def plot_fig3_ondata(s, sample):
    """Plot esky_c fits over sampled clear sky site data with altitude
    correction applied. Colormap by solar time.

    Parameters
    ----------
    s : str
        SURFRAD site code
    sample : [0.05, 0.25]
        If larger than 10%, temperature is filtered to +/- 2K.

    Returns
    -------
    None
    """
    site = import_cs_compare_csv("cs_compare_2012.csv", site=s)
    site = site.loc[site.zen < 80].copy()
    site["pp"] = site.pw_hpa * 100 / P_ATM

    site = site.sample(frac=sample, random_state=96)
    if sample > 0.1:  # apply filter to temperature
        site = site.loc[abs(site.t_a - 294.2) < 2]
        title = f"{s} 2012 {sample:.0%} sample, zen<80, +/-2K"
    else:
        title = f"{s} 2012 {sample:.0%} sample, zen<80"
    tmp = pd.DatetimeIndex(site.solar_time.copy())
    site["solar_tod"] = tmp.hour + (tmp.minute / 60) + (tmp.second / 3600)
    de_p = site.de_p.values[0]

    df = import_ijhmt_df("fig3_esky_i.csv")
    x = df.pw.to_numpy()
    df["total"] = df.pOverlaps
    df["pred_y"] = 0.6376 + (1.6026 * np.sqrt(x))
    df["best"] = 0.6376 + (1.6191 * np.sqrt(x))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    c = ax.scatter(
        site.pp, site.e_act_s, c=site.solar_tod, cmap="seismic",
        vmin=5, vmax=19, alpha=0.8,
    )
    ax.plot(x, df.total + de_p, label="LBL")
    ax.plot(x, df.pred_y + de_p, ls="--", label="(0.6376, 1.6026)")
    ax.plot(x, df.best + de_p, ls=":", label="(0.6376, 1.6191)")
    ax.legend()
    fig.colorbar(c, label="solar time")
    ax.set_ylabel("effective sky emissivity [-]")
    ax.set_xlabel("p$_w$ [-]")
    ax.set_xlim(0, 0.03)
    ax.set_ylim(0.60, 1.0)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    filename = os.path.join("figures", "fig3_ondata.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def plot_fig3_quantiles(site=None, yr=None, all_sites=False, tau=False,
                        temperature=False, shk=False, pct_clr_min=0.3,
                        pressure_bins=20, violin=False):
    # DISCLAIMER: tau may need to be adjusted since s["x"] is now defined
    #   using three_c_fit to solve for c1,c2, and c3 at once

    # Create dataset for training
    s = create_training_set(
        year=yr, temperature=temperature,
        cs_only=True, filter_pct_clr=0.05, filter_npts_clr=0.2, drive="server4"
    )
    s = reduce_to_equal_pts_per_site(s)  # reduce to num pts in common

    # import ijhmt data to plot
    df = import_ijhmt_df("fig3_esky_i.csv")
    x_lbl = df.pw.to_numpy()
    df["total"] = df.pOverlaps

    if tau:
        s["transmissivity"] = 1 - s.y  # create set will create y col (e_act)
        s["y"] = -1 * np.log(s.transmissivity)
        y_lbl = -1 * np.log(1 - df.total)
        ylabel = "optical depth [-]"
        # ymin, ymax = 0.8, 3.0
    else:
        y_lbl = df.total
        ylabel = "effective sky emissivity [-]"
        # ymin, ymax = 0.60, 1.0
    labels = ["Q05-95", "Q25-75", "Q40-60"]
    quantiles = [0.05, 0.25, 0.4, 0.6, 0.75, 0.95]
    pressure_bins = pressure_bins

    # Prepare data for quantiles plot
    c1, c2, c3, xq, yq = prep_plot_data_for_quantiles_plot(
        s, pressure_bins, quantiles, violin=violin
    )
    y_fit = c1 + c2 * np.sqrt(x_lbl)

    if shk:  # if using only one site, apply shakespeare model
        pw = (x_lbl * P_ATM) / 100  # convert back to partial pressure [hPa]
        pa_hpa = s.P_rep.values[0] / 100
        w = 0.62198 * pw / (pa_hpa - pw)
        q_values = w / (1 + w)
        he = s.he.values[0]

        lat1 = SURFRAD[site]["lat"]
        lon1 = SURFRAD[site]["lon"]
        h1, spline = shakespeare(lat1, lon1)

        tau_shakespeare = []
        for q1 in q_values:
            tau_shakespeare.append(spline.ev(q1, he).item())
        # don't need to make any altitude adjustments in shakespeare
        if tau:  # optical depth
            y_sp = np.array(tau_shakespeare)
        else:  # emissivity
            y_sp = (1 - np.exp(-1 * np.array(tau_shakespeare)))

    # set title and filename
    suffix = "_tau" if tau else ""
    suffix += "_ta" if temperature else ""
    suffix += f"_clr{pct_clr_min*100:.0f}"
    suffix += f"_{pressure_bins}"
    suffix += f"_violin" if violin else ""
    title_suffix = r", T~T$_0$" if temperature else ""
    title_suffix += f" day_clr>{pct_clr_min:.0%}"
    if len(yr) == 1:
        str_name = f"{yr[0]}"
        name = yr[0]
    else:
        str_name = f"{yr[0]} - {yr[-1]}"
        name = f"{yr[0]}_{yr[-1]}"

    if all_sites:
        filename = os.path.join("figures", f"fig3{name}_q{suffix}.png")
        title = f"SURFRAD {str_name} (n={s.shape[0]:,}) ST>8" + title_suffix
    else:
        filename = os.path.join("figures", f"fig3{name}_{site}_q{suffix}.png")
        title = f"{site} {str_name} (n={s.shape[0]:,}) ST>8" + title_suffix

    # Create figure
    if violin:
        violin_figure(
            x_lbl, y_lbl, y_fit, xq, yq, c1, c2, c3,
            title, filename, showmeans=True, showextrema=False
        )
    else:
        quantiles_figure(
            x_lbl, y_lbl, y_fit, c1, c2, c3, xq, yq, ylabel,
            title, filename, quantiles, labels
        )

    return None


def prep_plot_data_for_quantiles_plot(df, pressure_bins, quantiles, violin=False):
    # PREPARE PLOTTING DATA
    df["pp"] = df.pw_hpa * 100 / P_ATM
    df["quant"] = pd.qcut(df.pp, pressure_bins, labels=False)
    # find linear fit
    c1, c2, c3 = three_c_fit(df)
    print(c1, c2, c3)
    df["de_p"] = c3 * (P_ATM / 100000) * (np.exp(-1 * df.elev / 8500) - 1)
    df["y"] = df.y - df.de_p  # revise y and bring to sea level

    # Find quantile data per bin
    xq = np.zeros(pressure_bins)
    if violin:
        yq = []
    else:
        yq = np.zeros((pressure_bins, len(quantiles)))
    for i, group in df.groupby(df.quant):
        xq[i] = group.pp.median()
        if violin:
            yq.append(group.y.to_numpy())
        else:
            for j in range(len(quantiles)):
                yq[i, j] = group.y.quantile(quantiles[j])
    return c1, c2, c3, xq, yq


def quantiles_figure(x, y, y2, c1, c2, c3, xq, yq, ylabel, title,
                     filename, quantiles, labels, y_sp=None):
    clrs_g = ["#c8d5b9", "#8fc0a9", "#68b0ab", "#4a7c59", "#41624B"]
    clrs_p = ["#C9A6B7", "#995C7A", "#804D67", "663D52", "#4D2E3E"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    ax.plot(x, y, label="LBL", c="k")
    ax.plot(x, y2, label="fit", ls="--", lw=1, c="g")
    if y_sp is not None:
        ax.plot(x, y_sp, label="Shakespeare", c="0.2", ls=":")
    for i in range(int(len(quantiles) / 2)):
        t = int(-1 * (i + 1))
        ax.fill_between(
            xq, yq[:, i], yq[:, t], alpha=0.3, label=labels[i],
            fc=clrs_g[i], ec="0.9",
        )
        # to make smoother, use default step setting
    text = r"$\varepsilon$ = " + f"{c1} + {c2}" + r"$\sqrt{p_w}$" + \
           f" + {c3}(" + r"$e^{-z/H}$" + " - 1)"
    ax.text(
        0.95, 0.05, s=text, transform=ax.transAxes, ha="right", va="bottom",
        backgroundcolor="1.0", alpha=0.8
    )
    ax.set_xlabel("p$_w$ [-]")
    ax.set_xlim(0, 0.03)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel(ylabel)
    ax.legend(ncols=3, loc="upper left")
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def violin_figure(x, y, y2, xq, yq, c1, c2, c3, title, filename,
                  showmeans=True, showextrema=False):
    clrs_g = ["#c8d5b9", "#8fc0a9", "#68b0ab", "#4a7c59", "#41624B"]

    showmedians = False if showmeans else True
    parts_list = []
    if showmeans:
        parts_list.append("cmeans")
    if showmedians:
        parts_list.append("cmedians")
    if showextrema:
        parts_list.append("cmins")
        parts_list.append("cmaxes")
        parts_list.append("cbars")

    fig, (ax, ax0) = plt.subplots(2, 1, figsize=(8, 5),
                                  sharex=True, height_ratios=[41, 2])
    fig.subplots_adjust(hspace=0.05)
    parts = ax.violinplot(
        yq, xq, showmeans=showmeans,
        showextrema=showextrema,
        showmedians=showmedians,
        widths=np.diff(xq).min(),
    )
    ax.plot(x, y, label="LBL", c="k")
    ax.plot(x, y2, label="fit", ls="--", lw=1, c=clrs_g[-1])
    ax.legend()

    for pc in parts['bodies']:
        pc.set_facecolor(clrs_g[1])
        pc.set_edgecolor(clrs_g[1])
    for p in parts_list:
        vp = parts[p]
        vp.set_edgecolor(clrs_g[2])

    ax0.set_ylim(0, 0.02)
    ax.set_ylim(0.59, 1.0)
    # hide the spines between ax and ax2
    ax.spines.bottom.set_visible(False)
    ax0.xaxis.tick_bottom()
    ax0.spines.top.set_visible(False)
    ax.xaxis.tick_top()
    ax0.tick_params(labeltop=False)  # don't put tick labels at the top
    ax0.set_yticks([0])

    # Draw lines indicating broken axis
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
    ax0.plot([0, 1], [1, 1], transform=ax0.transAxes, **kwargs)
    ax0.set_xlim(0, 0.025)
    ax0.set_xlabel("p$_w$ [-]")
    ax.set_ylabel("effective sky emissivity [-]")

    text = r"$\varepsilon$ = " + f"{c1} + {c2}" + r"$\sqrt{p_w}$" + \
           f" + {c3}(" + r"$e^{-z/H}$" + " - 1)"
    ax.text(
        0.95, 0.0, s=text, transform=ax.transAxes, ha="right", va="bottom",
        backgroundcolor="1.0", alpha=0.8
    )

    ax.grid(True, alpha=0.3)
    ax0.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax0.set_axisbelow(True)
    ax.set_title(title, loc="left")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def ijhmt_to_tau(filename="lc2019_esky_i.csv"):
    df = import_ijhmt_df(filename=filename)
    df = df.set_index("pw")
    # (method 1) convert to aggregated tau, then disaggregate
    df = 1 - df  # each column now represents aggregated transmissivities
    col1 = df.H2O.to_numpy()  # preserve first column
    df = df.div(df.shift(axis=1), axis=1)  # shift axis 1 then divide
    df["H2O"] = col1  # replace first col of NaNs with H2O transmissivities
    # last value is the cumulative product of all previous values
    df["total"] = df.cumprod(axis=1).iloc[:, -1]

    # remove the first p in the column names
    for c in df.columns:
        if c[0] == "p":
            df = df.rename(columns={c: c[1:]})
    return df


def ijhmt_to_individual_e(filename="lc2019_esky_i.csv"):
    df = import_ijhmt_df(filename=filename)
    df = df.set_index("pw")
    col1 = df.H2O.to_numpy()
    df = df.diff(axis=1)  # each column is individual emissivity
    df["H2O"] = col1  # first column already individual emissivity
    df["total"] = df.cumsum(axis=1).iloc[:, -1]

    # remove the first p in the column names
    for c in df.columns:
        if c[0] == "p":
            df = df.rename(columns={c: c[1:]})
    return df


def plot_fig3_tau():
    # two subplots showing wideband individual and cumulative contributions
    lbl = [
        'H2O', 'CO2', 'O3', 'Aerosols', 'N2O', 'CH4', 'O2', 'N2', 'Overlaps'
    ]
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(len(lbl))]

    df = ijhmt_to_tau("fig3_esky_i.csv")  # tau, first p removed
    x = df.index.to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    i = 0
    y_ref = np.ones(len(x))
    for s in lbl:
        col = s if i == 0 else f"p{s}"
        ax1.plot(x, df[col], label=s, c=cmaplist[i])
        ax2.plot(x, y_ref * df[col], label=s, c=cmaplist[i])
        y_ref = y_ref * df[col]
        i += 1
    ax1.plot(x, y_ref, label="total", c="0.0", ls="--")
    ax2.plot(x, y_ref, label="total", c="0.0", ls="--")
    ax1.set_title("Individual contributions", loc="left")
    ax2.set_title("Cumulative transmissivity", loc="left")
    ax1.set_xlim(x[0], x[-1])
    ax1.set_xlabel("$p_w$ [-]")
    ax2.set_xlabel("$p_w$ [-]")
    ax1.set_ylabel("transmissivity [-]")
    ax2.set_ylabel("transmissivity [-]")
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 0.5)
    ax2.grid(alpha=0.3)
    ax1.grid(alpha=0.3)
    ax2.legend(ncol=2, loc="upper right")
    # plt.show()
    filename = os.path.join("figures", "fig3_tau.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_fit3_dopt():
    # single plot showing d_opt vs pw
    lbl = [
        'H2O', 'CO2', 'O3', 'Aerosols', 'N2O', 'CH4', 'O2', 'N2', 'Overlaps'
    ]
    labels = [
        'H$_2$O', 'CO$_2$', 'O$_3$', 'aerosols',
        'N$_2$O', 'CH$_4$', 'O$_2$', 'N$_2$', 'overlaps'
    ]
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(len(lbl))]

    # df = ijhmt_to_tau("fig3_esky_i.csv")
    df = ijhmt_to_tau("fig5_esky_ij_b4.csv")  # tau, first p removed
    # df = ijhmt_to_individual_e("fig3_esky_i.csv")
    x = df.index.to_numpy()

    fig, ax = plt.subplots(figsize=(5, 5), sharex=True)
    i = 0
    y_ref = np.zeros(len(x))
    # y_ref = np.ones(len(x))
    for s in lbl:
        y = -1 * np.log(df[s])
        # y = df[s]
        ax.plot(x, y, label=labels[i], c=cmaplist[i])
        # y_ref = y_ref * y  # transmissivity
        y_ref += y  # dopt or emissivity
        i += 1
    ax.plot(x, y_ref, label="total", c="0.0", ls="--")
    ax.set_title("Individual contributions (band 4)", loc="left")
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("$p_w$ [-]")
    ax.set_ylabel(r"$d_{\rm{opt}}$ [-]")
    # ax.set_ylabel("emissivity [-]")
    ax.set_ylim(0, 0.02)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, loc="upper right")
    plt.tight_layout()
    plt.show()
    filename = os.path.join("figures", "temp.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_tau_spectral_vs_wideband(tau=True, part="total"):
    if tau:
        df = ijhmt_to_tau("lc2019_esky_i.csv")
        plot_title = "transmissivity"
        legend_label = r"$\tau_{ij}$ over $j$"
        s = "tau"  # for figure name
    else:
        df = ijhmt_to_individual_e("lc2019_esky_i.csv")
        plot_title = "emissivity"
        legend_label = r"sum of $\varepsilon_{ij}$ over $j$"
        s = "eps"
    x = df.index.to_numpy()
    y = df[part].to_numpy()

    df = ijhmt_to_individual_e("lc2019_esky_i.csv")
    ye = df[part].to_numpy()

    y_b = np.zeros(len(x))
    d_opt = np.zeros(len(x))
    for i in np.arange(1, 8):
        if tau:
            df = ijhmt_to_tau(f"lc2019_esky_ij_b{i}.csv")
            d_opt += -1 * np.log(df[part].to_numpy())
        else:
            df = ijhmt_to_individual_e(f"lc2019_esky_ij_b{i}.csv")
        y_b = y_b + df[part].to_numpy()

    if tau:
        y_b = y_b - 6  # +1 -7 bands

    fig, ax = plt.subplots()
    ax.plot(x, y_b, lw=2, label=legend_label)
    ax.plot(x, y, ls="--", label="wideband")
    ax.plot(x, 1 - ye, ls=":", label="1 - e_i")
    # ax.plot(x, np.exp(-1 * d_opt), "s", c="r", label="sum of d$_{opt}$ over $j$")
    ax.set_title(plot_title + f" ({part})", loc="left")
    if part == "total":
        i = 10
    else:
        species = list(LI_TABLE1.keys())[:-1]
        i = species.index(part)
    ax.set_xlabel("p$_w$ [-]")
    # ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    filename = os.path.join("figures", f"{s}_{i}_{part}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def create_data_tables_from_lc2019():

    tab1 = LI_TABLE1

    tab2 = dict(
        b1=dict(
            H2O=[0.1725, 0, 0],
            CO2=[0, 0, 0],
            O3=[0, 0, 0],
            aerosols=[0, 0, 0],
            N2O=[0, 0, 0],
            CH4=[0, 0, 0],
            O2=[0, 0, 0],
            N2=[0, 0, 0],
            overlaps=[0, 0, 0],
            total=[0.1725, 0, 0]
        ),
        b2=dict(
            H2O=[0.1083, 0.0748, 270.8944],
            CO2=[0.0002, 0, 0],
            O3=[0, 0, 0],
            aerosols=[0.0002, 0, 0],
            N2O=[0.0001, 0, 0],
            CH4=[0, 0, 0],
            O2=[0, 0, 0],
            N2=[0, 0, 0],
            overlaps=[0.0003, 0, 0],
            total=[0.1170, 0.0662, 270.4686]
        ),
        b3=dict(
            H2O=[-0.2308, 0.6484, 0.1280],
            CO2=[0.3038, -0.5262, 0.1497],
            O3=[0, 0, 0],
            aerosols=[0.0001, 0, 0],
            N2O=[0.0001, 0, 0],
            CH4=[0, 0, 0],
            O2=[0, 0, 0],
            N2=[0, 0, 0],
            overlaps=[17.0770, -17.0907, 0.0002],
            total=[0.1457, 0.0417, 0.0992]
        ),
        b4=dict(
            H2O=[0.0289, 6.2436, 0.9010],
            CO2=[0.0144, -0.1740, 0.7268],
            O3=[0.0129, -0.4970, 1.1620],
            aerosols=[0.0159, -0.3040, 0.8828],
            N2O=[0.0018, 0, 0],
            CH4=[0.0243, -0.0312, 0.0795],
            O2=[0, 0, 0],
            N2=[0, 0, 0],
            overlaps=[0.0227, -0.2748, 0.7480],
            total=[0.1057, 5.8689, 0.9633]
        ),
        b5=dict(
            H2O=[0.0775, 0, 0],
            CO2=[0, 0, 0],
            O3=[-0.0002, 0, 0],
            aerosols=[0, 0, 0],
            N2O=[-0.0006, 0, 0],
            CH4=[0, 0, 0],
            O2=[0, 0, 0],
            N2=[0, 0, 0],
            overlaps=[-0.0001, 0, 0],
            total=[0.0766, 0, 0]
        ),
        b6=dict(
            H2O=[0.0044, 0, 0],
            CO2=[-0.0022, 0, 0],
            O3=[0, 0, 0],
            aerosols=[0, 0, 0],
            N2O=[0, 0, 0],
            CH4=[0, 0, 0],
            O2=[0, 0, 0],
            N2=[0, 0, 0],
            overlaps=[-0.0002, 0, 0],
            total=[0.0019, 0, 0]
        ),
        b7=dict(
            H2O=[0.0033, 0, 0],
            CO2=[-0.0003, 0, 0],
            O3=[0, 0, 0],
            aerosols=[-0.0001, 0, 0],
            N2O=[-0.0001, 0, 0],
            CH4=[0, 0, 0],
            O2=[0, 0, 0],
            N2=[0, 0, 0],
            overlaps=[-0.0002, 0, 0],
            total=[0.0026, 0, 0]
        )
    )

    folder = os.path.join("data", "ijhmt_2019_data")
    xmin, xmax = (0.001, 0.025)  # normalized
    x = np.geomspace(xmin, xmax, 50)  # normalized

    species = list(tab1.keys())[:-1]

    y_ref = np.zeros(len(x))
    df = pd.DataFrame(dict(pw=x))
    for i in species:
        c1, c2, c3 = tab1[i]
        y = c1 + c2 * np.power(x, c3)
        colname = i if i == "H2O" else f"p{i}"
        y_ref = y_ref + y
        df[colname] = y_ref

    filename = os.path.join(folder, "lc2019_esky_i.csv")
    df.to_csv(filename)

    # do this per band
    for j in np.arange(1, 8):
        df = pd.DataFrame(dict(pw=x))
        y_ref = np.zeros(len(x))
        band_dict = tab2[f"b{j}"]
        for i in species:
            c1, c2, c3 = band_dict[i]
            if j == 2:
                y = c1 + c2 * np.tanh(c3 * x)
            else:
                y = c1 + c2 * np.power(x, c3)
            colname = i if i == "H2O" else f"p{i}"
            y_ref = y_ref + y
            df[colname] = y_ref
        filename = os.path.join(folder, f"lc2019_esky_ij_b{j}.csv")
        df.to_csv(filename)
    return None


def sum_ei(e, i):
    """

    Parameters
    ----------
    e : DataFrame
        Individual e table (either esky_i or esky_ij)
    i : int
        Index where 1 <= i <= 9

    Returns
    -------
    sum of e_i up to and including i
    """
    # df should be an individual e table
    return e.iloc[:, :i].sum(axis=1).to_numpy()


if __name__ == "__main__":
    print()
    # t_a = 294.2  # [K]
    # rh = 50  # %
    # pw = get_pw_norm(t_a, rh)

    # plot_fig3_ondata("FPK", sample=0.05)
    # plot_fig3_quantiles(
    #     yr=[2010, 2011, 2012, 2013], all_sites=True, tau=False,
    #     temperature=True, pct_clr_min=0.5, pressure_bins=5, violin=True
    # )
    # plot_fig3_quantiles(
    #     yr=[2015], tau=False,
    #     temperature=False, pct_clr_min=0.05, pressure_bins=10, violin=True
    # )
    print()
    # df = import_ijhmt_df("fig3_esky_i.csv")  # original
    # df = ijhmt_to_tau("fig5_esky_ij_b4.csv")  # tau, first p removed
    # df = ijhmt_to_individual_e("fig3_esky_i.csv")  # e, disaggregated

    # create_data_tables_from_lc2019()
    # plot_wide_vs_banded(tau=True, part="CO2")

    # # calculate band weights for transmissivity
    # t = 294.2
    # lw_const = integrate.quad(func=planck_lambda, a=4, b=100000, args=(t,))[0]
    # bw = []
    # for i in np.arange(1, 8):
    #     l1, l2 = BANDS_L[f"b{i}"]
    #     out = integrate.quad(func=planck_lambda, a=l1, b=l2, args=(t,))[0]
    #     bw.append(out / lw_const)
    # bw = np.array(bw)

    df = ijhmt_to_individual_e("lc2019_esky_i.csv")
    x = df.index.to_numpy()
    e_total = df["total"].to_numpy()  # e_total
    df = ijhmt_to_tau("lc2019_esky_i.csv")
    t_total = df["total"].to_numpy()  # t_total

    species = list(df.columns[:-1])
    part = "O3"

    # e_j = np.zeros(len(x))

    # e_i = ijhmt_to_individual_e("lc2019_esky_i.csv")
    # t_i = ijhmt_to_tau("lc2019_esky_i.csv")
    # for part in species[2:]:
    #     # part = "total"
    #     ii = species.index(part)
    #     term1 = (1-t_i[part].to_numpy())
    #     term2 = sum_ei(e_i, ii-1)
    #     # term2 = e_i[part].to_numpy()
    #     lhs = t_i[part].to_numpy() + (term1*term2)
    #     t_j = np.ones(len(x))
    #     for j in np.arange(1, 8):
    #         tau = ijhmt_to_tau(f"lc2019_esky_ij_b{j}.csv")
    #         t_j = t_j * tau[part].to_numpy()
    #     compare = lhs - t_j
    #     print(part, compare.max(), compare.mean())

    part = "O3"

    df = ijhmt_to_tau("fig3_esky_i.csv")
    plot_title = "transmissivity"
    legend_label = r"$\tau_{ij}$ over $j$"
    s = "tau"  # for figure name
    x = df.index.to_numpy()
    y = df[part].to_numpy()

    df = ijhmt_to_individual_e("fig3_esky_i.csv")
    ye = df[part].to_numpy()
    m = 7  # number of bands
    idx = species.index(part)
    rhs = m - sum_ei(df, idx + 1)  # i-1 (+1 for index=0)
    # term2 = -1 * sum_ei(df, idx)
    term2 = -1 * y * sum_ei(df, idx)

    y_b = np.zeros(len(x))  # tau_ij
    d_opt = np.zeros(len(x))
    lhs = np.zeros(len(x))
    term1 = np.zeros(len(x))
    for i in np.arange(1, 8):
        df = ijhmt_to_tau(f"fig5_esky_ij_b{i}.csv")
        t_ij = df[part].to_numpy()
        d_opt += -1 * np.log(t_ij)
        y_b = y_b + t_ij
        df = ijhmt_to_individual_e(f"fig5_esky_ij_b{i}.csv")
        lhs = lhs + (t_ij * (1 - sum_ei(df, idx)))
        term1 += (t_ij * sum_ei(df, idx))

    # if tau:
    y_b = y_b - 6  # +1 -7 bands

    fig, ax = plt.subplots()
    ax.plot(x, y_b, lw=2, label=legend_label)
    ax.plot(x, y, ls="--", label="wideband")
    ax.plot(x, 1 - ye, ls=":", label="1 - e_i")
    # ax.fill_between(x, 1-ye, 1-ye+adj, alpha=0.5)
    # f = (term1 + term2).mean()
    # ax.fill_between(x, y, y +f, alpha=0.5)
    ax.fill_between(x, y, y + term1 + term2, alpha=0.25, label="adjustment")
    # ax.plot(x, rhs - 6, "rs", label="RHS")
    # ax.plot(x, lhs - 6, "g*", label="LHS")
    # ax.plot(x, np.exp(-1 * d_opt), "s", c="r", label="sum of d$_{opt}$ over $j$")
    ax.set_title(plot_title + f" ({part})", loc="left")
    if part == "total":
        i = 10
    else:
        species = list(LI_TABLE1.keys())[:-1]
        i = species.index(part)
    ax.set_xlabel("p$_w$ [-]")
    ax.legend()
    # ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    filename = os.path.join("figures", f"show_tij_to_ti_{part}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)