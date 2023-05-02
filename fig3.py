"""Recreate Fig 3 in Li and Coimbra 2019"""

import os
import scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from main import get_pw
from corr26b import shakespeare, import_cs_compare_csv, fit_linear, \
    compute_mbe, three_c_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from scipy.stats import pearsonr

from constants import LI_TABLE1, P_ATM, SIGMA, N_BANDS, N_SPECIES, SURFRAD, \
    ELEVATIONS


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

    cc = list(LI_TABLE1.keys())  # contributing components
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
    species = list(LI_TABLE1.keys())

    pw_x = np.linspace(0.1, 2.3, 20)
    e_broad = np.zeros((len(pw_x), N_SPECIES))

    e_tau = np.zeros(len(pw_x))
    e_tau_p0 = np.zeros(len(pw_x))
    site = "BON"
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
    colnames = ['pw', 'H2O', 'pCO2', 'pO3', 'pAerosols',
                'pN2O', 'pCH4', 'pO2', 'pN2', 'pOverlaps']
    df = pd.read_csv(filename, names=colnames, header=0)
    return df


def plot_fig3():
    # graph fig3 from data Mengying provided
    df = import_ijhmt_df("fig3_esky_i.csv")
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    fig, ax = plt.subplots()
    x = df.pw.to_numpy()
    species = list(LI_TABLE1.keys())
    species = species[::-1]
    j = 0
    for i in species:
        if i == "H2O":
            y = i
        elif i == "aerosols" or i == "overlaps":
            y = f"p{i[0].upper() + i[1:]}"
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
                        temperature=False, pressure=False, shk=False):
    # plot from cs_compare files, either "_{year}.csv" or "_{site}.csv"
    # DISCLAIMER: tau may need to be adjusted since s["x"] is now defined
    #   using three_c_fit to solve for c1,c2, and c3 at once

    # import afgl data
    filename = os.path.join("data", "afgl_midlatitude_summer.csv")
    afgl = pd.read_csv(filename)
    afgl_alt = afgl.alt_km.values * 1000  # m
    afgl_temp = afgl.temp_k.values
    afgl_pa = afgl.pres_mb.values

    if isinstance(yr, int) & all_sites:
        s = import_cs_compare_csv(f"cs_compare_{yr}.csv")
    elif isinstance(yr, int) & ~all_sites:
        s = import_cs_compare_csv(f"cs_compare_{yr}.csv", site=site)
    else:
        s = import_cs_compare_csv(f"cs_compare_{site}.csv")
    s = s.set_index("solar_time")
    s = s.loc[s.index.hour > 8].copy()  # remove data before 8am solar
    s["afgl_t0"] = np.interp(s.elev.values / 1000, afgl_alt, afgl_temp)
    s["afgl_p"] = np.interp(s.elev.values / 1000, afgl_alt, afgl_pa)

    clrs_g = ["#c8d5b9", "#8fc0a9", "#68b0ab", "#4a7c59", "#41624B"]
    labels = ["Q05-95", "Q25-75", "Q40-60"]
    quantiles = [0.05, 0.25, 0.4, 0.6, 0.75, 0.95]
    clrs_p = ["#C9A6B7", "#995C7A", "#804D67", "663D52", "#4D2E3E"]
    pressure_bins = 12

    if temperature:
        s = s.loc[abs(s.t_a - s.afgl_t0) <= 2].copy()
        # s = s.loc[abs(s.t_a - 294.2) < 2].copy()
    if pressure:
        s = s.loc[abs(s.pa_hpa - s.afgl_p) <= 50].copy()
    s["pp"] = s.pw_hpa * 100 / P_ATM
    s["quant"] = pd.qcut(s.pp, pressure_bins, labels=False)
    s["x"] = np.sqrt(s.pp)
    if tau:
        s["transmissivity"] = 1 - s.e_act
        s["y"] = -1 * np.log(s.transmissivity)
    else:
        s["y"] = s.e_act
    # processed and filtered dataset created

    # import ijhmt data to plot
    df = import_ijhmt_df("fig3_esky_i.csv")
    x = df.pw.to_numpy()
    df["total"] = df.pOverlaps
    if tau:
        y = -1 * np.log(1 - df.total)
    else:
        y = df.total

    # find linear fit
    c1, c2, c3 = three_c_fit(s)
    print(c1, c2, c3)
    y2 = c1 + c2 * np.sqrt(x)
    s["de_p"] = c3 * (P_ATM / 100000) * (np.exp(-1 * s.elev / 8500) - 1)
    s["y"] = s.e_act - s.de_p  # revise y and bring to sea level

    # Find quantile data per bin
    xq = np.zeros(pressure_bins)
    yq = np.zeros((pressure_bins, len(quantiles)))
    for i, g in s.groupby(s.quant):
        xq[i] = g.pp.median()
        for j in range(len(quantiles)):
            yq[i, j] = g.y.quantile(quantiles[j])

    if shk:  # if using only one site, apply shakespeare model
        pw = (x * P_ATM) / 100  # convert back to partial pressure [hPa]
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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    ax.plot(x, y, label="LBL", c="k")
    ax.plot(x, y2, label=f"{c1}+{c2}x", ls="--", lw=1, c="g")
    if shk:
        ax.plot(x, y_sp, label="Shakespeare", c="0.2", ls=":")
    for i in range(int(len(quantiles) / 2)):
        t = int(-1 * (i + 1))
        ax.fill_between(
            xq, yq[:, i], yq[:, t], alpha=0.3, label=labels[i],
            fc=clrs_g[i], ec="0.9", step="mid"
        )
        # to make smoother, use default step setting
    ax.set_xlabel("p$_w$ [-]")
    ax.set_xlim(0, 0.03)
    if tau:
        ax.set_ylabel("optical depth [-]")
        ax.set_ylim(0.8, 3.0)
    else:
        ax.set_ylabel("effective sky emissivity [-]")
        ax.set_ylim(0.60, 1.0)
    ax.legend(ncols=3, loc="upper left")
    ax.set_axisbelow(True)

    # set title and filename
    suffix = "_tau" if tau else ""
    suffix += "_ta" if temperature else ""
    suffix += "_pa" if pressure else ""
    suffix += f"_{pressure_bins}"
    title_suffix = r", T~T$_0$" if temperature else ""
    title_suffix += r", P~P$_0$" if pressure else ""
    if isinstance(yr, int) & all_sites:
        filename = os.path.join("figures", f"fig3_{yr}_q{suffix}.png")
        title = f"{yr} (n={s.shape[0]:,}) ST>0800" + title_suffix
    elif isinstance(yr, int) & ~all_sites:
        filename = os.path.join("figures", f"fig3_{site}_q{suffix}.png")
        title = f"{site} {yr} (n={s.shape[0]:,}) ST>0800" + title_suffix
    else:
        filename = os.path.join("figures", f"fig3_{site}_q5yr{suffix}.png")
        title = f"{site} 2012-2016 (n={s.shape[0]:,}) ST>0800" + title_suffix
    ax.set_title(title, loc="left")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    print()
    # t_a = 294.2  # [K]
    # rh = 50  # %
    # pw = get_pw_norm(t_a, rh)

    # plot_fig3_ondata("FPK", sample=0.05)
    plot_fig3_quantiles(
        yr=2012, all_sites=True, tau=False, temperature=True, pressure=True
    )