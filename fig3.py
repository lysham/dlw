"""Recreate Fig 3 in Li and Coimbra 2019"""

import os
import scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from main import get_pw
from corr26b import shakespeare, import_cs_compare_csv, fit_linear, compute_mbe
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


def plot_fig3_quantiles():
    site = "BON"
    s = import_cs_compare_csv("cs_compare_2012.csv", site=site)
    s = s.loc[s.zen < 80].copy()
    s["pp"] = s.pw_hpa * 100 / P_ATM
    s["quant"] = pd.qcut(s.pp, 20, labels=False)
    de_p = s.de_p.values[0]

    quantiles = [0.05, 0.1, 0.25, 0.45, 0.55, 0.75, 0.9, 0.95]
    xq = np.zeros(20)
    yq = np.zeros((20, len(quantiles)))
    for i, g in s.groupby(s.quant):
        xq[i] = g.pp.median()
        for j in range(len(quantiles)):
            yq[i, j] = g.e_act_s.quantile(quantiles[j])

    df = import_ijhmt_df("fig3_esky_i.csv")
    x = df.pw.to_numpy()
    df["total"] = df.pOverlaps
    df["pred_y"] = 0.6376 + (1.6026 * np.sqrt(x))
    df["best"] = 0.6376 + (1.6191 * np.sqrt(x))

    clrs = ["#c8d5b9", "#8fc0a9", "#68b0ab", "#4a7c59", "#41624B"]
    labels = ["Q5-95", "Q10-90", "Q25-75", "Q45-55"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    ax.plot(x, df.total + de_p, label="LBL", c="k")
    for i in range(int(len(quantiles) / 2)):
        t = int(-1 * (i + 1))
        ax.fill_between(
            xq, yq[:, i], yq[:, t], alpha=0.3, label=labels[i],
            fc=clrs[i], ec="0.9"
        )
    ax.set_ylabel("effective sky emissivity [-]")
    ax.set_xlabel("p$_w$ [-]")
    ax.set_xlim(0, 0.03)
    ax.set_ylim(0.60, 1.0)
    ax.set_axisbelow(True)
    ax.legend(ncols=3)
    ax.set_title(f"{site} 2012 (n={s.shape[0]:,}) zen<80", loc="left")
    filename = os.path.join("figures", f"fig3_{site.lower()}_quantiles.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
    return None


if __name__ == "__main__":
    print()
    # t_a = 294.2  # [K]
    # rh = 50  # %
    # pw = get_pw_norm(t_a, rh)

    # plot_fig3_ondata("FPK", sample=0.05)
    # plot_fig3_quantiles()

    site = import_cs_compare_csv("cs_compare_2014.csv")
    site["pp"] = site.pw_hpa * 100 / P_ATM
    site = site.loc[(site.zen < 80) & (abs(site.t_a - 294.2 < 2))].copy()
    site = site.sample(frac=0.25, random_state=96)
    site["esky"] = site.e_act_s - site.de_p
    site["transmissivity"] = 1 - site.esky
    site["tau"] = -1 * np.log(site.transmissivity)

    site["x"] = site.pp.to_numpy()
    site["y"] = site.tau.to_numpy()
    c1, c2 = fit_linear(site, print_out=True)

    site["x"] = np.sqrt(site.pp.to_numpy())
    site["y"] = site.esky.to_numpy()
    c1, c2 = fit_linear(site, print_out=True)

    df = import_ijhmt_df("fig3_esky_i.csv")
    df["transmissivity"] = 1 - df.pOverlaps
    df["tau"] = -1 * np.log(df.transmissivity)
    df["x"] = df.pw.to_numpy()
    df["y"] = df.tau.to_numpy()
    c1, c2 = fit_linear(df, print_out=True)

    fig, ax = plt.subplots()
    ax.scatter(site.pp, site.tau, alpha=0.01)
    ax.plot(df.x, df.tau)
    plt.show()