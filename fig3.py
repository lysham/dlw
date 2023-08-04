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


def tmp_spectral_vs_wideband_after_tau_adj():

    part = "O3"
    species = list(LI_TABLE1.keys())
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
    # df = import_ijhmt_df("fig3_esky_i.csv")  # original
    # df = ijhmt_to_tau("fig5_esky_ij_b4.csv")  # tau, first p removed
    # df = ijhmt_to_individual_e("fig3_esky_i.csv")  # e, disaggregated

    # create_data_tables_from_lc2019()
