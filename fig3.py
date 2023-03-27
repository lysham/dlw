"""Recreate Fig 3 in Li and Coimbra 2019"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from main import get_pw
from corr26b import shakespeare

from constants import LI_TABLE1, P_ATM, SIGMA, N_BANDS, N_SPECIES, SURFRAD


def get_pw_norm(t, rh):
    # get dimensionless pw
    pw = get_pw(t, rh)  # Pa
    return pw / P_ATM


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


if __name__ == "__main__":
    print()
    # t_a = 294.2  # [K]
    # rh = 50  # %
    # pw = get_pw_norm(t_a, rh)
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
    ax.set_xlim(pw_x[0], pw_x[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel("pw x 100")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_axisbelow(True)
    ax.legend(frameon=True, ncol=5, loc="lower right")
    plt.tight_layout()
    plt.show()
    # filename = os.path.join("figures", "fig3_s.png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)

