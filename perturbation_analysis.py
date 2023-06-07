"""explore linear perturbation of lbl correlations"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import integrate
from fraction import planck_lambda


def plot_fig5_band(pw, h2o, co2, de_h2o, de_co2, title="B2"):
    # FOR PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    ax = axes[0]
    ax.grid(axis="y")
    ax.fill_between(pw, 0, h2o, label="h2o", alpha=0.7)
    ax.fill_between(pw, h2o, h2o + co2, label="co2", alpha=0.7)
    ax.legend(ncol=2, bbox_to_anchor=(1.0, 1.01), loc="lower right")
    ax.set_xlim(pw[0], pw[-1])
    ax.set_ylim(0, 0.3)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    txt = r"$\varepsilon$"
    ax.set_ylabel(txt)
    ax.set_xlabel("dimensionless p$_w$")

    ax = axes[1]
    ax.grid(axis="y")
    ax.plot(pw, de_h2o, label="h2o")
    ax.plot(pw, de_co2, label="co2")
    ax.plot(pw, de_h2o + de_co2, ls="--", c="0.5", label="sum")
    ax.legend(ncol=3, bbox_to_anchor=(1.0, 1.01), loc="lower right")
    txt = r"d$\varepsilon$ / dp$_w$"
    ax.set_ylabel(txt)
    ax.set_xlabel("dimensionless p$_w$")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(axis="y")
    ax.plot(pw, de_b2_h2o + de_b2_co2, label="B2 sum")
    ax.plot(pw, de_b3_h2o + de_b3_co2, label="B3 sum")
    ax.plot(pw, de_b4_h2o + de_b4_co2, label="B4 sum")
    co2_total = de_b2_co2 + de_b3_co2 + de_b4_co2
    h2o_total = de_b2_h2o + de_b3_h2o + de_b4_h2o
    total = co2_total + h2o_total
    ax.plot(pw, h2o_total, c="0.5", ls=":", label="co2 sum")
    ax.plot(pw, co2_total, c="0.5", ls="--", label="h2o sum")
    ax.plot(pw, total, c="0.1", ls="-", label="total")
    ax.set_xlim(pw[0], pw[-1])
    txt = r"d$\varepsilon$ / dp$_w$"
    ax.set_ylabel(txt)
    ax.set_xlabel("dimensionless p$_w$")
    ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.01), loc="lower center")
    plt.tight_layout()
    plt.show()
    return None


if __name__ == "__main__":

    p0 = np.geomspace(0.001, 2.5/100, 100)
    frac_change_p = np.array([0.005, 0.01, 0.02, 0.03])

    b2_h2o = 0.1083 + 0.0748 * np.tanh(270.8944 * p0)
    b3_h2o = -0.2308 + 0.6484 * np.power(p0, 0.1280)
    b4_h2o = 0.0289 + 6.2436 * np.power(p0, 0.9010)
    de_b2_h2o = 20.2629 * np.power(np.cosh(270.8944 * p0), -2)
    de_b3_h2o = (0.6484 * 0.1280) * np.power(p0, 0.1280 - 1)
    de_b4_h2o = (6.2436 * 0.9010) * np.power(p0, 0.9010 - 1)

    b2_co2 = 0.0002
    b3_co2 = 0.3038 - 0.5262 * np.power(p0, 0.1497)
    b4_co2 = 0.0144 - 0.1740 * np.power(p0, 0.7268)
    de_b2_co2 = np.zeros(len(p0))
    de_b3_co2 = (-0.5262 * 0.1497) * np.power(p0, 0.1497 - 1)
    de_b4_co2 = (-0.1740 * 0.7268) * np.power(p0, 0.7268 - 1)

    b2_ovl = 0.0003
    b3_ovl = 17.0770 - 17.0907 * np.power(p0, 0.0002)
    b4_ovl = 0.0227 - 0.2748 * np.power(p0, 0.7480)
    de_b2_ovl = np.zeros(len(p0))
    de_b3_ovl = (-17.0907 * 0.0002) * np.power(p0, 0.0002 - 1)
    de_b4_ovl = (-0.2748 * 0.7480) * np.power(p0, 0.7480 - 1)

    # plot_fig5_band(p0, b2_h2o, b2_co2, de_b2_h2o, de_b2_co2, title="B2")

    total = \
        (de_b2_co2 / b2_co2) + (de_b3_co2 / b3_co2) + (de_b4_co2 / b4_co2) + \
        (de_b2_h2o / b2_h2o) + (de_b3_h2o / b3_h2o) + (de_b4_h2o / b4_h2o) + \
        (de_b2_ovl / b2_ovl) + (de_b3_ovl / b3_ovl) + (de_b4_ovl / b4_ovl)

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    ylabel = r"$p_0$ ${\epsilon}_{ij}'$ $\epsilon_{ij}^{-1}$"
    ax = axes[0]
    ax.plot(p0, p0 * (de_b2_h2o / b2_h2o), label="B2")
    ax.plot(p0, p0 * (de_b3_h2o / b3_h2o), label="B3")
    ax.plot(p0, p0 * (de_b4_h2o / b4_h2o), label="B4")
    ax.grid(alpha=0.3)
    ax.set_title("j = H2O", loc="left")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax = axes[1]
    ax.plot(p0, p0 * (de_b2_co2 / b2_co2), label="B2")
    ax.plot(p0, p0 * (de_b3_co2 / b3_co2), label="B3")
    ax.plot(p0, p0 * (de_b4_co2 / b4_co2), label="B4")
    ax.grid(alpha=0.3)
    ax.set_title("j = CO2", loc="left")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax = axes[2]
    ax.plot(p0, p0 * (de_b2_ovl / b2_ovl), label="B2")
    ax.plot(p0, p0 * (de_b3_ovl / b3_ovl), label="B3")
    ax.plot(p0, p0 * (de_b4_ovl / b4_ovl), label="B4")
    ax.grid(alpha=0.3)
    ax.set_title("j = overlaps", loc="left")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_xlabel("dimensionless water vapor (p$_0$)")
    plt.tight_layout()
    plt.show()

    total_num = \
        de_b2_co2 + de_b2_h2o + de_b2_ovl + \
        de_b3_co2 + de_b3_h2o + de_b3_ovl + \
        de_b4_co2 + de_b4_h2o + de_b4_ovl
    total_denom = \
        b2_co2 + b3_co2 + b4_co2 + b2_h2o + b3_h2o + \
        b4_h2o + b2_ovl + b3_ovl + b4_ovl
    t2_num = total_num - (de_b2_co2 + de_b3_co2 + de_b4_co2)
    t2_denom = total_denom - (b2_co2 + b3_co2 + b4_co2)

    y1 = p0 * (total_num / total_denom)
    y2 = p0 * (t2_num / t2_denom)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p0, y1, label="y1 where j=(H2O, CO2, overlaps)")
    ax.plot(p0, y2, label="y2 where j=(H2O, overlaps)")
    ax.plot(p0, y2 - y1, c="0.3", ls="--", label="y2 - y1")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, p0[-1])
    ax.set_xlabel("dimensionless water vapor (p$_0$)")
    ax.grid(alpha=0.3)
    ax.set_title("i=(B2, B3, B4)", loc="left")
    plt.tight_layout()
    plt.show()

    y1 = p0 * (de_b3_co2 + de_b3_h2o + de_b3_ovl) / (b3_co2 + b3_h2o + b3_ovl)
    y2 = p0 * (de_b3_h2o + de_b3_ovl) / (b3_h2o + b3_ovl)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p0, y1, label="y1 where j=(H2O, CO2, overlaps)")
    ax.plot(p0, y2, label="y2 where j=(H2O, overlaps)")
    ax.plot(p0, y2 - y1, c="0.3", ls="--", label="y2 - y1")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, p0[-1])
    ax.set_xlabel("dimensionless water vapor (p$_0$)")
    ax.grid(alpha=0.3)
    ax.set_title("i=B3", loc="left")
    plt.tight_layout()
    plt.show()

    # # Reference
    # np.interp(0.01, p0, (de_b3_h2o / b3_h2o))
    #
    # o2 = integrate.quad(func=planck_lambda, a=4, b=200, args=(288,))[0]
    # o1 = integrate.quad(func=planck_lambda, a=13.3, b=17.2, args=(288,))[0]