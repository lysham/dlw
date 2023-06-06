"""explore linear perturbation of lbl correlations"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    p0 = np.linspace(0.001, 2.5/100, 50)
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

    fig, ax = plt.subplots(figsize=(8, 5))
    f2 = (p0 / b2_h2o) * de_b2_h2o
    f3 = (p0 / b3_h2o) * de_b3_h2o
    f4 = (p0 / b4_h2o) * de_b4_h2o
    ax.plot(p0, f3 + f4 + f2, c="0.2", label="sum")
    ax.plot(p0, f2, "--", label="B2")
    ax.plot(p0, f3, "--", label="B3")
    ax.plot(p0, f4, "--", label="B4")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    ylabel = r"$p_0 / \epsilon_0$ (d$\epsilon$ / d$p_w$)"
    ax.set_ylabel(ylabel)
    ax.set_title("j = H2O", loc="left")
    ax.set_xlabel("dimensionless water vapor (p$_0$)")
    plt.tight_layout()
    plt.show()

    # plot_fig5_band(p0, b2_h2o, b2_co2, de_b2_h2o, de_b2_co2, title="B2")

    total2 = (de_b3_h2o / b3_h2o) + (de_b3_co2 / b3_co2) + \
             (de_b4_co2 / b4_co2) + (de_b4_h2o / b4_h2o)
    total = (de_b3_co2 / b3_co2) + (de_b3_h2o / b3_h2o) + \
            (de_b4_co2 / b4_co2) + (de_b4_h2o / b4_h2o)

    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    for f in frac_change_p:
        frac_change_e = (f * p0) * total
        ax.plot(p0, frac_change_e, label=f)
    ax.legend(title="frac change in p$_w$")
    ax.set_ylabel("fractional change in emissivity")
    ax.set_xlabel("dimensionless water vapor (p$_0$)")
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    ax = axes[0]
    ax.plot(p0, (de_b2_h2o / b2_h2o), label="B2")
    ax.plot(p0, (de_b3_h2o / b3_h2o), label="B3")
    ax.plot(p0, (de_b4_h2o / b4_h2o), label="B4")
    ax.grid(alpha=0.3)
    ax.set_title("j = H2O", loc="left")
    ax.set_ylabel("M$_{ij}$(p$_0$)")
    ax.legend()
    ax = axes[1]
    ax.plot(p0, (de_b2_co2 / b2_co2), label="B2")
    ax.plot(p0, (de_b3_co2 / b3_co2), label="B3")
    ax.plot(p0, (de_b4_co2 / b4_co2), label="B4")
    ax.grid(alpha=0.3)
    ax.set_title("j = CO2", loc="left")
    ax.set_ylabel("M$_{ij}$(p$_0$)")
    ax.legend()
    ax.set_xlabel("dimensionless water vapor (p$_0$)")
    plt.tight_layout()
    plt.show()

    np.interp(0.01, p0, (de_b3_h2o / b3_h2o))