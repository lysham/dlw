"""Implement and compare external and internal fraction functions."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, special

from constants import SIGMA, E_C1, E_C2


def planck_lambda(l, t):
    """Evaluate Planck's distribution for hemispheric blackbody emissivity
    in terms of wavelength (l) for temperature (t)

    Parameters
    ----------
    l : double
        Wavelength (lambda) in microns
    t : double
        Temperature in K

    Returns
    -------
    E_bl(l, t)
    """
    t1 = E_C1 * np.power(l, -5)
    t2 = np.exp(E_C2 / (l * t)) - 1
    return t1 / t2


def ext_fraction_lambda(l, t):
    const = 1 / (SIGMA * np.power(t, 4))
    out = integrate.quad(func=planck_lambda, a=0, b=l, args=(t,))[0]
    return const * out


def fe_lt(lt):
    """External fractional function as a function of lambda * T (lt)

    Parameters
    ----------
    lt : double
        Lambda * T [micron * K]

    Returns
    -------
    fe(lt)
    """
    const = E_C1 / (SIGMA * np.power(E_C2, 4))
    y = lambda z: np.power(z, 3) / (np.exp(z) - 1)
    out = integrate.quad(func=y, a=E_C2 / lt, b=256)[0]  # z to inf
    return const * out


def fi_lt(lt):
    """Internal fractional function as a function of lambda * T (lt)
    * some error at larger lt (e.g. order 10^-4 at lt of 10^5)

    Parameters
    ----------
    lt : double
        Lambda * T [micron * K]

    Returns
    -------
    fi(lt)
    """
    const = E_C1 / (4 * SIGMA * np.power(E_C2, 4))
    y = lambda z: (np.power(z, 4) * np.exp(z)) / np.power((np.exp(z) - 1), 2)
    out = integrate.quad(func=y, a=E_C2 / lt, b=256)[0]  # z to inf
    return const * out


def _fe(lt):
    return 1 - ((90/np.power(np.pi, 4)) * special.zeta(E_C2/lt, 4))


def fi_fe(x):
    return (15 / np.power(np.pi, 4)) * np.power(x, 4) / (np.exp(x) - 1)


if __name__ == "__main__":
    print()

    t = 294.2
    out = integrate.quad(func=planck_lambda, a=3, b=50, args=(t,))[0]
    print(out)

    # fraction function implementations imperfect (inf approximated as 256)
    # print(fi_lt(14300))  # 0.9875 Table 6.5

    x = np.linspace(0.001, 20, 100)
    y = fi_fe(x)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(True, alpha=0.1)
    ax.plot(x, y)
    ax.axvline(3.92069, c="0.4", ls="--", label=r"$X_z$")
    ax.axvline(12.23, c="0.6", ls=":", label=r"X$_{c,2}$")
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("X")
    ax.set_ylabel("F(X)")
    ax.set_axisbelow(True)
    ax.legend()
    filename = os.path.join("figures", "F_v_X.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    X_param = E_C2 / (7 * 290)
