"""Implement and compare external and internal fraction functions."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

from constants import SIGMA

# radiative constants
E_C1 = 3.7418e8  # W um^4 / m^2 ~> 2*pi*h*c^2
E_C2 = 1.4389e4  # um K         ~> hc/k_B


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


if __name__ == "__main__":
    print()

    t = 294.2
    out = integrate.quad(func=planck_lambda, a=3, b=50, args=(t,))[0]
    print(out)

    # fraction function implementations imperfect (inf approximated as 256)
    print(fi_lt(14300))  # 0.9875 Table 6.5


