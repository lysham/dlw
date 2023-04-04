"""Implement and compare external and internal fraction functions."""

import os
import numpy as np
import pandas as pd
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


def de_wrt_dt(l, t):
    t1 = np.exp(E_C2 / (l * t)) - 1
    term1 = -5 * E_C1 / (np.power(l, 6) * t1)
    t2 = np.exp(E_C2 / (l * t)) / (np.power(l, 7) * t * np.power(t1, 2))
    term2 = E_C1 * E_C2 * t2
    return term1 + term2


def ext_fraction_lambda(l, t):
    const = 1 / (SIGMA * np.power(t, 4))
    out = integrate.quad(func=planck_lambda, a=0, b=l, args=(t,))[0]
    return const * out


if __name__ == "__main__":
    print()

    t = 294.2
    out = integrate.quad(func=planck_lambda, a=3, b=50, args=(t,))[0]
    print(out)