"""
Functions for interfacing with HITRAN.

Author: Mengying Li

"""

import os
import numpy as np
from multiprocessing import Pool
import hapi
from . import set_pressure, set_temperature, set_height

# default data directory
DATA_DIR = os.path.join("data", "hitran")

__all__ = [
    "MolID_rho",
    "getDataFromHitran",
    "getKappa_SingleLayer",
    "getKappa_AllLayers",
]


# find Hitran Molecule ID by Molecule name
def MolID_rho(molecule, P, T):
    """Get HITRAN molecule ID and density.

    Get the HITRAN molecule ID based on the molecule name, and calculate the
    molecule's density.

    Parameters
    ----------
    molecule : str
        Molecule name.
    P : float
        Pressure [Pa].
    T : float
        Temperature [K].

    Returns
    -------
    mol_id : int
        HITRAN molecule ID.
    rho : float
        Density [g/cm^3].

    """

    if molecule == "H2O":
        mol_id = 1
        M = 18  # molecular weight g/mol
    elif molecule == "CO2":
        mol_id = 2
        M = 44
    elif molecule == "O3":
        mol_id = 3
        M = 48
    elif molecule == "N2O":
        mol_id = 4
        M = 44
    elif molecule == "CO":
        mol_id = 5
        M = 28
    elif molecule == "CH4":
        mol_id = 6
        M = 16
    elif molecule == "O2":
        mol_id = 7
        M = 32
    elif molecule == "N2":
        mol_id = 22
        M = 28

    Ru = 8.314  # [J/(mol * K)]
    rho = M * P / Ru / T  # [g/m^3]
    rho /= 1e6  # [g/cm^3]
    return mol_id, rho


def getDataFromHitran(molecule, nu_min, nu_max, data_dir=DATA_DIR):
    """Download data from HITRAN.

    Download data from HITRAN using the HITRAN API (hapi) and then store the
    results in a local folder.

    Parameter
    ---------
    molecule : str
        Name of the molecules (e.g. "H2O", "CO2").
    nu_min, nu_max : float
        Minimum and maximum wavenumber [cm^-1].
    data_dir : path
        Directory used for holding HITRAN data files.

    Return
    ------
    None

    """

    hapi.db_begin(data_dir)
    mol_id, rho = MolID_rho(molecule, 1.013 * 1e5, 296)
    hapi.fetch(molecule, mol_id, 1, nu_min, nu_max)


def getKappa_SingleLayer(args):
    """Get single layer absorption coefficients.

    Get the absoprtion coefficients of the M molecules for a single layer.

    Parameters
    ----------
    args : (4,) array_like
        0) molecules : (M,) array_like
            The names of the M molecules.
        1) nu : (N,) array_like
            Wavenumbers [cm^-1].
        2) T :
            Temperature [K].
        3) P :
            Pressure [Pa].

    Returns
    -------
    coeff_V : (M,) array_like
        Volumetric absorption coefficients [cm^-1] of each of the M molecules,
        at one layer.

    """

    # LAST MODIFIED: Mengying Li 04/20/2017
    # INPUTS: args is a list (help with parallel programming)
    # molecules: name of molecules (vector)
    # nu: vector of wavenumbers, cm-1
    # T: temperature, K (scalar)
    # P: pressure, Pa (scalar)
    # OUTPUT:
    # coeff: volumetric absorption coefficient of each modecules at one layer, cm-1, list of size len(molecules)
    # WRITTEN BY: Mengying Li 04/20/2017

    # get the arguments from list args
    layer = args[0]
    molecules = args[1]
    nu = args[2]
    T = args[3]
    P = args[4]
    print("Starting layer {}...".format(layer))

    # initialize coeff
    coeff = np.zeros((1, len(nu)))
    coeff_V = []  # list of coefficients, each element is coeff for one gas
    for i in range(0, len(molecules)):
        mol_id, mol_rho = MolID_rho(molecules[i], P, T)
        nu_raw, coeff = hapi.absorptionCoefficient_Voigt(
            Components=[(mol_id, 1, 1)],  # List of tuples (M,I,D)
            SourceTables=molecules[i],  # list of source tables
            HITRAN_units=False,  # unit in cm-1
            OmegaGrid=nu,  # wavenumber grid
            OmegaWing=25,  # absolute value of line wing, cm-1
            OmegaWingHW=50,  # line wing relative ratio to half width
            Environment={"p": P / 1.013 / 1e5, "T": T},  # in unit cm-1
        )
        # coeff= np.interp(nu, nu_raw, coeff_raw,left=0,right=0) # coeff=0 if out of range
        coeff /= mol_rho  # in unit cm2/g # mass absorption coefficient
        coeff_V.append(coeff)
    print("Finishing layer {}...".format(layer))
    return coeff_V


def getKappa_AllLayers(n_layers, profile, molecules, nu, T_delta=0, data_dir=DATA_DIR, night=False):
    """Get absorption coefficients for N layers.

    Get the absoprtion coefficients of the M molecules for N layers.

    Parameters
    ----------
    n_layer : int
        Number of layers.
    profile: str
        Atmospheric profile.
    molecules : (M,) array_like
        The names of the M molecules.
    nu : array_like
        Wavenumbers [cm^-1].
    T_delta : float
        Temperatue delta to apply to the atmospheric profile.
    data_dir : path
        Directory used for holding HITRAN data files.

    Returns
    -------
    coeff_M : list
        Absorption coefficients.

    """

    # LAST MODIFIED: Mengying Li 04/20/2017
    # INPUTS:
    # model: profile model, e.g. 'tropical'
    # N_layer: number of layers
    # molecules: name of molecules (vector)
    # nu: vector of wavenumbers, cm-1
    # ta: temperature of each layer, K (vector)
    # pa: pressure of each layer, Pa (vector)
    # OUTPUT:
    # coeff_M saved to file: volumetric absorption coefficient of each modecules at each layer, cm-1, list of size N_layer
    # WRITTEN BY: Mengying Li 04/20/2017

    # in case database isn't already initialized
    hapi.db_begin(data_dir)

    p, pa = set_pressure(n_layers)  # pressure [Pa]
    t, ta = set_temperature(profile, p, pa)  # temperature [K]
    z, za = set_height(profile, p, pa)  # height [m]

    ## shift temperature profiles
    #t += T_delta
    #ta += T_delta

    if night:
        T_night_offset = 6.0  # offset [K]
    else:
        T_night_offset = 0.0

    # linear troposphere with temperature shift
    # - shift T at the surface by T_delta
    # - linear intepolate T in the troposphere
    # - leave T above the troposphere the same
    if profile == "AFGL_midlatitude_summer":
        i = 26
    elif profile == "AFGL_US_standard":
        i = 24
        print(profile, i)
    else:
        print("Invalid profile")
    z_trop, za_trop = z[1:i], za[:i]
    t_trop, ta_trop = t[1:i], ta[:i]
    z_linear, za_linear = z_trop.copy(), za_trop.copy()
    t_linear = np.interp(z_linear, (z_trop[0], z_trop[-1]), (t_trop[0] + T_delta + 2 * T_night_offset, t_trop[-1]))
    ta_linear = np.interp(za_linear, (za_trop[0], za_trop[-1]), (ta_trop[0] + T_delta + 2 * T_night_offset, ta_trop[-1]))
    t[0] = t_linear[0]
    t[1:i] = t_linear
    ta[:i] = ta_linear

    # add temperature inversion at 1 km
    if T_night_offset > 0:
        idx = 8
        z_inv, za_inv = z[1:idx], za[:idx]
        t_inv, ta_inv = t[1:idx], ta[:idx]
        z_linear, za_linear = z_inv.copy(), za_inv.copy()
        t_linear = np.interp(z_linear, (z_inv[0], z_inv[-1]), (t_inv[0] - 2 * T_night_offset, t_inv[-1]))
        ta_linear = np.interp(za_linear, (za_inv[0], za_inv[-1]), (ta_inv[0] - 2 * T_night_offset, ta_inv[-1]))
        t[0] = t_linear[0]
        t[1:idx] = t_linear
        ta[:idx] = ta_linear

    list_args = []
    for i in range(0, n_layers + 1):
        args = [i, molecules, nu, ta[i], pa[i]]
        list_args.append(args)

    # run in parallel
    pool = Pool()
    coeff_M = list(pool.map(getKappa_SingleLayer, list_args))
    pool.terminate()
    return coeff_M