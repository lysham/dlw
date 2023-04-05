"""
Common functions for both shortwave (SW) and longwave (LW).

Author: Mengying Li
"""

import numpy as np
import scipy.integrate as integrate
from scipy.special import jv, yv
from multiprocessing import Pool
import os

__all__ = [
    "set_pressure",
    "set_temperature",
    "set_height",
    "set_ndensity",
    "set_vmr",
    "set_vmr_circ",
    "saturation_pressure",
    "getMixKappa",
    "absorptionContinuum_MTCKD",
    "absorptionContinuum_MTCKD_CO2",
    "absorptionContinuum_MTCKD_O3",
    "radiation_field",
    "kd_integrate",
    "Planck",
    "Mie",
    "Mie_ab",
    "aerosol_monoLam",
    "aerosol",
    "cloud",
    "rayleigh_kappa_s",
    "Z_air",
    "Planck_lam",
    "aerosol_CIRC",
    "surface_albedo",
]


def set_pressure(N_layer):
    """
    Set the pressure of the N layers.

    Parameters
    ----------
    N : int
        The number of atmosphere layers.

    Returns
    -------
    p : (N + 2,) array_like
        Presure [Pa] at each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] of each layer.

    """

    n = np.arange(0.5, N_layer + 1, 0.5)
    sig = (2 * N_layer - 2 * n + 1) / 2 / N_layer
    pa_temp = sig ** 2 * (3 - 2 * sig)

    # initialize, has a layer inside the Earth
    p = np.zeros(N_layer + 2)
    pa = np.zeros(N_layer + 1)
    for i in range(N_layer, 0, -1):
        p[i] = pa_temp[2 * i - 2]
        pa[i] = pa_temp[2 * i - 1]

    # ground layers
    p[0] = p[1]
    pa[0] = p[0]

    # convert to [Pa]
    p *= 1.013e5
    pa *= 1.013e5
    return p, pa


def set_temperature(model, p, pa):
    """
    Set temperature of the N layers.

    With N atmosphere layers, there are N + 2 node points.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    p : (N + 2,) array_like
        Presure [Pa] at each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] of each layer.

    Returns
    -------
    t : (N + 2,) array_like
        Temperature [K] at each node point.
    ta : (N + 1,) array_like
        Average temperature [K] of each layer.

    References
    ----------
    [1] G. P. Anderson, "AFGL atmospheric consituent profiles (0-120 km)",
        1986.

    """

    # data from AFGL Atmospheric Constituent Profiles (1986)
    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    ref_p = data[:, 1]
    ref_t = data[:, 2]
    ref_p = np.asarray(ref_p) * 1e2  # convert unit to Pa
    ref_t = np.asarray(ref_t)
    t = np.interp(-p, -ref_p, ref_t)
    ta = np.zeros((len(pa)))

    # pressure averaged temperature
    for i in range(1, len(pa)):
        ta[i] = (t[i] * (p[i] - pa[i]) + t[i + 1] * (pa[i] - p[i + 1])) / (
            p[i] - p[i + 1]
        )
    ta[0] = t[0]

    return t, ta


def set_height(model, p, pa):
    """
    Set the height of the N layers.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    p : (N + 2,) array_like
        Pressure [Pa] at each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] at each layer.

    Returns
    -------
    z : (N + 2,) array_like
        Height [m] at each node point.
    za : (N + 1,) array_like
        Average height [m] of each layer.

    """

    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    ref_p = data[:, 1]
    ref_z = data[:, 0]

    # convert units
    ref_p = np.asarray(ref_p) * 1e2  # pressure [Pa]
    ref_z = np.asarray(ref_z) * 1e3  # height [m]

    # xv needs to be increasing for np.interp to work properly
    z = np.interp(-p, -ref_p, ref_z)

    # average height
    z_avg = np.zeros((len(pa)))
    for i in range(1, len(pa)):
        # pressure averaged t
        z_avg[i] = (z[i] * (p[i] - pa[i]) + z[i + 1] * (pa[i] - p[i + 1])) / (
            p[i] - p[i + 1]
        )

    z_avg[0] = z[0]
    return z, z_avg


def set_ndensity(model, p, pa):
    """
    Set the number density of the N layers.

    Used for calculating the Rayleight scattering coefficient.

    Parameters
    ----------
    model : str
    p : (N + 2,) array_like
        Pressure [Pa] of each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] of each layer.

    Returns
    -------
    n : (N + 2,) array_like
        Number density [unit/cm^3] of each node point.
    na : (N + 1,) array_like
        Average number density [unit/cm^3] of each layer.

    """

    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    ref_p = data[:, 1]
    ref_n = data[:, 3]  # [1/cm^3]

    ref_p = np.asarray(ref_p) * 1e2  # convert to [Pa]
    ref_n = np.asarray(ref_n)

    # xv needs to be increasing for np.interp to work properly
    n = np.interp(-p, -ref_p, ref_n)
    na = np.zeros((len(pa)))
    for i in range(1, len(pa)):
        # pressure averaged t
        na[i] = (n[i] * (p[i] - pa[i]) + n[i + 1] * (pa[i] - p[i + 1])) / (
            p[i] - p[i + 1]
        )
    na[0] = n[0]
    return n, na


def set_vmr(model, molecules, vmr0, z):
    """
    Set volumetric mixing ratio (vmr) of atmosphere gases for N layers.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    molecules : (M,) array_like
        Names of the M molecules.
    vmr0 : float
        Surface vmr of molecules (for scaling purposes).
    z : (N + 2,) array_like
        Height [m] of each node point.

    Returns
    -------
    vmr : (N + 2,) array_like
        Volumetric mixing ratio (vmr) [ppmv] at each node point.
    densities : (N + 1,) array_like
        Average density of each gas in each layer (accumulated weight/volume
        [g/cm^3]).

    """

    vmr = np.zeros((len(molecules), len(z)))
    densities = np.zeros((len(molecules), len(z) - 1))
    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    data2 = np.genfromtxt("data/profiles/AFGL_molecule_profiles.csv", delimiter=",")

    # reference height (converted from [km] to [m]
    ref_z = data[:, 0] * 1e3

    for i in range(0, len(molecules)):
        if molecules[i] == "H2O":
            ref_vmr = data[:, 4]
            M = 18  # molecular weight, g/mol
        elif molecules[i] == "CO2":
            ref_vmr = data2[:, 2]
            M = 44
        elif molecules[i] == "O3":
            ref_vmr = data[:, 5]
            M = 48
        elif molecules[i] == "N2O":
            ref_vmr = data[:, 6]
            M = 44
        elif molecules[i] == "CH4":
            ref_vmr = data[:, 8]
            M = 16
        elif molecules[i] == "O2":
            ref_vmr = data2[:, 7]
            M = 32
        elif molecules[i] == "N2":
            ref_vmr = data2[:, 8]
            M = 28
        ref_vmr = np.asarray(ref_vmr) / 1e6  # change from ppv to unit 1
        ref_Ni = data[:, 3] * ref_vmr  # in unit of #molecules/cm^3
        NA = 6.022 * 1e23  # in unit #/mol

        # reference density [g/cm^3] (after scaling)
        ref_rho = ref_Ni / NA * M / ref_vmr[0] * vmr0[i]
        vmr[i, :] = np.interp(z, ref_z, ref_vmr) / ref_vmr[0] * vmr0[i]
        for j in range(1, len(z) - 1):
            zz = np.linspace(z[j], z[j + 1], 100)  # evenly spaced coordinate
            rrho = np.interp(zz, ref_z, ref_rho)

            # distance averaged density
            densities[i, j] = integrate.trapz(rrho, zz) / abs(zz[0] - zz[-1])

        densities[i, 0] = densities[i, 1]

    # output arrays so:
    # - rows = node points
    # - cols = molecules
    #
    vmr = np.transpose(vmr)
    densities = np.transpose(densities)
    return vmr, densities


def set_vmr_circ(model, molecules, pa, ta):
    """
    Set volumetric mixing ratio for N layers (for CIRC cases).

    Parameters
    ----------
    model : str
    molecules : (M,) array_like
        Names of the M molecules.
    pa : (N + 1,) array_like
        Average pressure [Pa] of all layers.
    ta : (N + 1,) array_like
        Average temperature [K] of all layers.

    Returns
    -------
    vmr : (N + 1, M)
        Volume mixing ratio (vmr) of each layer for the M molcules.
    densities :

    """

    vmr = np.zeros((len(molecules), len(pa)))
    densities = np.zeros((len(molecules), len(pa)))
    data = np.genfromtxt(
        "results/CIRC/{}_input&output/layer_input_{}.txt".format(model, model),
        skip_header=3,
    )
    data = np.vstack((data[0, :], data))
    for i in range(0, len(molecules)):
        if molecules[i] == "H2O":
            ref_vmr = data[:, 3]
            M = 18  # molecular weight, g/mol
        elif molecules[i] == "CO2":
            ref_vmr = data[:, 4]
            M = 44
        elif molecules[i] == "O3":
            ref_vmr = data[:, 5]
            M = 48
        elif molecules[i] == "N2O":
            ref_vmr = data[:, 6]
            M = 44
        elif molecules[i] == "CO":  # has CO in CIRC
            ref_vmr = data[:, 7]
            M = 28
        elif molecules[i] == "CH4":
            ref_vmr = data[:, 8]
            M = 16
        elif molecules[i] == "O2":
            ref_vmr = data[:, 9]
            M = 32
        elif molecules[i] == "N2":
            ref_vmr = data[:, 9] / 21 * 78  # no N2 in CIRC
            M = 28
        for j in range(1, len(pa)):
            Ru = 8.314  # J/mol K
            density = pa[j] * ref_vmr[j] * M / Ru / ta[j]  # g/m3
            densities[i, j] = density / 1e6  # g/cm3

        densities[i, 0] = densities[i, 1]
        vmr[i, :] = ref_vmr

    densities = np.transpose(densities)
    vmr = np.transpose(vmr)
    return vmr, densities


def saturation_pressure(T):
    """
    Saturation pressure of water vapor of the N layers.

    Calculate the saturation pressure of water vapor using the Magus expression (see [1]).

    Parameters
    ----------
    T : (N + 2,) array_like
        Temperature [K] of the node points.

    Returns
    -------
    P_sat : (N + 2,) array_like
        Saturation pressure [Pa] of water vapor at each of the node points.

    References
    ----------
    [1] M. Li, Y. Jiang, and C. F. M. Coimbra (2017) “On the Determination of
        Atmospheric Longwave Irradiance Under All-Sky Conditions,” Solar Energy
        (144), pp. 40-48.

    """

    P_sat = 610.94 * np.exp(17.625 * (T - 273.15) / (T - 30.11))
    return P_sat


def getMixKappa(inputs, densities, pa, ta, z, za, na, AOD, COD, kap, data_dir, T_delta):
    """
    Absorption coefficients of gas mixture for N layers.

    Parameters
    ----------
    inputs :
    densities : array_like
    pa : (N + 1,) array_like
        Average pressure [Pa] of the layers.
    ta : (N + 1,) array_like
        Average temperature [K] of the layers.
    z : (N + 2,) array_like
        Height [m] of the node points.
    za : (N + 1,) array_like
        Average height [m] of the layers.
    na : (N + 1,) array_like
        Average number density [1/cm^3] of the layers.
    AOD : float
        Aerosol optical depth.
    COD : float
        Cloud optical depth.

    Returns
    -------
    [ka_gas_M, ks_gas_M, g_gas_M] :
        Gas
    [ka_aer_M, ks_aer_M, g_aer_M] :
        Aerosol
    [ka_cld_M, ks_cld_M, g_cld_M] :
        Cloud
    [ka_all_M, ks_all_M, g_all_M] :
        All

    """

    # LAST MODIFIED: Mengying Li 03/08/2018, includes aerosols and clouds and accommodates for SW Monte Carlo.
    # INPUTS:
    # inputs: N_layer,model,molecules,nu
    # densities: partial density of each moledule in each layer
    # pa: pressure, Pa (vector)
    # ta: temperature, K (vector)
    # AOD: aerosol optical depth @ 497.5 nm.
    # COD: cloud optical depth @ 497.5 nm.
    # kap: layers containing clouds. e.g. kap=[5,6,7] clouds are in layers 5,6 and 7.
    # OUTPUT:
    # volumetric absorption/scattering coefficients (cm-1) and asymmetry factor.
    # WRITTEN BY: Mengying Li 04/21/2017

    N_layer = inputs[0]
    model = inputs[1]
    molecules = inputs[2]
    nu = inputs[3]
    cld_model = inputs[4]
    spectral = inputs[5]
    coeff_M = np.load(os.path.join(data_dir, "coeffM_{}layers_{}_tdelta={:.0f}.npy".format(N_layer, model, T_delta)))

    # Add aerosols and clouds
    cldS = np.zeros(N_layer + 1)
    aerS = np.zeros(N_layer + 1)
    lam = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))
    nu_ref = 1e4 / lam
    lam0 = 0.4975
    if AOD > 0:
        aer_ka = np.load("results/ka_aerosol.npy")
        aer_ks = np.load("results/ks_aerosol.npy")
        aer_g = np.load("results/g_aerosol.npy")
        # add new aerosol vertical profile
        aer_vp = np.genfromtxt("data/profiles/aerosol_profile.csv", delimiter=",")
        aerS = (
            np.interp(za, aer_vp[:, 0], aer_vp[:, 3], left=0, right=0) * AOD
        )  # vertical AOD @ 497.5nm
    if COD > 0:
        cld_ks, cld_ka, cld_g = cloud(cld_model, z, kap)
        cldS[kap] = COD  # for CIRC cases

    ka_gas_M, ks_gas_M, g_gas_M, ka_aer_M, ks_aer_M, g_aer_M = [
        np.zeros([N_layer + 1, len(nu)]) for i in range(0, 6)
    ]
    ka_cld_M, ks_cld_M, g_cld_M, ka_all_M, ks_all_M, g_all_M = [
        np.zeros([N_layer + 1, len(nu)]) for i in range(0, 6)
    ]
    for i in range(1, N_layer + 1):
        ka_gas, ks_gas, ka_aer, ks_aer, g_aer, ka_cld, ks_cld, g_cld = [
            np.zeros(len(nu)) for i in range(0, 8)
        ]
        RH = 0  # defalt no water vapor present
        for j in range(0, len(molecules)):
            if densities[i, j] > 0:  # only if the gas is present
                # add continuum of water vapor
                if molecules[j] == "H2O":
                    # check relative humidity does not exceed 100%
                    x_h2o = (
                        (densities[i, j]) / 18 * 8.314 * ta[i] / pa[i]
                    )  # mole fraction
                    x_h2o *= 1e6  # unit conversion
                    ps = saturation_pressure(ta[i])  # saturated pressure
                    RH = pa[i] * x_h2o / ps * 100  # ranges from 0 to 100
                    if RH > 100:  # if exceeds 100
                        RH = 100
                    elif (cldS[i]>0): # cloud present in this layer
                        RH = 100.0 # RH=95% for cloud layers
                    x_h2o = RH / 100 * ps / pa[i]
                    x_h2o /= 1e6
                    densities[i, j] = (
                        x_h2o * pa[i] / ta[i] / 8.314 * 18
                    )  # change densities according to RH change
                    # #MTCKD continuum
                    ka_cont = absorptionContinuum_MTCKD(
                        nu, pa[i], ta[i], densities[i, j]
                    )  # return mass absorption coeff
                    ka_gas += (
                        ka_cont * densities[i, j]
                    )  # return volume absorption coeff
                # add continuum of CO2
                if molecules[j] == "CO2":
                    # #MTCKD continuum
                    ka_cont = absorptionContinuum_MTCKD_CO2(
                        nu, pa[i], ta[i], densities[i, j]
                    )  # return mass absorption coeff
                    ka_gas += (
                        ka_cont * densities[i, j]
                    )  # return volume absorption coeff
                    vmr_co2 = (densities[i, j]) / 44 * 8.314 * ta[i] / pa[i]
                    vmr_co2 *= 1e6  # unit conversion to 1
                # add continuum of O3
                if molecules[j] == "O3":
                    # #MTCKD continuum
                    ka_cont = absorptionContinuum_MTCKD_O3(
                        nu, pa[i], ta[i], densities[i, j]
                    )  # return mass absorption coeff
                    ka_gas += (
                        ka_cont * densities[i, j]
                    )  # return volume absorption coeff
                ka_gas += coeff_M[i][j] * densities[i, j]
        # Add Rayleigh scattering coefficient
        if spectral == "SW":
            ks_gas, temp = rayleigh_kappa_s(
                1e4 / nu, ta[i], pa[i], na[i], vmr_co2, x_h2o
            )
            # ks_gas*=1.5
            # ks_gas*=z[i+1]-z[i] # cummulated number of molecules, added on 3/28/2018
        # add aerosols
        if aerS[i] > 0:
            if RH == 0:
                ka_ref = aer_ka[0, :]
                ks_ref = aer_ks[0, :]
                g_ref = aer_g[0, :]
            else:
                n1 = int(np.floor(RH / 10))
                n2 = int(np.ceil(RH / 10))
                if n1 == n2:
                    ka_ref = aer_ka[n1, :]
                    ks_ref = aer_ks[n1, :]
                    g_ref = aer_g[n1, :]
                else:
                    ka_ref = aer_ka[n1, :] + (aer_ka[n2, :] - aer_ka[n1, :]) * (
                        RH / 10 - n1
                    ) / (n2 - n1)
                    ks_ref = aer_ks[n1, :] + (aer_ks[n2, :] - aer_ks[n1, :]) * (
                        RH / 10 - n1
                    ) / (n2 - n1)
                    g_ref = aer_g[n1, :] + (aer_g[n2, :] - aer_g[n1, :]) * (
                        RH / 10 - n1
                    ) / (n2 - n1)
            # scale aerosol according to desired AOD @ 500nm
            dz = 1575  # scale height,average of 2010_Yu
            kappa_e_ref = aerS[i] / (dz * 100)  # cm-1,desired extincion coeff
            kappa_e = np.interp(lam0, lam, ks_ref + ka_ref)
            ratio = kappa_e_ref / kappa_e
            ka_aer = (
                np.interp(-nu, -nu_ref, ka_ref, left=0, right=0) * ratio
            )  # correct using aerosol vertical profile
            ks_aer = (
                np.interp(-nu, -nu_ref, ks_ref, left=0, right=0) * ratio
            )  # correct using aerosol vertical profile
            g_aer = np.interp(-nu, -nu_ref, g_ref, left=0, right=0)
        # add clouds
        if cldS[i] > 0:
            ka_cld = np.interp(-nu, -nu_ref, cld_ka[i, :], left=0, right=0) * cldS[i]
            ks_cld = np.interp(-nu, -nu_ref, cld_ks[i, :], left=0, right=0) * cldS[i]
            g_cld = np.interp(-nu, -nu_ref, cld_g[i, :], left=0, right=0)
        # combine g of aerosol and cloud
        ks_all = ks_gas + ks_aer + ks_cld
        g_mix = g_aer * ks_aer + g_cld * ks_cld
        if aerS[i] > 0.0 or cldS[i] > 0.0:  # avoid dividing by zero
            g_mix /= ks_all

        ka_gas_M[i, :] = ka_gas
        ks_gas_M[i, :] = ks_gas
        ka_aer_M[i, :] = ka_aer
        ks_aer_M[i, :] = ks_aer
        g_aer_M[i, :] = g_aer
        ka_cld_M[i, :] = ka_cld
        ks_cld_M[i, :] = ks_cld
        g_cld_M[i, :] = g_cld
        g_all_M[i, :] = g_mix
    ka_all_M = ka_gas_M + ka_aer_M + ka_cld_M
    ks_all_M = ks_gas_M + ks_aer_M + ks_cld_M
    return (
        [ka_gas_M, ks_gas_M, g_gas_M],
        [ka_aer_M, ks_aer_M, g_aer_M],
        [ka_cld_M, ks_cld_M, g_cld_M],
        [ka_all_M, ks_all_M, g_all_M],
    )


def absorptionContinuum_MTCKD(nu, P, T, density):
    """
    Continuum absorption coefficients of water vapor.

    Parameters
    ----------
    nu :
        Wavenumbers [cm^-1].
    P :
        Total pressure [Pa].
    T :
        Temperature [K].
    density :
        Density [g/cm^3] of water vapor.

    Returns
    -------
    coeff_cont :
        Mass continuum absorption coefficients [cm^2/g] of water vapor.

    """

    # LAST MODIFIED: Mengying Li 05/18/2017 data from contnm.f90 and processed in matlab
    # calculate the continuum absorption coefficients of water vapor
    # nu: wavenumber considered (cm-1)
    # P: total pressure (Pa)
    # T: temperature (K)
    # density: density of water vapor (g/cm3)
    # OUTPUT:
    # coeff_cont: mass continuum absorption coefficients of water vapor (cm^2/g)

    data_frgn = np.genfromtxt("data/profiles/frgnContm.csv", delimiter=",")
    coeff_frgn = data_frgn[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    cf_frgn = data_frgn[:, 2]
    data_self = np.genfromtxt("data/profiles/selfContm.csv", delimiter=",")
    nu_self = data_self[:, 0]
    coeff_self_296 = data_self[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    coeff_self_260 = data_self[:, 2]
    cf_self = data_self[:, 3]

    # compute R factor
    T0 = 296
    P0 = 1.013e5
    c_h2o = density / 18  # molar density of h2o, in unit of mol/cm3
    c_tot = P / 8.314 / T / 1e6  # molar density of air, in unit of mol/cm3
    h2o_fac = c_h2o / c_tot
    RHOave = (P / P0) * (T0 / T)
    R_self = h2o_fac * RHOave  # consider partial pressure
    R_frgn = (1 - h2o_fac) * RHOave

    # compute self continuum coefficients
    tfac = (T - T0) / (260 - T0)
    coeff_self = (
        coeff_self_296 * (coeff_self_260 / coeff_self_296) ** tfac
    )  # temeprature correction
    coeff_self *= cf_self * R_self
    # compute foreign continuum coefficients
    coeff_frgn *= cf_frgn * R_frgn
    # sum the two
    coeff_tot = (
        (coeff_self + coeff_frgn) * 6.022 * 1e3
    )  # unit cm2/mol (cm)-1 * 6.022*10**23*10**(-20)
    coeff_tot *= c_h2o  # unit of cm2/cm3 (cm)-1
    coeff_tot /= density  # unit of cm2/g (cm)-1
    # interpelate to user defiend grid
    coeff_cont = np.interp(nu, nu_self, coeff_tot, left=0, right=0)
    RADFN = radiation_field(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN
    return coeff_cont


def absorptionContinuum_MTCKD_CO2(nu, P, T, density):
    """
    Continuum absorption coeffcients of CO2.

    Parameters
    ----------
    nu :
    P :
        Pressure [Pa].
    T :
        Temperature [K].
    density :

    Returns
    -------
    coeff_cont :
        Mass absoption coefficient [cm^2/g].

    """

    # LAST MODIFIED: Mengying Li 05/18/2017 data from contnm.f90 and processed in Matlab
    # calculate the continuum absorption coefficients of CO2
    # nu: wavenumber considered (cm-1)
    # P: total pressure (Pa)
    # T: temperature (K)
    # density: density of CO2 (g/cm3)
    # OUTPUT:
    # coeff_cont: mass continuum absorption coefficients of CO2 (cm^2/g)

    data_frgn = np.genfromtxt("data/profiles/frgnContm_CO2.csv", delimiter=",")
    nu_frgn = data_frgn[:, 0]  # wavenumber in cm-1
    coeff_frgn = data_frgn[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    cfac = data_frgn[:, 2]
    tdep = data_frgn[:, 3]
    trat = T / 246
    coeff_frgn *= cfac * trat ** tdep

    # compute R factor
    T0 = 296
    P0 = 1.013 * 1e5
    c_co2 = density / 44  # molar density of h2o, in unit of mol/cm3
    # c_tot = P / 8.314 / T / 1e6  # molar density of air, in unit of mol/cm3
    # co2_fac = c_co2 / c_tot
    RHOave = (P / P0) * (T0 / T)

    coeff_frgn *= RHOave  # co2_fac corrected 4/10/2018
    coeff_frgn *= 6.022 * 1e3  # unit cm2/mol (cm)-1 * 6.022*10**23*10**(-20)
    coeff_frgn *= c_co2  # unit of cm2/cm3 (cm)-1
    coeff_frgn /= density  # unit of cm2/g (cm)-1

    # apply radiation field
    coeff_cont = np.interp(nu, nu_frgn, coeff_frgn, left=0, right=0)
    RADFN = radiation_field(nu, T)

    # mass absorption coeffcient [cm^2/g]
    coeff_cont *= RADFN
    return coeff_cont


def absorptionContinuum_MTCKD_O3(nu, P, T, density):
    """
    Continuum absorption coeffcients of O3.

    Parameters
    ----------
    nu : array_like
        Wavenumber [cm^-1].
    P :
        Pressure [Pa].
    T :
        Temperature [K].
    density :

    Returns
    -------
    coeff_cont :
        Mass absoption coefficient [cm^2/g].

    """

    # LAST MODIFIED: Mengying Li 04/09/2018 data from contnm.f90 and processed in Matlab
    # calculate the continuum absorption coefficients of O3
    # nu: wavenumber considered (cm-1)
    # P: total pressure (Pa)
    # T: temperature (K)
    # density: density of O3 (g/cm3)
    # OUTPUT:
    # coeff_cont: mass continuum absorption coefficients of CO2 (cm^2/g)

    data_contm = np.genfromtxt("data/profiles/Contm_O3.csv", delimiter=",")
    nu_contm = data_contm[:, 0]  # wavenumber in cm-1
    c0_contm = data_contm[:, 1]
    c1_contm = data_contm[:, 2]
    c2_contm = data_contm[:, 3]

    c_o3 = density / 48  # molar density of O3, in unit of mol/cm3
    # c_tot = P / 8.314 / T / 1e6  # molar density of air, in unit of mol/cm3
    # o3_frac = c_o3 / c_tot
    # print (o3_frac)

    DT = T - 273.15
    contm = c0_contm + c1_contm * DT + c2_contm * DT ** 2
    # contm*=o3_frac
    contm *= 6.022 * 1e3  # unit cm2/mol (cm)-1 * 6.022e23 * 1e-20
    contm *= c_o3  # unit of cm2/cm3 (cm)-1
    contm /= density  # unit of cm2/g (cm)-1

    # apply radiation field
    coeff_cont = np.interp(nu, nu_contm, contm, left=0, right=0)
    RADFN = radiation_field(nu, T)

    # mass absorption coeffcient [cm^2/g]
    coeff_cont *= RADFN
    return coeff_cont


def radiation_field(nu, T):
    """
    The 'radiation field' for calculation of continuum coefficient.

    Parameters
    ----------
    nu :
    T :

    Returns
    -------
    RADFN :
        Radiation field

    """

    # times the 'radiation field' to get rid of (cm)-1 in the denominator
    # see cntnv_progr.f function RADFN(VI,XKT)
    XKT = T / 1.4387752  # 1.4387752 is a constant from phys_consts.f90
    RADFN = np.zeros(len(nu))
    for i in range(0, len(nu)):
        XVIOKT = nu[i] / XKT
        if XVIOKT <= 0.01:
            RADFN[i] = 0.5 * XVIOKT * nu[i]
        elif XVIOKT <= 10:
            EXPVKT = np.exp(-XVIOKT)
            RADFN[i] = nu[i] * (1.0 - EXPVKT) / (1.0 + EXPVKT)
        else:
            RADFN[i] = nu[i]
    return RADFN


def kd_integrate(x, y, method, partition):
    """
    Integration via the LBL or k-distribution methods.

    LBL or k-distribution method of calculating integration of y over x, when y
    varies a lot with x (e.g. kappa vs nu).

    Parameters
    ----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.
    method : {'LBL', 'k-distribution'}
        Integration method.
    partition : int
        Number of partitions in the k-distribution method.

    Returns
    -------
    integral : float
        The integrated value.

    """

    # INPUTS:
    # x: x-axis variable
    # y: a function of x
    # method='LBL' for LBL integration, method ='k-distribution' for k-distribution integration
    # partition: the # of partitions in k-distribution method
    # OUTPUTS:
    # integral: int_{x_min}^{x_max} y dx

    where_are_NaNs = np.isnan(y)  # replace NaN with zero
    y[where_are_NaNs] = 0
    delta = 1e-19 + (-min(y))  # avoid y goes below or equal to zero
    ka = y + delta
    if method == "LBL":
        integral = integrate.trapz(ka, x)
    elif method == "k-distribution":
        [hist, X] = np.histogram(ka, partition)
        bin_edge = np.delete(X, -1)
        pk = hist / len(ka)  # probability of ka
        gk = np.cumsum(pk)  # accumulated probability of ka
        integral = (max(x) - min(x)) * integrate.trapz(bin_edge, gk)
    integral -= delta * (max(x) - min(x))
    return integral


def Planck(nu, T):
    """
    Planck's law as a function of wavenumber [cm^-1].

    Planck's law (see equaton on page 453 of [1]).

    Parameters
    ----------
    nu : float or array_like
        Wavenumber [cm^-1].
    T : float or array_like
        Temperature [K].

    Returns
    -------
    Eb :
        Blackbody emission intensity density [W/(m^2 sr cm^-1)]

    References
    ----------
    [1] Mill and Coimbra, "Basic Heat and Mass Transfer"

    """

    h = 6.6261e-34  # Planck's constant [J s]
    kB = 1.3806485e-23  # Boltzmann constant [J / K]
    c = 299792458  # speed of light [m / s]
    C1 = 2 * h * c ** 2  # coefficient 1
    C2 = h * c / kB  # coefficient 2
    nu = nu * 100  # convert from [cm^-1] to [m^-1]

    # blackbody emission
    # equivalent to MATLAB dot calculations
    Eb_nu = C1 * nu ** 3 / (np.exp(C2 * nu / T) - 1)

    # convert to [W/(m^2 sr cm^-1)]
    Eb_nu *= 100
    return Eb_nu


def Mie(lam, radii, refrac):
    """
    Mie refractive.

    Parameters
    ----------
    lam : float
        Wavelength [um].
    radii : float
        Radii [um] of the particles.
    refrac : float
        Index of refraction (complex number).

    Returns
    -------
    Qext :
        Extinction efficiency.
    Qabs :
        Absoprtion efficiency.
    Qsca :
        Scattering efficiency.
    Qbsca :
        ???
    g :
        Assymetry parameter (optional).

    """

    # LAST MODIFIED: Mengying Li 06/27/2017
    # INPUTS:
    # nu: wavelength, um (scalar)
    # radii: radii of particles, um (scalar)
    # refrac: index of rafraction (complex number)
    # OUTPUT:
    # Qext,Qabs,Qsca: extinction/absorption/scattering efficiency
    # g: assmmetry parameter (optional)
    # WRITTEN BY: Mengying Li 06/20/2017 according to "A first course in atmospheric
    #            radiation" and "mie.py" provided by "Principles of Planetary Climate"

    size = 2 * np.pi * radii / lam  # size parameter
    Nlim = round(
        size + 4 * size ** (1 / 3) + 2
    )  # number of terms of infinite series, from Petty's P359
    an, bn = Mie_ab(size, refrac, Nlim + 1)
    # compute efficiencies and g
    Qext = 0
    Qsca = 0
    Qbsca = 0
    g = 0
    sn = an + bn  # an and bn are Mie scattering coefficients
    for n in range(1, int(Nlim + 1)):
        Qext += (2 * n + 1) * sn[n - 1].real
        Qsca += (2 * n + 1) * (abs(an[n - 1]) ** 2 + abs(bn[n - 1]) ** 2)
        Qbsca += (2 * n + 1) * (-1) ** n * (an[n - 1] - bn[n - 1])
        temp1 = an[n - 1] * an[n].conjugate() + bn[n - 1] * bn[n].conjugate()
        temp2 = an[n - 1] * bn[n - 1].conjugate()
        g += n * (n + 2) / (n + 1) * temp1.real + (2 * n + 1) / n / (n + 1) * temp2.real

    Qext *= 2 / size ** 2
    Qsca *= 2 / size ** 2
    Qabs = Qext - Qsca
    Qbsca = abs(Qbsca) ** 2 / size ** 2
    g *= 4 / size ** 2 / Qsca
    return Qext, Qabs, Qsca, Qbsca, g


def Mie_ab(size, refrac, Nlim):
    """
    Mie scattering and absorption.

    MIE scattering and absoprtion, calculated according to "MATLAB Functions
    for Mie Scattering and Absorption."

    Parameters
    ----------
    size : float
        Size parameter.
    refrac : float
        Index of refraction (complex number).
    Nlim :

    Returns
    -------
    an, bn : float
        Mie scatter coefficients.

    """

    # INPUTS:
    # size: size parameter (scalar)
    # refrac: index of rafraction (complex number)
    # OUTPUT:
    # an,bn: Mie scattering coefficients
    # WRITTEN BY: Mengying Li 06/22/2017 according to "MATLAB Functions for Mie
    #            Scattering and Absorption"

    n_all = np.arange(
        0, Nlim + 1, 1
    )  # number of modes (start with zero to count for n-1)
    n = n_all[1:]
    # nu = n+0.5 # to be used in Bessel function
    z = size * refrac
    m2 = refrac * refrac
    sqx = np.sqrt(0.5 * np.pi / size)
    sqz = np.sqrt(0.5 * np.pi / z)

    bx_all = jv(n_all + 0.5, size) * sqx
    bz_all = jv(n_all + 0.5, z) * sqz
    yx_all = yv(n_all + 0.5, size) * sqx
    hx_all = bx_all + yx_all * 1j

    bx = bx_all[1:]
    bz = bz_all[1:]
    # yx = yx_all[1:]
    hx = hx_all[1:]

    b1x = bx_all[0:-1]
    b1z = bz_all[0:-1]
    # y1x = yx_all[0:-1]
    h1x = hx_all[0:-1]

    ax = size * b1x - n * bx
    az = z * b1z - n * bz
    ahx = size * h1x - n * hx

    an = (m2 * bz * ax - bx * az) / (m2 * bz * ahx - hx * az)
    bn = (bz * ax - bx * az) / (bz * ahx - hx * az)
    return an, bn


# allow parallel computing
def aerosol_monoLam(inputs):
    """
    Aerosol monochromatic ???

    Parameters
    ----------
    inputs : (3,) array_like
        0) lam :
            Wavelength [um].
        1) refrac :
        2) r : array_like

    Returns
    -------
    Qsca :
    Qabs :
    g :

    """

    lam = inputs[0]
    refrac = inputs[1]
    r = inputs[2]
    Qsca = np.zeros(len(r))
    Qabs = np.zeros(len(r))
    g = np.zeros(len(r))
    for j in range(len(r)):
        Qext, Qabs[j], Qsca[j], Qbsca, g[j] = Mie(lam, r[j], refrac)
    return Qsca, Qabs, g


def aerosol():
    """
    Aerosol absoprtion/scattering coefficients and single albedo.

    Compute the absoprtion/scattering coefficients and single albedo for aerosols,
    according to [1].

    Parameters
    ----------
    None

    Returns
    -------
    None

    References
    ----------
    [1] Lubin 2002.

    """

    lam = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))
    data_aer = np.genfromtxt("data/profiles/aerosol_refraction.csv", delimiter=",")
    data_w = np.genfromtxt("data/profiles/water_refraction.csv", delimiter=",")

    real_dry = np.interp(lam, data_aer[:, 0], data_aer[:, 1])
    img_dry = np.interp(lam, data_aer[:, 0], data_aer[:, 2])
    real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
    img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])

    # -------------modify refraction index and size distribution according to RH-------------
    RH = np.arange(0, 110, 10)  # relative humidity
    rh_fac = np.asarray(
        [1.0, 1.0, 1.0, 1.031, 1.055, 1.090, 1.150, 1.260, 1.554, 1.851, 2.151]
    )  # last number is infered by M.Li
    real_intM = np.zeros([len(RH), len(lam)])
    img_intM = np.zeros([len(RH), len(lam)])

    r = np.concatenate(
        (np.arange(0.001, 0.3, 0.001), np.arange(0.3, 20.05, 0.1))
    )  # in um
    rm0 = np.asarray([0.135, 0.955])
    sigma0 = np.asarray([2.477, 2.051])  # * rh_fac[i]# in um
    n0 = np.asarray([1e4, 1])  # number/um-3
    Nr = np.zeros([len(RH), len(r)])  # number/um-3
    for i in range(0, len(RH)):
        # refraction index modification
        real_intM[i, :] = real_dry * rh_fac[i] ** (-3) + real_w * (
            1 - rh_fac[i] ** (-3)
        )
        img_intM[i, :] = img_dry * rh_fac[i] ** (-3) + img_w * (1 - rh_fac[i] ** (-3))
        # size modification
        rm = rm0 * rh_fac[i]
        sigma = sigma0  # *rh_fac[i] #* rh_fac[i]# in um
        dNr = (
            n0[0]
            / (np.sqrt(2 * np.pi) * np.log(sigma[0]))
            * np.exp(-0.5 * (np.log(r / rm[0]) / np.log(sigma[0])) ** 2)
        )
        dNr += (
            n0[1]
            / (np.sqrt(2 * np.pi) * np.log(sigma[1]))
            * np.exp(-0.5 * (np.log(r / rm[1]) / np.log(sigma[1])) ** 2)
        )
        Nr[i, :] = dNr / r  # same length as r, number/cm-4
    refrac_intM = real_intM + 1j * img_intM

    # create input list of args to parallel computation
    kappa_s = np.zeros([len(RH), len(lam)])
    kappa_a = np.zeros([len(RH), len(lam)])
    g_all = np.zeros([len(RH), len(lam)])
    for i in range(0, len(RH)):
        # args = [lam, refrac_intM[i, :], r, Nr[i, :]]
        list_args = []
        for j in range(0, len(lam)):
            args = [lam[j], refrac_intM[i, j], r]  # single value of lam,refrac and r
            list_args.append(args)

        # run in parallel
        pool = Pool()
        results = list(pool.map(aerosol_monoLam, list_args))
        pool.terminate()

        # re-organize the results from parallel computation
        for j in range(0, len(lam)):
            Qsca = results[j][0]
            Qabs = results[j][1]
            g = results[j][2]
            kappa_s[i, j] = integrate.trapz(Qsca * Nr[i, :] * np.pi * r ** 2, r)
            kappa_a[i, j] = integrate.trapz(Qabs * Nr[i, :] * np.pi * r ** 2, r)
            g_all[i, j] = (
                integrate.trapz(Qsca * g * Nr[i, :] * np.pi * r ** 2, r) / kappa_s[i, j]
            )  # without infer number of particles-- correct in getMixKappa function
    np.save("results/ks_aerosol", kappa_s)
    np.save("results/ka_aerosol", kappa_a)
    np.save("results/g_aerosol", g_all)


def cloud_efficiency(z):
    """
    Cloud absoprtion/scattering coefficents and single albedo.

    Compute the absoprtion/scattering coefficients, single albedo, and
    asymetry factor of clouds.

    Parameters
    ----------
    z : array (n_layer + 2,)
        Heights [m] of the layers.

    Returns
    -------
    None

    """

    n_layers = len(z) - 2

    # Last modified: Mengying Li 2/9/2018
    lam = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))
    lam0 = 0.4975
    data_w = np.genfromtxt("data/profiles/water_refraction.csv", delimiter=",")
    real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
    img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])
    refrac_w = real_w + 1j * img_w
    r = np.arange(0.1, 50, 0.1)  # in um
    # calculate the Qsca,Qabs,g for all combination of lam, refrac_w and r
    list_args = []
    for j in range(0, len(lam)):
        args = [lam[j], refrac_w[j], r]  # single value of lam,refrac and r
        list_args.append(args)

    # run in parallel
    pool = Pool()
    results = list(pool.map(aerosol_monoLam, list_args))
    pool.terminate()

    # re-organize the results from parallel computation
    Qsca, Qabs, g_M = [np.zeros([len(lam), len(r)]) for i in range(0, 3)]
    for j in range(0, len(lam)):
        Qsca[j, :] = results[j][0]
        Qabs[j, :] = results[j][1]
        g_M[j, :] = results[j][2]

    np.save("results/Qsca_{}layers".format(n_layers), Qsca)
    np.save("results/Qabs_{}layers".format(n_layers), Qabs)
    np.save("results/gM_{}layers".format(n_layers), g_M)
    print(Qsca.shape, Qabs.shape, g_M.shape)


# ----------------------------------------------------------------------------------
# compute absorption/scattering coefficents and single albedo, asymetry factor of clouds
def cloud(cld_model, z, kap):
    """
    Cloud absoprtion/scattering coefficents and single albedo.

    Compute the absoprtion/scattering coefficients, single albedo, and
    asymetry factor of clouds.

    Parameters
    ----------
    z : array (n_layer + 2,)
        Heights of the layers.
    kap : array
        Layers with clouds.

    Returns
    -------
    ks_cld, ka_cld, g_cld : array (n_lambda,)
        Cloud scattering and absorption coefficients (ks and ka), and asymmetry
        factors (g).

    """

    lam = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))  # wavelength [um]
    lam0 = 0.4975
    r = np.arange(0.1, 50, 0.1)  # in um

    # load pre-computed Qsca, Qabs, g_M
    n_layers = len(z) - 2
    Qsca = np.load("results/Qsca_{}layers.npy".format(n_layers))
    Qabs = np.load("results/Qabs_{}layers.npy".format(n_layers))
    g_M = np.load("results/gM_{}layers.npy".format(n_layers))

    ks_cld, ka_cld, g_cld = [np.zeros([len(z) - 1, len(lam)]) for i in range(0, 3)]

    # default cloud model (re=10, sig_e=0.1)
    if "default" in cld_model:  # cld_model=='default'):
        re = 10  # effective radius in um, Barker 2003
        sig_e = 0.1  # effective variance in um, Barker 2003
        Nr = r ** (1 / sig_e - 3) * np.exp(-r / re / sig_e)  # size distribution (gamma)
        ks, ka, g = [np.zeros(len(lam)) for i in range(0, 3)]
        for j in range(0, len(lam)):
            ks[j] = integrate.trapz(Qsca[j, :] * Nr * np.pi * r ** 2, r)
            ka[j] = integrate.trapz(Qabs[j, :] * Nr * np.pi * r ** 2, r)
            g[j] = (
                integrate.trapz(Qsca[j, :] * g_M[j, :] * Nr * np.pi * r ** 2, r) / ks[j]
            )
        dz_cld = z[kap[-1] + 1] - z[kap[0]]  # in m
        ke_cld_ref = 1.0 / (dz_cld * 100)  # in cm-1, COD by default = 1.0
        ratio_cld = ke_cld_ref / np.interp(lam0, lam, ka + ks)
        for i in range(len(kap)):
            ks_cld[kap[i], :] = ks * ratio_cld
            ka_cld[kap[i], :] = ka * ratio_cld
            g_cld[kap[i], :] = g
    else:  # cloud model of CIRC cases
        cld_file = (
            "results/CIRC/" + cld_model + "_input&output/cloud_input_" + cld_model + ".txt"
        )
        cld_input = np.genfromtxt(
            cld_file, skip_header=2
        )  # layer number, CF, LWP, IWP,re_liq, re_ice
        reS = cld_input[:, 4]
        LWP = cld_input[:, 2]
        # create input list of args to parallel computation
        list_args = []
        for i in range(len(kap)):
            if "re10" in cld_model:
                re = 10
                sig_e = 0.1
            else:
                re = reS[kap[i]]
                sig_e = 0.014  # spectral diserpation of 0.12 for all CIRC cases
            Nr = r ** (1 / sig_e - 3) * np.exp(
                -r / re / sig_e
            )  # size distribution (gamma)
            x_frac = integrate.trapz(
                Nr * 4 / 3 * np.pi * r ** 3, r
            )  # volume fraction of water in air.
            ratio_cld = (
                LWP[kap[i]] / (z[kap[i] + 1] - z[kap[i]]) / 100
            ) / x_frac  # z should in cm
            ks, ka, g = [np.zeros(len(lam)) for i in range(0, 3)]
            for j in range(0, len(lam)):
                ks[j] = integrate.trapz(Qsca[j, :] * Nr * np.pi * r ** 2, r)
                ka[j] = integrate.trapz(Qabs[j, :] * Nr * np.pi * r ** 2, r)
                g[j] = (
                    integrate.trapz(Qsca[j, :] * g_M[j, :] * Nr * np.pi * r ** 2, r)
                    / ks[j]
                )
            ks_cld[kap[i], :] = ks * ratio_cld
            ka_cld[kap[i], :] = ka * ratio_cld
            g_cld[kap[i], :] = g

    # corrected for LWP already
    return ks_cld, ka_cld, g_cld


def rayleigh_kappa_s(lam, T, P, N, vmr_co2, vmr_h2o):
    """
    Refractive index of air.

    Refractive index of air (see Ciddar's formula).

    Parameters
    ----------
    lam :
        Wavelength [um].
    T :
        Temperature [K].
    P :
        Pressure [Pa].
    N :
        Number density [unit/cm^3] of air molecules.
    vmr_co2, vmr_h2o :
        Volumetric mixing ratio of CO2 and H2O.

    Returns
    -------
    kappa_s :
        Scattering coefficient [cm^-1].
    m_r :
        Refractive index of air.

    """

    # LAST MODIFIED: Mengying Li 11/09/2017
    # INPUTS:
    # lam: wavelength, in um   # T: temperature, in K
    # P: pressure, in Pa       # N: number density of air molecules, in unit/cm3
    # vmr_co2/vmr_h2o: volume mixing ratio of co2 and h2o, unit of 1 e.g.400/10**6
    # constants for water vapor content
    # OUTPUTS:
    # kappa_s: scattering coefficient, cm-1
    # m_r: refractive index of air.

    vmr_co2 *= 1e6  # in unit of ppm
    w0, w1, w2, w3 = 295.235, 2.6422, -0.03238, 0.004028
    # constants for temerature scale and CO2 content
    k0, k1, k2, k3 = 238.0185, 5792105, 57.362, 167917
    # start of calculation
    S = 1 / lam ** 2  # lam in um
    # refractivity of standard air
    n_as = 1e-8 * (k1 / (k0 - S) + k3 / (k2 - S)) + 1
    # refractivity with variable CO2 content
    n_axs = (n_as - 1) * (1 + 5.34 * 1e-7 * (vmr_co2 - 450)) + 1
    # refractivity of water vapor at 20 C and 1333 Pa
    n_ws = 1.022 * 1e-8 * (w0 + w1 * S + w2 * S ** 2 + w3 * S ** 3) + 1
    Ma = 1e-3 * (28.9635 + 12.011 * 1e-6 * (vmr_co2 - 400))  # kg/mol
    Mw = 0.018015  # molar mass of water vapor, kg/mol
    R = 8.314510  # gas constant J/mol K
    rho_a = (
        (1 - vmr_h2o) * P * Ma / (Z_air(T, P, vmr_h2o) * R * T)
    )  # *(1-vmr_h2o*(1-Mw/Ma))
    rho_w = vmr_h2o * P * Ma / (Z_air(T, P, vmr_h2o) * R * T)  # *(1-vmr_h2o*(1-Mw/Ma))
    rho_axs = 101325 * Ma / (Z_air(15 + 273.15, 101325, 0) * R * (15 + 273.15))
    rho_ws = 1333 * Ma / (Z_air(20 + 273.15, 1333, 1) * R * (20 + 273.15))
    # prop method of calculating the refractive index
    m_r = 1 + (rho_a / rho_axs) * (n_axs - 1) + (rho_w / rho_ws) * (n_ws - 1)

    # find the scattering coefficient with the refractive index
    # according to Penndorf 1995 with some modifications
    N = N * 1e-12  # number density unit/um3
    # method 1 calculating kappa_s
    # Ns=2.54743*10**19*10**(-12) # standard number density
    # kappa_s = 8 * np.pi ** 3 / 3 * (m_r ** 2 - 1) ** 2 / lam ** 4 / N  # in unit of um-1
    # rho_n = 0.035  # depolarization factor
    # kappa_s *= (6 + 3 * rho_n) / (6 - 7 * rho_n)
    # method 2 calculating kappa_s
    # kappa_s=32*np.pi**3*(m_r-1)**2/3/lam**4/N # P73 Thomas & Stamnes book
    # kappa_s *= 1e4  # in unit of cm-1
    # method 3 calculating kappa_s, from code lblrtm.f
    nu = 1e4 / lam
    conv_cm2mol = 1e-20 / (2.68675 * 1e-1 * 1e5)
    xnu = nu / 1e4
    wtot = N  # wtot is the total number of molecules in the calculation
    ks_4 = xnu ** 4 / (9.38076 * 1e2 - 10.8426 * xnu ** 2) * (wtot * conv_cm2mol)
    kappa_s = ks_4 * 1e12

    return kappa_s, m_r


def Z_air(T, P, vmr_h2o):
    """
    Density of moist air.

    Density of moist air (see Equation 4 of [1]).

    Parameters
    ----------
    T :
        Temperature [K].
    P :
        Pressure [Pa].
    vmr_h2o :
        Volumetric mixing ratio.

    Returns
    -------
    Z :
        Molar mass of dry air, considering CO2 content.

    References
    ----------
    [1] Ciddar paper

    """

    # Inputs: temperature T in K, pressure P in Pa
    #        vmr_co2/vmr_h2o: volume fraction of co2 and h2o in unit of 1
    # Outputs: compressibility of air
    # constants to calculate compressibility

    a0, a1, a2 = 1.58123 * 1e-6, -2.9331 * 1e-8, 1.1043 * 1e-10
    b0, b1 = 5.707 * 1e-6, -2.051 * 1e-8
    c0, c1 = 1.9898 * 1e-4, -2.376 * 1e-6
    d = 1.83 * 1e-11
    e = -0.765 * 1e-8

    # convert temperature from [K] to [C].
    t = T - 273.15

    # molar mass of dry air with consideration of Co2 content
    Z = (
        1
        - (P / T)
        * (
            a0
            + a1 * t
            + a2 * t ** 2
            + (b0 + b1 * t) * vmr_h2o
            + (c0 + c1 * t) * vmr_h2o ** 2
        )
        + (P / T) ** 2 * (d + e * vmr_h2o ** 2)
    )
    return Z


def Planck_lam(lam, T):
    """
    Planck's law as a function of wavelength [um].

    Uses equation on page 453 of BHMT.

    Parameters
    ----------
    lam : float or array_like
        Wavelength [um].
    T : float or array_like
        Temperature [K].

    Returns
    -------
    Eb_lam :
        Blackbody emission intensity density [W/(m^2 sr um)].

    References
    ----------
    [1] Mill and Coimbra, "Basic Heat and Mass Transfer"

    """

    # LAST MODIFIED: Mengying 11/11/2017
    # INPUTS:
    # lam: wavelength, um (vector or scalar)
    # T: temperature, K (vector or scalar)
    # OUTPUT:
    # Eb: blackbody emission intensity density (vector or scalar), W/(m**2 sr um)
    # WRITTEN BY: Mengying Li 11/11/2017 using Eq on P453 of Mills&Coimbra's BHMT book

    h = 6.6261 * 1e-34  # Planck's constant, J s
    kB = 1.3806485 * 1e-23  # Boltzmann constant, J/K
    c = 299792458  # speed of light, m/s
    C1 = 2 * h * c ** 2
    C2 = h * c / kB
    lam = lam / 1e6  # change unit to m
    Eb_lam = C1 / lam ** 5 / (np.exp(C2 / lam / T) - 1)
    Eb_lam /= 1e6  # in unit of W/m2 um sr
    return Eb_lam


def aerosol_CIRC(model, lam, z):
    """
    Aerosol properties for CIRC cases.

    Parameters
    ----------
    model : str
    lam : array_like
        Wavelengths [um].
    z : array_like
        Height of the node points.

    Returns
    -------
    ka_aer_M :
    ks_aer_M :
    g_aer_M :
        Assymetry factor.
    """

    filename = "results/CIRC/{}_input&output/aerosol_input_{}.txt".format(model, model)
    A = np.genfromtxt(filename, skip_header=3, max_rows=1)
    data = np.genfromtxt(filename, skip_header=5)
    N_layer = data.shape[0] - 1
    ka_aer_M, ks_aer_M, g_aer_M = [
        np.zeros([N_layer + 1, len(lam)]) for i in range(0, 3)
    ]
    for i in range(1, N_layer + 1):
        dz = (z[i + 1] - z[i]) * 100  # in cm
        if data[i, 1] > 0:
            tau = data[i, 1] * lam ** (-A)
            ke_aer = tau / dz  # extinction coeff
            ks_aer_M[i, :] = ke_aer * data[i, 2]  # SSA is not a function of lam
            ka_aer_M[i, :] = ke_aer - ks_aer_M[i, :]
            g_aer_M[i, :] = data[i, 3]  # g is not a function of lam
    return [ka_aer_M, ks_aer_M, g_aer_M]


def surface_albedo(nu, surface):
    """Get surface albedo for different materials.

    Parameters
    ----------
    nu: (N_nu,) array_like
        spectral grid in wavenumber [cm-1].
    surface: string
        considered surface type, CIRC cases or PV or CSP
    Returns
    -------
    rho_s: (N_nu,N_deg) array_like
        spectral surface albedo.
    """
    lam = 1e4 / nu
    if 'case' in surface:
        filename = "results/CIRC/" + surface + "_input&output/sfcalbedo_input_" + surface + ".txt"
        data = np.genfromtxt(filename, skip_header=6)
        rho_s = np.interp(nu, data[:, 0], data[:, 1])
    if surface == 'PV':
        filename = "Profile data/Reflectance of PV.txt"
        data = np.genfromtxt(filename, skip_header=0)
        rho_s = np.interp(lam, data[:, 0] / 1e3, data[:, 1] / 1e2)  # data in nm and %
    if (surface == 'CSP'):
        file = "Profile data/Reflectance of CSP.txt"
        data = np.genfromtxt(file, skip_header=1)
        rho_s1 = np.interp(lam, data[:, 0] / 1e3, data[:, 1])  # data in nm and %
        rho_s2 = np.interp(lam, data[:, 0] / 1e3, data[:, 2])  # data in nm and %
        rho_s3 = np.interp(lam, data[:, 0] / 1e3, data[:, 3])  # data in nm and %
        rho_s = np.concatenate((np.vstack(rho_s1), np.vstack(rho_s2), np.vstack(rho_s3)), axis=1)
    return rho_s