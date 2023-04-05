"""
Longwave (LW) specific code for the LBL model.

Author: Mengying Li
"""

import numpy as np
import copy
from . import Planck
from scipy.special import expn  # exponential integral E_n


__all__ = ["monoFlux_expInt_scatter", "monoFluxN", "modifiedFs"]


def monoFlux_expInt_scatter(
    N_layer, nu, ka_M, ks_M, g_M, ta, z, alpha_s, F_dw_os, rh_0
):
    """
    Monochromatic flux density of N Layers.

    Parameters
    ----------
    N_layer : int
        Number of atmosphere layers.
    nu : (M,) array_like
        Wavenumbers [cm^-1].
    ka_M : (N + 1, M) array_like
        Absorption coefficient of the layers and wavenumbers.
    ks_M : (N + 1, M) array_like
        Scatter coefficient of the layers and wavenumbers.
    g_M : (N + 1, M) array_like
        Asymetry factor of the layers and wavenumbers.
    ta : array_like (N_layer,)
        Average temperature [K] of each layer (including the surface)
    z :
        Height [m] of the layer boundaries.
    alpha_s :
        ???
    F_dw_os : float
        Downward outerspace intensity [W/(m^2 sr)] from space.
    rh_0 :
        Surface relative humidity [???].

    Returns
    -------
    F_net_mono, F_dw_mono, F_uw_mono : (N + 1, M)
        Net, downwelling, and upwelling flux density [W/(m^2 cm^-1)] of the N layers
        and M wavenumbers.

    """

    # INPUTS:
    # N_layer: number of atmosphere layers
    # nu: wavelength vector, in cm-1
    # ka_M/ks_M: absorption/scattering coefficient of all layers, #N_layer lists of [coeff of all nu].
    # ta: average temperature of each layer including surface,vector, K
    # z: height of layer boudaries, m
    # F_dw_os: downwarding intensity from space, vector, W/m2 sr
    # rh_0: surface relative humidity

    # OUTPUTS:
    # F_net_mono,F_dw_mono,F_uw_mono: net, downward, upward flux density of each layer at each wavelengh,
    #                                (N_layer+1)*len(nu) matrix, W/(m2 cm-1)
    # WRITTEN BY: Mengying Li 06/28/2017

    # calculate and save variables as numpy 2D array
    abs_coeff = np.zeros((N_layer + 1, len(nu)))  # volumetric
    sca_coeff = np.zeros((N_layer + 1, len(nu)))  # volume
    ext_coeff = np.zeros((N_layer + 1, len(nu)))  # volume
    g_asy = np.zeros((N_layer + 1, len(nu)))
    albedo = np.zeros((N_layer + 2, len(nu)))  # volume
    A_star = np.zeros((N_layer + 2, len(nu))) + 1  # A*=1 for surfaces, boundary
    Ib = np.zeros((N_layer + 2, len(nu)))

    for i in range(1, N_layer + 1):
        abs_coeff[i, :] = ka_M[i, :]
        sca_coeff[i, :] = ks_M[i, :]
        g_asy[i, :] = g_M[i, :]
        ext_coeff[i, :] = sca_coeff[i, :] + abs_coeff[i, :]
        albedo[i, :] = sca_coeff[i, :] / ext_coeff[i, :]

        # delta_M scaling for anisotropic scattering
        ext_coeff[i, :] *= 1 - albedo[i, :] * g_asy[i, :]
        albedo[i, :] *= (1 - g_asy[i, :]) / (1 - albedo[i, :] * g_asy[i, :])

        # use scaled extinction coefficient (based on z [m])
        A_star[i, :] = 4 * ext_coeff[i, :] * (z[i + 1] - z[i]) * 1e2
        Ib[i, :] = Planck(nu, ta[i]) * np.pi

    Ib[0, :] = Planck(nu, ta[0]) * np.pi * alpha_s
    Ib[N_layer + 1, :] = F_dw_os * np.pi  # outer space irradiance
    albedo[0, :] = 1 - alpha_s  # surface albedo

    # calculate normal optical depth and Delta optical depth
    # inside Earth has od=-inf, outer space has od=+inf
    #
    # (N + 3, M):
    #   - N + 1 optical depth boundary
    #   - 1 with -inf
    #   - 1 with +inf
    #   - ==> N + 3 total
    #
    od = np.zeros((N_layer + 3, len(nu)))
    od[0, :] = np.zeros(len(nu)) - 1e39  # -inf od
    od[N_layer + 2, :] = np.zeros(len(nu)) + 1e39  # +inf od
    D_od = np.zeros((N_layer + 1, len(nu)))
    for i in range(1, N_layer + 1):  # i=[1....,18]
        D_od[i, :] = ext_coeff[i, :] * (z[i + 1] - z[i]) * 1e2  # unit of 1
        od[i + 1, :] = od[i, :] + D_od[i, :]

    # compute exp3 of optDepth[i,:]-optDepth[j,:] to save computation cost (for i>j)
    exp3 = []  # size (N_layer+3)*(N_layer+3)
    for i in range(N_layer + 3):
        expi = []
        for j in range(0, N_layer + 3):
            expj = expn(3, abs(od[i, :] - od[j, :]))
            expi.append(expj)
        exp3.append(expi)
    # print ("Finish calculating exp3.")

    # compute transfer factors
    # initialize size of (N_layer+2)*(N_layer+2), each with len(nu) vector
    Fs = [[np.zeros(len(nu)) for i in range(N_layer + 2)] for i in range(N_layer + 2)]
    # compute inter transfer factors
    for i in range(N_layer + 2):
        for j in range(N_layer + 2):
            if i != j:
                Fs[i][j] = (
                    2 * exp3[j][i + 1]
                    + 2 * exp3[j + 1][i]
                    - 2 * exp3[j][i]
                    - 2 * exp3[j + 1][i + 1]
                )
                Fs[i][j] /= A_star[i, :]
            else:
                Fs[i][j] = 1 - 0.5 / (od[i + 1, :] - od[i, :]) * (
                    1 - 2 * exp3[i][i + 1]
                )
                if i == 0 or i == N_layer + 1:
                    Fs[i][j] = np.zeros(len(nu))  # for Earth and outer space

    # use modFs to calculate Js and Gs
    modFs = modifiedFs(Fs, albedo)
    del Fs

    Js = np.zeros((N_layer + 2, len(nu)))
    Gs = np.zeros((N_layer + 2, len(nu)))
    for i in range(N_layer + 2):
        for j in range(N_layer + 2):
            Gs[i, :] += modFs[i][j] * Ib[j, :]
        Js[i, :] = (1 - albedo[i, :]) * Ib[i, :] + albedo[i, :] * Gs[i, :]
    del modFs

    # calculate downwelling flux from Js and re-calculate Fs
    F_dw_mono = np.zeros((N_layer + 2, len(nu)))
    # F_dw_mono[N_layer + 1, :] = Js[N_layer + 1, :]  # outer space
    for n in range(1, N_layer + 2):  # n = 1,2... N_layer
        fs = np.zeros((N_layer + 2, len(nu)))
        for j in range(n, N_layer + 2):  # j =  n,...N_layer
            fs[j, :] = 2 * exp3[j][n] - 2 * exp3[j + 1][n]
            # fs[j, :] = 2 * expn(3, abs(od[j, :] - od[n, :])) - 2 * expn(3, abs(od[j+1, :] - od[n, :]))
            F_dw_mono[n, :] += fs[j, :] * Js[j, :]

    # calculate upwelling flux from Js and re-calculate Fs
    F_uw_mono = np.zeros((N_layer + 2, len(nu)))
    F_uw_mono[1, :] = Js[0, :]  # surface flux
    for n in range(2, N_layer + 2):
        fs = np.zeros((N_layer + 2, len(nu)))
        for j in range(0, n):  # j=1,... n-1
            fs[j, :] = 2 * exp3[j + 1][n] - 2 * exp3[j][n]
            # fs[j, :] = 2 * expn(3, abs(od[j+1, :] - od[n, :])) - 2 * expn(3, abs(od[j, :] - od[n, :]))
            F_uw_mono[n, :] += fs[j, :] * Js[j, :]
    F_net_mono = F_uw_mono - F_dw_mono

    return F_net_mono, F_dw_mono, F_uw_mono


def monoFluxN(args):
    """
    Monochromatic radiosity of the N layers.

    Parameters
    ----------
    args : list
        albedo : (N + 2,) array_like
            Single scattering albedo.
        Ib : (N + 2,) array_like
            Planck's emissive flux [W/(m^2 cm^-1)]
        Fs: (N + 2, N + 2) array_like
            The transfer factor matrix.

    Returns
    -------
    J_v: array_like (N,)
        Monochromatic radiosity of the N layers.
    """

    # unpack inputs
    albedo = args[0]
    eps = 1 - albedo
    Ib = args[1]
    Fs = args[2]  # F matrix (N+1)*(N+1)

    # construct matrix A and b
    N = len(Ib)
    mtx_A = np.zeros([2 * N, 2 * N])
    mtx_A[0:N, N:] = Fs
    mtx_A[N:, 0:N] = np.diag(albedo, 0)
    b_v = np.zeros([2 * N, 1])
    b_v[N:, 0] = eps * Ib

    # compute radiosity and irradiance vector
    mtx_I = np.identity(2 * N)  # identity matrix
    y_v = np.linalg.solve(mtx_I - mtx_A, b_v)  # solve (mtx_I-mtx_A)y_v=b_v
    J_v = np.hstack(y_v[N:])  # radiosity,horizontal vector
    return J_v


def modifiedFs(Fs, rho):
    """
    Modified transfer factors of the N layers.

    Use the plating algorithm to compute modified transfer factors for
    scattering medium.

    Parameters
    ---------
    Fs : (N + 2, N + 2) array_like
        Transfer factor of non-scattering medium.
    rho : (N + 1,) array_like
        Single albedo of the layers.

    Returns
    -------
    modFs :
        Modified transfer factors.
    """

    # Use plating algorithm to compute modified transfer factors for scattering medium
    # Fs: transfer factor of non-scattering medium, list of (N_layer+2)*(N_layer+2), each element is 1*len(nu) array
    # rho: single albedo, (N_layer+1)*len(nu) matrix

    N_layer = len(Fs) - 2
    modFs = copy.deepcopy(Fs)
    newFs = copy.deepcopy(Fs)
    del Fs

    for k in range(1, N_layer + 1):  # plating only gas layers: 1,2,...N_layer
        # print ('Plating layer '+str(k))
        rho_k = rho[k, :]  # single albedo of k-th gas layer
        eps_k = 1 - rho_k
        D = 1 - rho_k * modFs[k][k]
        for i in range(0, N_layer + 2):
            for j in range(0, N_layer + 2):
                if i != k:
                    if j != k:
                        newFs[i][j] = (
                            modFs[i][j] + rho_k * modFs[i][k] * modFs[k][j] / D
                        )
                    elif j == k:
                        newFs[i][j] = modFs[i][j] * eps_k / D
                elif i == k:
                    if j != k:
                        newFs[i][j] = modFs[i][j] * eps_k / D
                    elif j == k:
                        newFs[i][j] = modFs[i][j] * eps_k * eps_k / D
        modFs = copy.deepcopy(newFs)
    return modFs