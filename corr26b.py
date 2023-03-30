"""Attempt to reproduce corr26b from Li et al. 2017.

A fair amount of code is copied from sky_emissivity GitHub repo,
some of which is imported from JYJ (Yuanjie) code.
"""

import os
import time
import pvlib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from scipy.io import loadmat

from main import get_pw, get_esky_c, li_lw, CORR26A, compute_mbe
from pcmap_data_funcs import get_asos_stations

from constants import SIGMA, SURFRAD, SURF_COLS, SURF_ASOS, SURF_SITE_CODES


# def pw_tdp(t_a, rh):
#     # the function of calculating Pw(Pa) and Tdp(C)
#     pw = 610.94 * (rh / 100) * np.exp(17.625 * (t_a - 273.15) / (t_a - 30.11))
#     tdp = (243.04 * np.log(pw / 610.94)) / (17.625 - np.log(pw / 610.94))
#     return pw, tdp


def int_func(x, t):
    c1 = 3.7418e8  # W um^4 / m^2 ~> 2*pi*h*c^2
    c2 = 1.4389e4  # um K         ~> hc/k_B
    out = (c1 * x ** -5) / (np.exp(c2 / (x * t)) - 1)
    return out


def get_tsky(t, ir_mea):
    # function to iteratively solve for T_sky
    l1 = 3  # um
    l2 = 50  # um
    c1 = 3.7418e8  # W um^4 / m^2 ~> 2*pi*h*c^2
    c2 = 1.4389e4  # um K         ~> hc/k_B
    y = lambda x: (c1 * np.power(x, -5)) / (np.exp(c2 / (x * t)) - 1)
    out = integrate.quad(func=y, a=l1, b=l2)
    result = out[0]  # output: result, error, infodict(?), message, ...
    delta = ir_mea - result

    delta_thresh = 0.1  # SET: within 0.1 W/m^2 of measured DLW is good enough
    if (delta > 0) & (
            np.abs(delta) > delta_thresh):  # if delta>0 guessed Tsky should increase
        t = t * 1.01
        result, t = get_tsky(t, ir_mea)
    elif (delta < 0) & (np.abs(delta) > delta_thresh):
        t = t * 0.99
        result, t = get_tsky(t, ir_mea)
    return result, t


def get_pw_esky(rh, t):
    pw = 610.94 * (rh / 100) * (np.exp(17.625 * (t - 273.15) / (t - 30.11)))
    # (22a) -- daytime clear-sky model Pw in hPa
    esky = 0.598 + 0.057 * np.sqrt(pw / 100)
    return esky, pw


def import_asos_yr(yr):
    """Import ASOS station data for a given year to a local directory.
    References get_asos_stations() from pcmap_data_funcs.py

    Parameters
    ----------
    yr : int
        Specify year to import

    Returns
    -------
    None
    """
    df = pd.DataFrame.from_dict(SURF_ASOS, orient="index")
    local_dir = os.path.join("data", f"asos_{yr}")
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    get_asos_stations(year=yr, local_dir=local_dir,
                      usaf=df.usaf.values, wban=df.wban.values)
    return None


def process_site(site, yr="2012"):
    """
    Gather data for a given SURFRAD site in a given year.
    Perform quality control checks, format, and sample data.
    Requires access to hard drive.
    Processed data is saved as csv to data/SURFRAD folder.

    Parameters
    ----------
    site : string
    yr : string, optional

    Returns
    -------
    None
    """
    start_time = time.time()
    site_name = SURFRAD[site]["name"]
    lat = SURFRAD[site]["lat"]
    lon = SURFRAD[site]["lon"]
    alt = SURFRAD[site]["alt"]  # m
    directory = os.path.join(
        "/Volumes", "Lysha_drive", "Archive",
        "proj_data", "SURFRAD", site_name
    )
    all_years = os.listdir(directory)
    keep_cols = [
        'zen', 'dw_solar', 'qc1', 'direct_n', 'qc3', 'diffuse', 'qc4',
        'dw_ir', 'qc5', 'temp', 'qc16', 'rh', 'qc17', 'pressure', 'qc20'
    ]  # narrow down columns from SURF_COLNAMES
    if yr in all_years:
        folder = os.path.join(directory, yr)
        lst = os.listdir(folder)
        lst.sort()
        tmp = pd.DataFrame()
        expected_columns = len(SURF_COLS)
        for f in lst:  # import data by day and concatenate to `tmp`
            filename = os.path.join(folder, f)
            try:
                df = pd.DataFrame(
                    np.loadtxt(filename, skiprows=2), columns=SURF_COLS
                )
                if len(df.columns) == expected_columns:
                    df['TS'] = pd.to_datetime(
                        {'year': df.yr, 'month': df.month, 'day': df.day,
                         'hour': df.hr, 'minute': df.minute}
                    )
                    df = df.set_index('TS')
                    df = df[keep_cols]
                    tmp = pd.concat([tmp, df])
                else:
                    print(
                        f"{filename} does not have expected number of columns."
                    )
            except pd.errors.ParserError as e:
                print(f"Error: {e}")
        df = tmp.copy()  # switch back to df
        print("Data collected.", time.time() - start_time)

        # Do some clean-up
        df = df[
            (df.qc1 == 0) & (df.qc3 == 0) &
            (df.qc4 == 0) & (df.qc5 == 0) &
            (df.qc16 == 0) & (df.qc17 == 0) &
            (df.qc20 == 0)
        ]
        df = df[
            ['zen', 'direct_n', 'diffuse', 'dw_ir', 'dw_solar', 'temp',
             'rh', 'pressure']]  # reduce data columns
        df['t_a'] = df.temp + 273.15  # convert celsius to kelvin

        # remove negative GHI and DNI values
        df = df.loc[(df.dw_solar > 0) & (df.direct_n > 0)]

        # apply clear sky analysis
        location = pvlib.location.Location(lat, lon, altitude=alt)
        cs_out = location.get_clearsky(df.index)
        col_rename = {
            'direct_n': 'DNI_m', 'dw_solar': 'GHI_m',
            'ghi': 'GHI_c', 'dni': 'DNI_c'
        }
        df = df.merge(cs_out, how='outer', left_index=True, right_index=True)
        df = df.rename(columns=col_rename)
        print("QC and clear sky applied.", time.time() - start_time)

        # Apply clear sky period filter
        df = find_clearsky(df)
        # need to apply clear sky filter before data is sampled
        print("Clear sky filter applied.", time.time() - start_time)

        # Reduce sample size TODO remove later(?) (orig 1%)
        df = df.sample(frac=0.05, random_state=96)

        # Determine T_sky values, and correct for 3-50 micron range
        temp = df.t_a.values
        dwir = df.dw_ir.values
        t_sky = []
        dlw = []
        for i in range(df.shape[0]):
            ir, tsky = get_tsky(temp[i], dwir[i])
            ir_act = SIGMA * tsky ** 4  # determine actual DLW using sky temp
            t_sky.append(tsky)
            dlw.append(ir_act)
        df['t_sky'] = t_sky
        df['lw_s'] = dlw
        # df['kT'] = df.GHI_m / df.GHI_c
        # kTc = df.GHI_m / df.GHI_c
        # kTc[kTc > 1] = 1
        # df['kTc'] = kTc
        print("T_sky determined.", time.time() - start_time)

        filename = os.path.join("data", "SURFRAD", f"{site}_{yr}.csv")
        df.to_csv(filename)
    else:
        print(f"Error: {yr} is not available for {site_name}")

    dt = time.time() - start_time
    print(f"Completed {site} in {dt:.0f}s")
    return None


def join_surfrad_asos(site="BON"):
    # assumes SURFRAD data has already been processed, assumes 2012
    usaf = SURF_ASOS[site]["usaf"]
    wban = SURF_ASOS[site]["wban"]
    # get SURFRAD data (after processing)
    filename = os.path.join("data", "SURFRAD", f"{site}_2012.csv")
    tmp = pd.read_csv(filename, index_col=0, parse_dates=True)
    # site = site.loc[site.location == SURFRAD["BON"]["name"]]
    tmp.sort_index(inplace=True)
    tmp = tmp.tz_localize("UTC")
    # get ASOS station data
    filename = os.path.join("data", "asos_2012", f"{usaf}-{wban:05}-2012.csv")
    df = pd.read_csv(filename, skiprows=1, index_col=0, parse_dates=True)
    df = df[['CF2', 'CBH2', 'TEMP', 'VSB', 'DEWP', 'SLP', 'PERCP']]
    mask = df.index.duplicated(keep="first")
    df = df[~mask].sort_index()  # drop duplicate indices
    df = df.rename(columns={"CF2": "cf"})

    # merge
    # TODO try other methods of merging (strict bfill, interpolation)
    df = pd.merge_asof(
        tmp, df, left_index=True, right_index=True,
        tolerance=pd.Timedelta("1h"), direction="nearest"
    )
    df["asos_t"] = df.TEMP + 273.15  # K
    df["pw"] = get_pw(df.t_a, df.rh) / 100  # hPa
    df["esky_c"] = get_esky_c(df.pw)
    df["lw_c"] = df.esky_c * SIGMA * np.power(df.t_a, 4)

    # drop rows with missing values in parameter columns
    # x = df.shape[0]
    df.dropna(subset=["t_a", "rh", "cf", "lw_c", "lw_s"], inplace=True)
    # print(x, df.shape[0])
    return df


def custom_fit(df):
    df = df.assign(
        t1=-1 * df.lw_c * df.cf,
        t2=SIGMA * (df.t_a ** 4) * df.cf * (df.rh / 100),
        y=df.lw_s - df.lw_c
    )
    # inexact fit, only solving for two params

    b = df.lw_c.to_numpy()
    train_x = df[["t1", "t2"]].to_numpy()
    train_y = df.y.to_numpy()

    model = LinearRegression(fit_intercept=False)
    model.fit(train_x, train_y)
    print(model.coef_, model.intercept_)
    rmse = np.sqrt(mean_squared_error(train_y, model.predict(train_x)))
    print(f"{rmse:.2f}")

    y_true = train_y + b
    y_pred = model.predict(train_x) + b
    return model, y_true, y_pred, rmse


def plot_fit(site, coefs, y_true, y_pred, rmse):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True, alpha=0.3)
    ax.scatter(y_true, y_pred, alpha=0.05)
    ax.axline((300, 300), slope=1, c="0.1", ls="--")
    c1, c3 = coefs
    ax.text(0.1, 0.9, s=f"c1={c1:.3f}, c3={c3:.3f}",
            transform=ax.transAxes, backgroundcolor="1.0")
    ax.text(0.1, 0.8, s=f"RMSE={rmse:.2f} W/m$^2$",
            transform=ax.transAxes, backgroundcolor="1.0")
    ax.set_xlabel("LW measured [W/m$^2$]")
    ax.set_ylabel("Modeled [W/m$^2$]")
    ax.set_title(f"{site}")
    ax.set_xlim(100, 600)
    ax.set_ylim(100, 600)
    # plt.show()
    filename = os.path.join("figures", f"{site}_fit.png")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    return None


def retrieve_asos_site_info(site):
    filename = os.path.join("data", "isd_history.csv")
    asos = pd.read_csv(filename)
    asos_site = asos.loc[(asos.USAF == str(SURF_ASOS[site]["usaf"])) &
                         (asos.WBAN == SURF_ASOS[site]["wban"])].iloc[0]
    return asos_site


def cs_metrics_1to5(obs, clr, thresholds="ghi"):
    """
    Supporting function to determine clear sky periods using thresholds set
    in Section 3.2 in Li et al. 2017

    Parameters
    ----------
    obs : series, array-like
        Measured values in a sliding window period
    clr : series, array-like
        Clear ksy modeled values in a siding window period
    thresholds : ["ghi", "dni"]
        Indicate whether to use the GHI or DNI thresholds

    Returns
    -------
    metrics_sum : int
        For each condition that passes the threshold, metrics_sum is
        incremented by one. A clear sky period needs to pass the five
        conditions tested for in this function.
    """
    metrics_sum = 0
    # threshold values from Table 3
    if thresholds == "dni":
        t1 = 200
        t2 = 200
        t3 = 100
        t4 = 15
        t5 = 0.015
    else:
        t1 = 100
        t2 = 100
        t3 = 50
        t4 = 10
        t5 = 0.01

    if (np.abs(np.nanmean(obs) - np.nanmean(clr))) < t1:
        metrics_sum += 1

    if (np.abs(np.max(obs) - np.max(clr))) < t2:
        metrics_sum += 1

    tmp = (np.abs(np.abs(np.diff(obs)).sum() - np.abs(np.diff(clr)).sum()))
    if tmp < t3:
        metrics_sum += 1

    if (np.max(np.abs(np.diff(obs) - np.diff(clr)))) < t4:
        metrics_sum += 1

    # std of slopes normalized by irradiance mean
    t1 = np.std(np.diff(obs)) / np.mean(obs)
    t2 = np.std(np.diff(clr)) / np.mean(clr)
    if np.abs(t1 - t2) < t5:
        metrics_sum += 1

    return metrics_sum


def find_clearsky(df):
    # only applies for daytime values
    df["day"] = df.zen < 85

    starts = df.index.values
    ends = df.index + dt.timedelta(minutes=10)  # ten-minute sliding window
    # note that sliding window may preferentially treat shoulders of days
    # since windows will have fewer points

    cs_array = np.zeros(df.shape[0])  # start as False for clear sky
    for i in range(0, len(starts)):
        window_start = starts[i]
        window_end = ends[i]
        # select for sliding window
        sw = df[(df.index < window_end) & (df.index >= window_start)]
        day_check = sw.day.sum() == len(sw)  # all pts are labelled day
        npts_check = len(sw) > 5  # at least 5min of data
        if npts_check & day_check:
            obs = sw.GHI_m.to_numpy()
            clr = sw.GHI_c.to_numpy()
            ghi_metrics_sum = cs_metrics_1to5(obs, clr, thresholds="ghi")

            obs = sw.DNI_m.to_numpy()
            clr = sw.DNI_c.to_numpy()
            dni_metrics_sum = cs_metrics_1to5(obs, clr, thresholds="dni")

            if ghi_metrics_sum + dni_metrics_sum == 10:
                # all 10 criteria passed as true
                for j in range(sw.shape[0]):
                    # individually change each value in sliding window range
                    cs_array[i+j] = 1
    df['cs_period'] = cs_array
    df["cs_period"] = df["cs_period"].astype('bool')
    df = df.drop(columns=["day"])  # drop the day label column
    return df


def shakespeare(lat, lon):
    """

    Parameters
    ----------
    lat : double
        Latitude of site in degrees.
    lon : double
        Longitude of site in degrees.

    Returns
    -------
    h1 : double
        Scale height interpolated to given coordinate (lat, lon)
    tau_spline : spline
        Interpolated tau grid over q and He that can be evaluated at a given
        q and He.
    """
    filename = os.path.join("data", "shakespeare", "data.mat")
    f = loadmat(filename)
    # Get scale height H
    lon_pts = f["lon"]
    lat_pts = np.flip(f["lat"])  # must be in ascending order for interp
    h = np.flip(f["H"], axis=1)
    h_spline = interpolate.RectBivariateSpline(lon_pts, lat_pts, h)
    h1 = h_spline.ev(lon, lat).item()
    # he = (h1 / np.cos(40.3 * np.pi / 180)) * (df.p_ratio ** 1.8)
    tau_spline = interpolate.RectBivariateSpline(
        f["q"], f["Heff"], f["tau_eff_400"]
    )
    return h1, tau_spline


def shakespeare_comparison(site, year="2012"):
    # clear sky comparison only
    lat1 = SURFRAD[site]["lat"]
    lon1 = SURFRAD[site]["lon"]
    h1, spline = shakespeare(lat1, lon1)

    # use surfrad-only data (no cloud info)
    filename = os.path.join("data", "SURFRAD", f"{site}_{year}.csv")
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df = df.tz_localize("UTC")
    df = df[[
        "rh", "pressure", "t_a", "dw_ir", "lw_s",
        "zen", "cs_period"
    ]]
    df["pw_hpa"] = get_pw(df.t_a, df.rh) / 100  # hPa
    df["esky_c"] = get_esky_c(df.pw_hpa)
    df["lw_c"] = df.esky_c * SIGMA * np.power(df.t_a, 4)
    # drop rows with missing values in parameter columns
    df.dropna(subset=["t_a", "rh", "lw_c", "lw_s"], inplace=True)

    df = df.rename(columns={"pressure": "pa_hpa"})
    df["w"] = 0.62198 * df.pw_hpa / (df.pa_hpa - df.pw_hpa)
    df["q"] = df.w / (1 + df.w)
    p0 = 101325  # Pa
    df["p_ratio"] = (df.pa_hpa * 100) / p0
    df["he"] = (h1 / np.cos(40.3 * np.pi / 180)) * (df.p_ratio ** 1.8)
    df = df.drop(columns=["p_ratio"])
    # solve for tau at each q and he
    tau = []
    for q1, he1 in zip(df.q.values, df.he.values):
        tau.append(spline.ev(q1, he1).item())
    df["tau"] = tau

    # calc emissivity
    df["esky_t"] = 1 - np.exp(-1 * df.tau)
    df["lw_c_t"] = df.esky_t * SIGMA * np.power(df.t_a, 4)
    return df


def plot_lwerr_bin(df, mod, x, nbins=4, site=None, save_fig=False):
    if mod == "t":
        y_mod = "lw_c_t"
        y_true = "dw_ir"
        err = "lw_err_t"
        xlabel = r"$\Delta LW = LW_{\tau} - LW$ [W/m$^2$]"
    elif mod == "b":
        y_mod = "lw_c"
        y_true = "lw_s"
        err = "lw_err_b"
        xlabel = r"$\Delta LW = LW_{B} - LW_{s}$ [W/m$^2$]"
    if site is not None:
        df = df.loc[df.site == site].copy()
    if x == "pw":
        df["w_bin"] = pd.qcut(df.pw_hpa, nbins, labels=False)
    elif x == "rh":
        df["w_bin"] = pd.qcut(df.rh, nbins, labels=False)
    elif x == "tk":
        df["w_bin"] = pd.qcut(df.t_a, nbins, labels=False)
    elif x == "pa":
        df["w_bin"] = pd.qcut(df.pa_hpa, nbins, labels=False)
    # FIGURE
    fig, axes = plt.subplots(
        nbins, 1, figsize=(8, 10), sharex=True, sharey=True)
    for i in range(nbins):
        ax = axes[i]
        ax.grid(axis="x")
        q = df.loc[df.w_bin == i].copy()
        ax.hist(q[[err]], bins=30)
        ax.set_xlim(-30, 30)
        if x == "pw":
            title = f"Q{i+1}: p$_w$ [{q.pw_hpa.min():.2f}-{q.pw_hpa.max():.2f}]"
        elif x == "rh":
            title = f"Q{i + 1}: RH [{q.rh.min():.2f}-{q.rh.max():.2f}]"
        elif x == "tk":
            title = f"Q{i + 1}: T [{q.t_a.min():.2f}-{q.t_a.max():.2f}]"
        elif x == "pa":
            title = f"Q{i + 1}: P [{q.pa_hpa.min():.2f}-{q.pa_hpa.max():.2f}]"
        # ax.set_title(title, loc="left")
        ax.text(
            0.02, 0.8, s=title, backgroundcolor="1.0", size="medium",
            transform=ax.transAxes
        )
        ax.set_axisbelow(True)
        rmse = np.sqrt(mean_squared_error(q[[y_true]], q[[y_mod]]))
        mbe = compute_mbe(q[[y_true]].values, q[[y_mod]].values)[0]
        err_str = f"RMSE={rmse:.2f} W/m$^2$ \n MBE={mbe:.2f} W/m$^2$"
        ax.text(
            0.02, 0.3, s=err_str, backgroundcolor="1.0",
            transform=ax.transAxes
        )
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    if save_fig:
        if site is None:
            f = f"LWerr_{x}_bin={nbins}_{mod}.png"
        else:
            f = f"LWerr_{x}_bin={nbins}_{mod}_{site}.png"
        filename = os.path.join("figures", f)
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    return None


if __name__ == "__main__":
    print()
    # s = "BON"
    # process_site(s, "2013")
    # process_site(s, "2014")
    # process_site(s, "2015")
    # process_site(s, "2016")

    # # ASOS
    # # TODO create look up table for closest ASOS station
    # import_asos_yr(yr=2012)  # import closest asos stations for a given year
    # # NOTE: specified stations may not be available for a given year
    # # find info for a specific site

    # SURFRAD
    # site = "BON"
    # process_site(site=site, yr='2012')
    # df = join_surfrad_asos(site)
    # model, y_true, y_pred, rmse = custom_fit(df)
    # plot_fit(site, model.coef_, y_true, y_pred, rmse)

    # df = df.loc[df.zen < 85]  # remove night values
    # df["cmf"] = 1 - (df.GHI_m / df.GHI_c)
    # # apply 26a correlation
    # # df["li_lw"] = li_lw(df.cmf, df.t_a, df.rh, c=CORR26A, lwc=df.lw_c)

    # df["f"] = df.lw_s / df.dw_ir
    # df.f.hist(bins=100)

    # df = pd.DataFrame()
    # xvar = "year"
    # xlist = [2012, 2013, 2014, 2015, 2016]
    # const = "DRA"  # either keep year constant or site constant
    # for x in xlist:
    #     # for site in SURF_SITE_CODES:
    #     if xvar == "year":
    #         tmp = shakespeare_comparison(site=const, year=str(x))
    #         tmp = tmp.loc[tmp.cs_period]  # keep only clear skies
    #         # tmp["site"] = site
    #         tmp["year"] = x
    #     elif xvar == "site":
    #         tmp = shakespeare_comparison(site=x, year=const)
    #         tmp = tmp.loc[tmp.cs_period]  # keep only clear skies
    #         tmp["site"] = x
    #     df = pd.concat([df, tmp])
    # print(df.shape)
    # filename = os.path.join("data", f"cs_compare_{const}.csv")
    # df.to_csv(filename)

    filename = os.path.join("data", "cs_compare_2012.csv")
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    # site = "BOU"
    # df = df.sample(frac=0.1, random_state=96)
    df["e_act"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))
    df["e_act_s"] = df.lw_s / (SIGMA * np.power(df.t_a, 4))
    df["lw_err_t"] = df.lw_c_t - df.dw_ir
    df["lw_err_b"] = df.lw_c - df.lw_s
    # df = df.loc[df.site == site]

    # Graph histograms of error by quartile of some humidity metric
    nbins = 4
    xvar = "rh"  # ["pw", "rh", "tk", "pa"]
    # mod = "t"  # ["t", "b"] model type (tau or Brunt)
    plot_lwerr_bin(df, "t", xvar, nbins=nbins, save_fig=0)
    plot_lwerr_bin(df, "b", xvar, nbins=nbins, save_fig=0)

    pdf = df.sample(frac=0.2, random_state=96)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(250, 305, 20)
    y90 = get_pw(x, 90) / 100  # hpa
    y70 = get_pw(x, 70) / 100  # hpa
    y50 = get_pw(x, 50) / 100  # hpa
    ax.grid(alpha=0.3)
    c = ax.scatter(
        pdf.t_a, pdf.pw_hpa, marker=".", alpha=0.7, c=pdf.lw_err_b,
        cmap="seismic", vmin=-20, vmax=20, s=0.3
    )
    ax.plot(x, y90, c="0.9", ls="-", label="RH=90%")
    ax.plot(x, y70, c="0.7", ls="--", label="RH=70%")
    ax.plot(x, y50, c="0.5", ls=":", label="RH=50%")
    # norm = mpl.colors.SymLogNorm(linthresh=10, linscale=0.5, vmin=-20, vmax=20)
    ax.set_ylim(0, 35)
    ax.set_ylabel("P$_w$ [hPa]")
    ax.set_xlabel("T [K]")
    ax.set_axisbelow(True)
    ax.legend()
    clabel = r"$\Delta LW = LW_{\tau} - LW$ [W/m$^2$]"
    # clabel = r"$\Delta LW = LW_{B} - LW_{s}$ [W/m$^2$]"
    fig.colorbar(c, label=clabel, extend="both")
    ax.set_title("All sites, clr")
    # plt.show()
    filename = os.path.join("figures", "LWerr_T_v_pw_2.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)






