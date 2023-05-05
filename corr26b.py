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
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from copy import deepcopy
from scipy.io import loadmat
from scipy.optimize import curve_fit

from main import get_pw, get_esky_c, li_lw, CORR26A, compute_mbe, pw2tdp, tdp2pw
from pcmap_data_funcs import get_asos_stations

from constants import SIGMA, SURFRAD, SURF_COLS, SURF_ASOS, SURF_SITE_CODES, \
    P_ATM, E_C1, E_C2, ELEV_DICT, ELEVATIONS, LON_DICT, SEVEN_COLORS, P_ATM_BAR


def tsky_table(l1, l2):
    """Form a lookup table for T_sky over a range of IR measurements.
    Save as csv.

    Parameters
    ----------
    l1 : int [micron]
        Lower wavelength of transmission window.
    l2 : int [micron]
        Upper wavelength of transmission window.

    Returns
    -------
    None
    """
    ir = np.linspace(70, 500, 1000)
    tsky = np.zeros(len(ir))
    for i in range(len(ir)):
        if ir[i] < 100:  # semi-optimize first guess to speed up process
            t = 200
        elif 100 <= ir[i] < 200:
            t = 230
        elif 200 <= ir[i] < 300:
            t = 260
        elif 300 <= ir[i] < 400:
            t = 280
        else:
            t = 300
        tsky[i] = get_tsky(t, ir[i], l1=l1, l2=l2)[1]
    df = pd.DataFrame({"ir_meas": ir, "tsky": tsky})
    filename = os.path.join("data", f"tsky_table_{l1}_{l2}.csv")
    df.to_csv(filename, index=False)
    return None


def get_tsky(t, ir_mea, l1=3, l2=50):
    # function to recursively solve for T_sky
    y = lambda x: (E_C1 * np.power(x, -5)) / (np.exp(E_C2 / (x * t)) - 1)
    out = integrate.quad(func=y, a=l1, b=l2)
    result = out[0]  # output: result, error, infodict(?), message, ...
    delta = ir_mea - result
    delta_thresh = 0.1
    # if delta > (delta_thresh * 100):
    #     step = 0.1
    # else:
    step = 0.01
    if (delta > 0) & (np.abs(delta) > delta_thresh):
        # if delta>0 guessed Tsky should increase
        t = t * (1 + step)
        result, t = get_tsky(t, ir_mea, l1=l1, l2=l2)
    elif (delta < 0) & (np.abs(delta) > delta_thresh):
        t = t * (1 - step)
        result, t = get_tsky(t, ir_mea, l1=l1, l2=l2)
    return result, t


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
    # directory = os.path.join("/Volumes", "LMM_drive", "SURFRAD", site_name)
    directory = os.path.join("data", "SURFRAD_raw", site_name)
    # all_years = os.listdir(directory)
    keep_cols = [
        'zen', 'dw_solar', 'qc1', 'direct_n', 'qc3', 'diffuse', 'qc4',
        'dw_ir', 'qc5', 'temp', 'qc16', 'rh', 'qc17', 'pressure', 'qc20',
        'dw_casetemp', 'dw_dometemp', 'uw_ir',
        'uw_castemp', 'uw_dometemp', 'uvb', 'par',
        'windspd', 'winddir'
    ]  # narrow down columns from SURF_COLNAMES
    # if yr in all_years:
    folder = os.path.join(directory, yr)
    lst = os.listdir(folder)
    # remove hidden files in MacOS
    lst = [i for i in lst if not i.startswith(".")]
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
    df = df[df.columns.drop(list(df.filter(regex='qc')))]  # drop qc cols
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

    # # Reduce sample size TODO remove later(?) (orig 1%)
    # df = df.sample(frac=0.05, random_state=96)

    # Determine T_sky values, and correct for 3-50 micron range
    f = pd.read_csv(os.path.join("data", "tsky_table_3_50.csv"))
    t_sky = np.interp(df.dw_ir.values, f['ir_meas'].values, f['tsky'].values)
    dlw = SIGMA * np.power(t_sky, 4)
    df['t_sky'] = t_sky
    df['lw_s'] = dlw
    print("T_sky determined.", time.time() - start_time)

    filename = os.path.join("data", "SURFRAD", f"{site}_{yr}.csv")
    df.to_csv(filename)

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


def get_cs_metrics_1to5(obs, clr, thresholds="ghi"):
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


def find_clearsky(df, window=10, min_sample=5):
    # only applies for daytime values
    df["day"] = df.zen < 85

    starts = df.index.values
    ends = df.index + dt.timedelta(minutes=window)  # ten-minute sliding window
    # note that sliding window may preferentially treat shoulders of days
    # since windows will have fewer points

    cs_array = np.zeros(df.shape[0])  # start as False for clear sky
    for i in range(0, len(starts)):
        window_start = starts[i]
        window_end = ends[i]
        # select for sliding window
        sw = df[(df.index < window_end) & (df.index >= window_start)]
        day_check = sw.day.sum() == len(sw)  # all pts are labelled day
        npts_check = len(sw) > min_sample  # at least X min of data
        if npts_check & day_check:
            obs = sw.GHI_m.to_numpy()
            clr = sw.GHI_c.to_numpy()
            ghi_metrics_sum = get_cs_metrics_1to5(obs, clr, thresholds="ghi")

            obs = sw.DNI_m.to_numpy()
            clr = sw.DNI_c.to_numpy()
            dni_metrics_sum = get_cs_metrics_1to5(obs, clr, thresholds="dni")

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
    # drop rows with missing values in parameter columns
    df.dropna(subset=["t_a", "rh", "lw_s"], inplace=True)

    df = df.rename(columns={"pressure": "pa_hpa"})
    df["pw_hpa"] = get_pw(df.t_a, df.rh) / 100  # hPa
    tmp = np.log(df.pw_hpa * 100 / 610.94)
    df["tdp"] = 273.15 + ((243.04 * tmp) / (17.625 - tmp))
    # df["esky_c"] = get_esky_c(df.pw_hpa)
    # df["lw_c"] = df.esky_c * SIGMA * np.power(df.t_a, 4)

    # add elevation and altitude correction
    df["elev"] = ELEV_DICT[site]  # Add elevation
    df["P_rep"] = P_ATM * np.exp(-1 * df.elev / 8500)  # Pa
    df["de_p"] = 0.00012 * ((df.P_rep / 100) - 1000)  # TODO

    # shakespeare variables
    df["w"] = 0.62198 * df.pw_hpa / (df.pa_hpa - df.pw_hpa)
    df["q"] = df.w / (1 + df.w)
    df["p_ratio"] = (df.pa_hpa * 100) / P_ATM
    df["he"] = (h1 / np.cos(40.3 * np.pi / 180)) * np.power(df.p_ratio, 1.8)
    df["tau"] = spline.ev(df.q.to_numpy(), df.he.to_numpy())
    df = df.drop(columns=["p_ratio", "w"])

    # calc emissivity
    df["esky_t"] = 1 - np.exp(-1 * df.tau)
    df["lw_c_t"] = df.esky_t * SIGMA * np.power(df.t_a, 4)
    return df


def plot_lwerr_TvPw():
    df = import_cs_compare_csv("cs_compare_2012.csv")
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
    return None


def create_cs_compare_csv(xvar, const, xlist):
    """Create a compiled data file. Store in cs_compare folder.

    Parameters
    ----------
    xvar : ["year", "site"]

    const : str
        Name to be appended to filename as cs_compare_{const}.csv
    xlist : list
        Values of xvar to loop through. Either list of years or list of sites.

    Returns
    -------
    None
    """
    df = pd.DataFrame()
    for x in xlist:
        if xvar == "year":
            tmp = shakespeare_comparison(site=const, year=str(x))
            tmp = tmp.loc[tmp.cs_period]  # keep only clear skies
            # tmp["site"] = site
            tmp["year"] = x
        elif xvar == "site":
            tmp = shakespeare_comparison(site=x, year=const)
            tmp = tmp.loc[tmp.cs_period]  # keep only clear skies
            tmp["site"] = x
        df = pd.concat([df, tmp])
    filename = os.path.join("data", "cs_compare", f"cs_compare_{const}.csv")
    df.to_csv(filename)
    return None


def import_cs_compare_csv(csvname, site=None):
    # csvname has to end with either _{year}.csv or _{site}.csv
    filename = os.path.join("data", "cs_compare", csvname)
    df = pd.read_csv(filename, index_col=0, parse_dates=True)

    if site is not None:
        df = df.loc[df.site == site]

    # add solar time correction
    df = add_solar_time(df)
    # # add solar time correction
    # tmp = pd.DatetimeIndex(df.solar_time.copy())
    # t = tmp.hour + (tmp.minute / 60) + (tmp.second / 3600)
    # df["de_t"] = 0.013 * np.cos(np.pi * (t / 12))

    # IJHMT fit
    df["esky_c"] = 0.6376 + 1.6026 * np.sqrt((df.pw_hpa * 100) / P_ATM)
    df["lw_c"] = df.esky_c * SIGMA * np.power(df.t_a, 4)

    df["e_act"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))
    return df


def reduce_to_equal_pts_per_site(df):
    # keep equal number of points per site
    site_pts = df.groupby(df.site).t_a.count().sort_values().to_dict()
    min_pts = df.groupby(df.site).t_a.count().values.min()
    new_df = pd.DataFrame()
    for s in site_pts.keys():
        tmp = df.loc[df.site == s].copy()
        tmp = tmp.sample(min_pts, random_state=30)
        new_df = pd.concat([new_df, tmp])
    return new_df.copy()


def e_time(n):
    # equation of time, matches spencer71 from pvlib out to 10^-2
    b = ((n - 1) / 365) * 2 * np.pi  # rad
    e = 229.2 * (0.0000075 + (0.001868 * np.cos(b)) - (0.032077 * np.sin(b)) -
                 (0.014615 * np.cos(2 * b)) - (0.04089 * np.sin(2 * b)))
    return e


def add_solar_time(df):
    # index should be in UTC, df must include site
    doy = df.index.dayofyear.to_numpy()
    eq_of_time = pvlib.solarposition.equation_of_time_spencer71(doy)
    df["lon"] = df["site"].map(LON_DICT)
    df["dloc"] = 4 * df.lon
    minutes = df.dloc + eq_of_time  # difference in min
    df["dtime"] = pd.to_timedelta(minutes, unit="m")
    df["solar_time"] = df.index + df.dtime
    df = df.drop(columns=["dtime", "lon", "dloc"])
    return df


def fit_linear(df, set_intercept=None, print_out=False):
    """Linear regression on emissivity. Output fitted coefficients.

    Parameters
    ----------
    df : DataFrame
        Data to fit regression. DataFrame must have columns: x and y
    set_intercept : double, optional
        Pass in the value of c1.
        Default is None. If None, the linear regression with default to
        fitting to its own y-intercept (i.e. c1)
    print_out : bool, optional
        Print coefficients, RMSE, and R2 evaluated on the same data on which
        the regression was fitted.

    Returns
    -------
    c1, c2 : double
        Return c1 and c2 constants rounded to 4th decimal place.
    """
    # linear regression on esky_c data
    # df["pp"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    # df["y"] = df.e_act_s3 + df.de_p - 0.6376
    train_x = df.x.to_numpy().reshape(-1, 1)
    train_y = df.y.to_numpy().reshape(-1, 1)
    if set_intercept is not None:  # fix c1
        model = LinearRegression(fit_intercept=False)
    else:
        model = LinearRegression(fit_intercept=True)
    model.fit(train_x, train_y)
    c2 = model.coef_[0].round(4)
    if set_intercept is not None:  # use given c1
        c1 = set_intercept
        pred_y = c2 * df.x
    else:  # use model-fitted c1
        c1 = model.intercept_.round(4)
        pred_y = c1 + (c2 * df.x)
    if isinstance(c1, np.ndarray):
        c1 = c1[0]
    if isinstance(c2, np.ndarray):
        c2 = c2[0]

    if print_out:
        rmse = np.sqrt(mean_squared_error(train_y, pred_y))
        r2 = r2_score(train_y, pred_y)
        print("(c1, c2): ", c1, c2)
        print(f"RMSE: {rmse.round(5)} | R2: {r2.round(5)}")
        print(f"npts={df.shape[0]:,}")
    # curve fitting code in reference_func
    return c1, c2


def three_c_fit(df, c1=None):
    # simultaneously fit model (c1, c2) and altitude correction (c3)
    # df must have columns: x, y, elev, site
    # check if mult-location or not
    if len(np.unique(df.site.to_numpy())) == 1:
        if c1 is not None:
            c1, c2 = fit_linear(df, set_intercept=c1)
        else:
            c1, c2 = fit_linear(df)
        c3 = 0
    else:
        df['correction'] = np.exp(-1 * df.elev / 8500) - 1
        x = df[['x', 'correction']]
        if c1 is not None:
            y = df['y'] - c1
            model = LinearRegression(fit_intercept=False)
            model.fit(x, y)
            c2, c3 = model.coef_.round(4)
        else:
            y = df['y']
            model = LinearRegression(fit_intercept=True)
            model.fit(x, y)
            c2, c3 = model.coef_.round(4)
            c1 = model.intercept_.round(4)
    return c1, c2, c3


def add_afgl_t0_p0(df):
    # df must have column "elev"
    filename = os.path.join("data", "afgl_midlatitude_summer.csv")
    afgl = pd.read_csv(filename)
    afgl_alt = afgl.alt_km.values * 1000  # m
    afgl_temp = afgl.temp_k.values
    afgl_pa = afgl.pres_mb.values
    df["afgl_t0"] = np.interp(df.elev.values, afgl_alt, afgl_temp)
    df["afgl_p0"] = np.interp(df.elev.values, afgl_alt, afgl_pa)
    return df


def create_training_set(year=[2012, 2013], all_sites=True, site=None,
                        temperature=False, cs_only=True, pct_clr_min=0.3):
    # start broad then filter
    keep_cols = [
        "zen", "GHI_m", "DNI_m", "diffuse", "dw_ir", "t_a", "rh",
        "pa_hpa", "pw_hpa", "cs_period", "elev", "P_rep", "tdp", "q", "he"
    ]

    if all_sites:
        site_codes = SURF_SITE_CODES
    else:
        site_codes = [site]

    df = pd.DataFrame()
    for s in site_codes:
        for yr in year:
            tmp = shakespeare_comparison(s, yr)
            tmp = tmp[keep_cols]
            tmp["site"] = s
            tmp = add_solar_time(tmp)
            tmp = tmp.set_index("solar_time")
            if pct_clr_min is not None:
                tmp_clr = tmp["cs_period"].resample("D").mean()
                tmp["daily_clr"] = tmp_clr.reindex(tmp.index, method="ffill")
                tmp = tmp.loc[tmp.daily_clr >= pct_clr_min].copy()

            df = pd.concat([df, tmp])

    # filter solar time
    df = df.loc[df.index.hour > 8].copy()

    if temperature:
        df = add_afgl_t0_p0(df)
        df = df.loc[(abs(df.t_a - df.afgl_t0) <= 2)].copy()

    if cs_only:
        df = df.loc[df.cs_period].copy()  # reduce to only clear skies

    df["x"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["y"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))

    return df


if __name__ == "__main__":
    print()
    # start_time = time.time()
    # for s in ['BON', 'BOU', 'GWC', 'DRA', 'FPK', 'SXF', 'PSU']:
    #     process_site(s, yr="2015")
    #     process_site(s, yr="2016")
    #     print(s, time.time() - start_time)

    # tsky_table(3, 50)
    # tsky_table(4, 50)

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

    # CREATE CS_COMPARE
    # create_cs_compare_csv(xvar="site", const="2012", xlist=SURF_SITE_CODES)
    # xlist = [2012, 2013, 2014, 2015, 2016]
    # create_cs_compare_csv(xvar="year", const="GWC", xlist=[2012])

    print()
    tmp = pd.DataFrame()
    for s in SURF_SITE_CODES:
        df = shakespeare_comparison(s, 2012)
        df["site"] = s
        df = add_solar_time(df)
        df = df.set_index("solar_time")
        # add column for average clearness of the day
        df["pct_clr"] = df["cs_period"].resample(
            "D").mean().reindex(df.index, method="ffill")
        df = df.loc[df.index.hour > 8].copy()  # remove data before 8am solar
        tmp = pd.concat([tmp, df])

    df = add_afgl_t0_p0(df)
    df = df.loc[(abs(df.t_a - df.afgl_t0) <= 2) &
                (abs(df.pa_hpa - df.afgl_p0) <= 50)].copy()

    ref = tmp.copy()
    df = ref.loc[ref.pct_clr >= 0.3].copy()
    print(df.shape, ref.shape)

    df = df.loc[df.cs_period].copy()  # reduce to only clear skies
    print(df.shape)

    df["x"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["y"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))
    c1, c2, c3 = three_c_fit(df)
    print(c1, c2, c3)

    # df["y"] = df.y - 0.6376
    # c1, c2 = fit_linear(df, set_intercept=0.6376)
    # print(c1, c2)
