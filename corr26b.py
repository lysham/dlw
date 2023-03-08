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
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from copy import deepcopy

from main import SIGMA, get_pw, get_esky_c, li_lw, CORR26A
from pcmap_data_funcs import get_asos_stations

from constants import SURFRAD, SURF_COLS, SURF_ASOS


def pw_tdp(t_a, rh):
    # the function of calculating Pw(Pa) and Tdp(C)
    pw = 610.94 * (rh / 100) * np.exp(17.625 * (t_a - 273.15) / (t_a - 30.11))
    tdp = (243.04 * np.log(pw / 610.94)) / (17.625 - np.log(pw / 610.94))
    return pw, tdp


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


def process_site(site, yr="2012"):
    # export to csv of processed, sampled data from one SURF site for one year
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
        # df = df[df.zen < 85]  # remove night values

        # remove negative GHI and DNI values
        df = df.loc[(df.dw_solar > 0) & (df.direct_n > 0)]

        # Reduce sample here TODO remove later(?) (orig 1%)
        df = df.sample(frac=0.05, random_state=96)

        # apply clear sky analysis
        location = pvlib.location.Location(lat, lon, altitude=alt)
        cs_out = location.get_clearsky(df.index)
        col_rename = {
            'direct_n': 'DNI_m', 'dw_solar': 'GHI_m',
            'ghi': 'GHI_c', 'dni': 'DNI_c'
        }
        df = df.merge(cs_out, how='outer', left_index=True, right_index=True)
        df = df.rename(columns=col_rename)
        # sdf = find_clearsky(sdf) # no clear-sky analysis for now

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

        filename = os.path.join("data", "SURFRAD", f"{site}_{yr}.csv")
        df.to_csv(filename)
    else:
        print(f"Error: {yr} is not available for {site_name}")

    dt = time.time() - start_time
    print(f"Completed {site} in {dt:.0f}s")
    return None


def isd_history():
    # Import and store ASOS station info
    file_address = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv'
    df = pd.read_csv(file_address)  # total of 29705 stations worldwide
    filename = os.path.join("data", "isd_history.csv")
    df = df.rename(columns={"STATION NAME": "STATION_NAME", "ELEV(M)": "ELEV"})
    df.to_csv(filename, index=False)
    return None


def asos_site_data():
    return None


if __name__ == "__main__":
    print()

    process_site(site="BON", yr='2012')
    # process_site_yr(yr='2013')

    # TODO create look up table for closest ASOS station
    # modify script to look up one station specifically
    # process asos station data
    # get_asos_stations(year=2012, local_dir=os.path.join("data", "asos_2012"))

    # site = "BON"  # BONDVILLE_IL, UNIVERSITY OF WILLARD
    # usaf = SURF_ASOS[site]["usaf"]
    # wban = SURF_ASOS[site]["wban"]
    # # get ASOS station metadata
    # filename = os.path.join("data", "isd_history.csv")
    # asos = pd.read_csv(filename)
    # asos_site = asos.loc[(asos.USAF == str(usaf)) &
    #                      (asos.WBAN == wban)].iloc[0]
    # # get SURFRAD data (after processing)
    # filename = os.path.join("data", "SURFRAD", f"{site}_2012.csv")
    # site = pd.read_csv(filename, index_col=0, parse_dates=True)
    # # site = site.loc[site.location == SURFRAD["BON"]["name"]]
    # site.sort_index(inplace=True)
    # site = site.tz_localize("UTC")
    # # get ASOS station data
    # filename = os.path.join("data", "asos_2012", f"{usaf}-{wban}-2012.csv")
    # df = pd.read_csv(filename, skiprows=1, index_col=0, parse_dates=True)
    # df = df[['CF2', 'CBH2', 'TEMP', 'VSB', 'DEWP', 'SLP', 'PERCP']]
    # mask = df.index.duplicated(keep="first")
    # df = df[~mask].sort_index()  # drop duplicate indices
    #
    # # merge
    # df = pd.merge_asof(
    #     site, df, left_index=True, right_index=True,
    #     tolerance=pd.Timedelta("1h"), direction="nearest"
    # )
    # df["asos_t"] = df.TEMP + 273.15
    # df[["asos_t", "t_a"]].corr()
    # df["pw"] = get_pw(df.t_a, df.rh) / 100  # hPa
    # df["esky_c"] = get_esky_c(df.pw)
    # df["lw_c"] = df.esky_c * SIGMA * np.power(df.t_a, 4)
    # # TODO check different ways to join (bfill, linear interp)
    # check = df.copy()

    # df = df.rename(columns={"CF2": "cf"})
    # # drop rows with missing values in parameter columns
    # x = df.shape[0]
    # df.dropna(subset=["t_a", "rh", "cf", "lw_c", "lw_s"], inplace=True)
    # print(x, df.shape[0])
    #
    # df = df.assign(
    #     t1=df.lw_c * df.cf,
    #     t2=SIGMA * (df.t_a ** 4) * df.cf * (df.rh / 100),
    #     y=df.lw_s - df.lw_c
    # )
    # # inexact fit, only solving for two params
    #
    # b = df.lw_c.to_numpy()
    # train_x = df[["t1", "t2"]].to_numpy()
    # train_y = df.y.to_numpy()
    #
    # model = LinearRegression(fit_intercept=False)
    # model.fit(train_x, train_y)
    # print(model.coef_, model.intercept_)
    # rmse = np.sqrt(mean_squared_error(train_y, model.predict(train_x)))
    # print(f"{rmse:.2f}")
    #
    # y_true = train_y + b
    # y_pred = model.predict(train_x) + b
    #
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.grid(True, alpha=0.3)
    # ax.scatter(y_true, y_pred, alpha=0.3)
    # ax.axline((300, 300), slope=1, c="0.1", ls="--")
    # ax.set_xlabel("LW measured [W/m$^2$]")
    # ax.set_ylabel("Modeled [W/m$^2$]")
    # ax.set_title("BON")
    # plt.show()
    # # filename = os.path.join("figures", "first_fit.png")
    # # fig.savefig(filename, dpi=300, bbox_inches="tight")

    # TODO make a clear sky filter based on 10 tests

    # df = df.loc[df.zen < 85]  # remove night values
    # df["cmf"] = 1 - (df.GHI_m / df.GHI_c)
    # # apply 26a correlation
    # # df["li_lw"] = li_lw(df.cmf, df.t_a, df.rh, c=CORR26A, lwc=df.lw_c)

    # df["f"] = df.lw_s / df.dw_ir
    # df.f.hist(bins=100)