"""File dedicated to processing raw SURFRAD data."""

import os
import time
import pvlib
import numpy as np
import pandas as pd
import datetime as dt

from constants import SURFRAD, SURF_COLS, SIGMA, SURF_SITE_CODES


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


def process_site(site, folder, yr="2012"):
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

    # all_years = os.listdir(directory)
    keep_cols = [
        'zen', 'dw_solar', 'qc1', 'direct_n', 'qc3', 'diffuse', 'qc4',
        'dw_ir', 'qc5', 'temp', 'qc16', 'rh', 'qc17', 'pressure', 'qc20',
        'dw_casetemp', 'dw_dometemp', 'uw_ir',
        'uw_castemp', 'uw_dometemp', 'uvb', 'par',
        'windspd', 'winddir'
    ]  # narrow down columns from SURF_COLNAMES
    # if yr in all_years:
    folder = os.path.join(folder, site_name, yr)
    lst = os.listdir(folder)
    # remove hidden files in MacOS
    lst = [i for i in lst if not i.startswith(".")]
    lst.sort()
    tmp = pd.DataFrame()
    expected_columns = len(SURF_COLS)
    for f in lst:  # import data by day and concatenate to `tmp`
        filename = os.path.join(folder, f)
        try:
            data = np.loadtxt(filename, skiprows=2)
            # Check number of columns is correct and number of rows > 1
            if len(data.shape) > 1:
                df = pd.DataFrame(data, columns=SURF_COLS)
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
            else:
                print(f"{filename} does not have rows.")
        except pd.errors.ParserError as e:
            print(f"Error: {e}")
    df = tmp.copy()  # switch back to df
    # print("Data collected.", time.time() - start_time)

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
    # print("QC and clear sky applied.", time.time() - start_time)

    # Apply clear sky period filter
    df = find_clearsky(df)
    # need to apply clear sky filter before data is sampled
    # print("Clear sky filter applied.", time.time() - start_time)

    # # Reduce sample size TODO remove later(?) (orig 1%)
    # df = df.sample(frac=0.05, random_state=96)

    # # Determine T_sky values, and correct for 3-50 micron range
    # f = pd.read_csv(os.path.join("data", "tsky_table_3_50.csv"))
    # t_sky = np.interp(df.dw_ir.values, f['ir_meas'].values, f['tsky'].values)
    # dlw = SIGMA * np.power(t_sky, 4)
    # df['t_sky'] = t_sky
    # df['lw_s'] = dlw
    # print("T_sky determined.", time.time() - start_time)

    if not os.path.exists(os.path.join("data", "SURFRAD")):
        os.makedirs(os.path.join("data", "SURFRAD"))

    filename = os.path.join("data", "SURFRAD", f"{site}_{yr}.csv")
    df.to_csv(filename)
    print(df.shape)

    dt = time.time() - start_time
    print(f"Completed {site} {yr} in {dt:.0f}s")
    return None


if __name__ == "__main__":
    # filepath to folder SURFRAD data
    # directory = os.path.join("/Volumes", "LMM_drive", "SURFRAD")
    folder = os.path.join("data", "SURFRAD_raw")

    start_time = time.time()
    for s in SURF_SITE_CODES:
        if s != "SXF":
            process_site(s, folder=folder, yr="2000")
            print(s, time.time() - start_time)

    # 2020 data is incomplete

