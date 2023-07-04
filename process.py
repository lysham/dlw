"""File dedicated to processing raw SURFRAD data."""

import os
import time
import pvlib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

from constants import SURFRAD, SURF_COLS, SIGMA, SURF_SITE_CODES, LON_DICT


def find_clearsky(df, window=10, min_sample=2):
    # only applies for daytime values
    df["day"] = df.zen < 85

    starts = df.index.values
    ends = df.index.values + np.timedelta64(window, "m")  # X minutes
    # note that sliding window may preferentially treat shoulders of days
    # since windows will have fewer points

    cs_array = np.zeros(df.shape[0])  # start as False for clear sky
    for i in range(0, len(starts)):
        window_start = starts[i]
        window_end = ends[i]
        # select for sliding window
        sw = df[(df.index.values < window_end) &
                (df.index.values >= window_start)]
        day_check = sw.day.sum() == len(sw)  # all pts labelled day
        npts_check = len(sw) > min_sample  # at least X pts in sliding window
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
        # if i % 10000 == 0:
        #     print(i)
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
        For each condition that passes the threshold, `metrics_sum` is
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


def process_site(site, folder, yr="2012", min_sample=2):
    """
    Gather data for a given SURFRAD site in a given year.
    Perform quality control checks, format, and sample data.
    Requires access to hard drive.
    Processed data is saved as csv to data/SURFRAD folder.

    Parameters
    ----------
    site : string
    yr : string, optional
    min_sample : int, optional
        Minimum number of samples that must be in sliding window evaluated by
        the `find_clearsky()` function

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

    # Check quality control flags
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
    df = find_clearsky(df, min_sample=min_sample)
    df = df.asfreq("1T")
    df["cs_period"] = df.cs_period.fillna(False)  # ensure boolean column
    df["reno_cs"] = pvlib.clearsky.detect_clearsky(df.GHI_m, df.GHI_c)

    if not os.path.exists(os.path.join("data", "SURFRAD")):
        os.makedirs(os.path.join("data", "SURFRAD"))

    filename = os.path.join("data", "SURFRAD", f"{site}_{yr}.csv")
    # filename = os.path.join("data", "SURFRAD", f"{site}_{yr}_pvlib.csv")
    df.to_csv(filename)
    print(df.shape)

    dt = time.time() - start_time
    print(f"Completed {site} {yr} in {dt:.0f}s")
    return None


def process_site_night(site, folder, yr):
    start_time = time.time()
    site_name = SURFRAD[site]["name"]
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

    # remove daytime
    df = df.loc[df.zen > 90]

    col_rename = {
        'direct_n': 'DNI_m', 'dw_solar': 'GHI_m',
        'ghi': 'GHI_c', 'dni': 'DNI_c'
    }
    df = df.rename(columns=col_rename)
    filename = os.path.join("data", "SURFRAD", f"{site}_night_{yr}.csv")
    df.to_csv(filename)
    print(df.shape)

    dt = time.time() - start_time
    print(f"Completed {site} {yr} in {dt:.0f}s")
    return None


def add_pvlib_cs(site, year, drive="hdd"):
    # import processed files and add a column for reno_cs determination
    if drive == "usb":
        folder = os.path.join("/Volumes", "LMM_drive", "SURFRAD_processed")
    elif drive == "hdd":
        folder = os.path.join("/Volumes", "Lysha_drive", "SURFRAD_processed")
    else:
        folder = os.path.join("data", "SURFRAD")

    filename = os.path.join(folder, f"{site}_{year}.csv")
    df = pd.read_csv(filename, index_col=0, parse_dates=True)

    # find most appropriate sample frequency
    vals, counts = np.unique(np.diff(df.index.to_numpy()), return_counts=True)
    freq = vals[np.argmax(counts)]  # most frequent delta
    freq = freq / np.timedelta64(1, 's')  # convert to seconds
    # convert to even sample frequency
    tmp = df.asfreq(str(freq) + "S")
    # evaluate
    tmp["reno_cs"] = pvlib.clearsky.detect_clearsky(tmp.GHI_m, tmp.GHI_c)
    # return to original index
    df = df.merge(tmp["reno_cs"], how="left", left_index=True, right_index=True)
    df.to_csv(filename)  # save as same filename
    return None


def import_site_year(site, year, drive="hdd"):
    """Import a single site year of SURFRAD data from processed SURFRAD file.
    The DataFrame output has been produced by process_site() in process.py.

    Parameters
    ----------
    site : str
    year : int, str
    drive : ["usb", "hdd", "server4"], optional

    Returns
    -------
    df
    """
    # use surfrad-only data (no cloud fraction info)
    if drive == "usb":
        folder = os.path.join("/Volumes", "LMM_drive", "SURFRAD_processed")
    elif drive == "hdd":
        folder = os.path.join("/Volumes", "Lysha_drive", "SURFRAD_processed")
    else:
        folder = os.path.join("data", "SURFRAD")

    filename = os.path.join(folder, f"{site}_{year}.csv")

    column_dtypes = {
        'zen': 'float64',
        'GHI_m': 'float64',
        'DNI_m': 'float64',
        'diffuse': 'float64',
        'dw_ir': 'float64',
        'temp': 'float64',
        'rh': 'float64',
        'pa_hpa': 'float64',
        'dw_casetemp': 'float64',
        'dw_dometemp': 'float64',
        'uw_ir': 'float64',
        'uw_castemp': 'float64',
        'uw_dometemp': 'float64',
        'uvb': 'float64',
        'par': 'float64',
        'windspd': 'float64',
        'winddir': 'float64',
        't_a': 'float64',
        'GHI_c': 'float64',
        'DNI_c': 'float64',
        'dhi': 'float64',
        'cs_period': 'bool',
        'reno_cs': 'bool',
    }

    df = pd.read_csv(
        filename, index_col=0, parse_dates=True, dtype=column_dtypes
    )
    df.sort_index(inplace=True)
    df = df.tz_localize("UTC")
    # drop rows with missing values in parameter columns
    df.dropna(subset=["t_a", "rh"], inplace=True)
    df = df.rename(columns={"pressure": "pa_hpa"})

    df["site"] = site
    return df


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


if __name__ == "__main__":
    print()
    # filepath to folder SURFRAD data
    # directory = os.path.join("/Volumes", "LMM_drive", "SURFRAD")
    folder = os.path.join("data", "SURFRAD_raw")
    for site in SURF_SITE_CODES:
        start_time = time.time()
        for year in np.arange(2004, 2009):
            process_site(site=site, folder=folder, yr=f"{year}")
        print(site, time.time() - start_time, "\n")

    # df = pd.DataFrame()
    # for year in np.arange(2008, 2023):
    #     filename = os.path.join("data", "SURFRAD", f"DRA_night_{year}.csv")
    #     tmp = pd.read_csv(filename, index_col=0, parse_dates=True)
    #     df = pd.concat([df, tmp])
    #
    # mean_ir = []
    # for yr, group1 in df.groupby(df.index.year):
    #     for m, group2 in group1.groupby(group1.index.month):
    #         mean_ir.append(group2.dw_ir.mean())
    #
    # x = np.arange(len(mean_ir))
    # mean_ir = np.array(mean_ir)
    #
    # fig, ax = plt.subplots()
    # ax.grid(alpha=0.3)
    # ax.plot(x, mean_ir)
    # plt.show()

    # start_time = time.time()
    # for s in SURF_SITE_CODES:
    #     add_pvlib_cs(s, year="2008", drive="server4")
    #     print(s, time.time() - start_time)

    # # todo potential issue with existing t_sky columns and duplicate reno_cs
    # # redo all sites for consistency
    #
    # red = "#C54459"  # muted red
    # blue = "#4C6BE6"  # powder blue
    # mint = "#6CB289"
    # gold = "#E0A500"
    #
    # s = "DRA"
    # df = import_site_year(s, year=2020, drive="server4")
    # df = add_solar_time(df)
    # df = df.set_index("solar_time")
    # x = df[["cs_period", "reno_cs"]].groupby(df.index.date).sum()
    # x["daily_diff"] = np.abs(x.cs_period - x.reno_cs)
    # x["cr_diff"] = x.cs_period - x.reno_cs
    # # x = x.sort_values('daily_diff')
    #
    # fig, ax = plt.subplots(figsize=(11, 4))
    # ax.grid(alpha=0.3)
    # x1 = x.loc[x.cr_diff == 0].copy()
    # ax.scatter(x1.index, x1.cr_diff, c=mint, marker=".", label="C = R")
    # x1 = x.loc[x.cr_diff < 0].copy()
    # ax.scatter(x1.index, x1.cr_diff, c=blue, marker=".", label="C < R")
    # x1 = x.loc[x.cr_diff > 0].copy()
    # ax.scatter(x1.index, x1.cr_diff, c=red, marker=".", label="C > R")
    # ax.set_ylim(-400, 400)
    # ax.set_xlim(x.index[0], x.index[-1])
    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    # ax.set_ylabel("Daily C - R")
    # ax.set_title(f"{s} {x.index[-1].year}", loc="left")
    # ax.legend(frameon=True, ncol=3, bbox_to_anchor=(1, 0.99), loc="upper right")
    # plt.tight_layout()
    # plt.show()
    #
    # # fig, ax = plt.subplots(figsize=(10, 3))
    # # ax.scatter(range(x.shape[0]), x.cs_period, label="cs")
    # # ax.scatter(range(x.shape[0]), x.reno_cs, label="reno")
    # # plt.show()
    #
    # # FIGURE
    # plot_date = dt.date(2020, 8, 20)
    # pdf = df.loc[df.index.date == plot_date].copy()
    #
    # npts_r = pdf.loc[pdf.reno_cs].shape[0]
    # npts_c = pdf.loc[pdf.cs_period].shape[0]
    # npts_o = pdf.loc[pdf.reno_cs & pdf.cs_period].shape[0]
    # frac_r = npts_r / pdf.shape[0]
    # frac_c = npts_c / pdf.shape[0]
    # frac_o = npts_o / pdf.shape[0]
    # txt1 = f"frac clr: C {frac_c:.2f} | R {frac_r:.2f} | O {frac_o:.2f}"
    # txt2 = f"npts clr: C {npts_c} | R {npts_r} | O {npts_o}"
    #
    # fig, ax = plt.subplots(figsize=(12, 4))
    # date_str = plot_date.strftime("%m-%d-%Y")
    # title = f"{s} {date_str}"
    # ax.set_title(title, loc="left")
    # ax.plot(pdf.index, pdf.DNI_m, c=blue, label="DNI")
    # ax.plot(pdf.index, pdf.DNI_c, c=blue, ls="--", label="DNI_c")
    # ax.plot(pdf.index, pdf.GHI_m, c=mint, label="GHI")
    # ax.plot(pdf.index, pdf.GHI_c, c=mint, ls="--", label="GHI_c")
    # ax.fill_between(
    #     pdf.index, 0, pdf.GHI_m, where=pdf.cs_period, fc="0.7", alpha=0.4,
    #     label="CS", ec="0.3"
    # )
    # ax.fill_between(
    #     pdf.index, 0, pdf.GHI_m, where=pdf.reno_cs, fc="0.9", alpha=0.4,
    #     hatch="//", label="Reno", ec="0.5"
    # )
    # ax.text(0.01, 0.98, txt1, va="top", transform=ax.transAxes)
    # ax.text(0.01, 0.92, txt2, va="top", transform=ax.transAxes)
    # ax.set_ylim(bottom=0)
    # ax.legend(frameon=False, ncol=6, bbox_to_anchor=(1, 1), loc="lower right")
    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    # plt.show()
