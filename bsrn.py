"""Download and process bsrn data."""

import os
import time
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from process import find_clearsky
from main import get_pw
from constants import SIGMA, P_ATM
from corr26b import fit_linear


def get_data(station_code="gob", years=np.arange(2013, 2023)):
    # List of stations: wiki.pangaea.de/wiki/BSRN#Sortable_Table_of_Stations
    folder = os.path.join("data", "bsrn", station_code)
    if not os.path.exists(folder):
        os.makedirs(folder)

    bsrn_usr = str(input("Enter username: "))
    bsrn_pwd = str(input("Enter password: "))

    start_time = time.time()
    for yr in years:
        # downloads data up to and including entire month of end time
        data, metadata = pvlib.iotools.get_bsrn(
            start=pd.Timestamp(yr, 1, 1), end=pd.Timestamp(yr, 12, 1),
            station=station_code, username=bsrn_usr, password=bsrn_pwd
        )
        end_time = time.time()
        filename = os.path.join(folder, f"{station_code}_{yr}.csv")
        data.to_csv(filename)
        print(yr, end_time - start_time)
    return None


def process_data(site, folder, lat, lon, alt):
    start_time = time.time()
    # in the style of process_site in process.py
    lst = os.listdir(folder)
    lst = [i for i in lst if not i.startswith(".")]  # remove hidden OS files
    lst.sort()
    tmp = pd.DataFrame()

    keep_cols = [
        "ghi", "dni", "dhi", "lwd", "temp_air", "relative_humidity", "pressure"
    ]

    for f in lst:  # import data by year and concatenate to `tmp`
        filename = os.path.join(folder, f)
        df = pd.read_csv(
            filename, parse_dates=True, index_col=0
        )
        if not df.empty:  # if file is not empty
            df = df[keep_cols]
            df = df.rename(columns=dict(
                temp_air="t_a", relative_humidity="rh", pressure="pa_hpa",
                lwd="dw_ir", ghi="GHI", dni="DNI", dhi="DHI"
            ))
            # check values in dw_ir, t_a, rh
            t1, t2, t3 = df[["dw_ir", "t_a", "rh"]].count()
            if (t1 > 0) & (t2 > 0) & (t3 > 0):
                tmp = pd.concat([tmp, df])
    df = tmp.copy()  # switch back to df

    if not df.empty:
        df.t_a += 273.15  # convert celsius to kelvin
        df = df.loc[(df.GHI > 0) & (df.DNI > 0)]  # rmv GHI and DNI < 0 rows
        # apply helpful variable columns
        df["pw_hpa"] = get_pw(df.t_a, df.rh) / 100  # hPa
        df["site"] = site
        df["elev"] = alt
        df["e_act"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))

        # apply clear sky analysis
        location = pvlib.location.Location(lat, lon, altitude=alt)
        sol_pos = location.get_solarposition(df.index)
        df["zen"] = sol_pos.zenith
        df["eq_of_time"] = sol_pos.equation_of_time

        cs_out = location.get_clearsky(df.index)
        # find_clearsky uses capitalized irradiance headers
        cs_out = cs_out.rename(columns=dict(ghi="GHI", dni="DNI", dhi="DHI"))
        df = df.merge(
            cs_out, how='outer', left_index=True, right_index=True,
            suffixes=["_m", "_c"]
        )
        print("checkpoint1", time.time() - start_time)

        # Apply clear sky period filter
        df = find_clearsky(df, min_sample=5)  # adds cs_period column
        print("checkpoint2", time.time() - start_time)
        # apply reno hansen clear sky filter
        tmp = df.asfreq("60S")
        tmp["reno_cs"] = pvlib.clearsky.detect_clearsky(tmp.GHI_m, tmp.GHI_c)
        # return to original index
        df = df.merge(tmp["reno_cs"], how="left",
                      left_index=True, right_index=True)

        filename = os.path.join("data", "bsrn", f"{site}.csv")
        df.to_csv(filename)
    else:
        print("DataFrame is empty.")
    return None


if __name__ == "__main__":
    print()
    # get_data(station_code="tam", years=np.arange(2009, 2017))

    # site = "gob"
    # lat = -23.5614
    # lon = 15.042
    # alt = 416

    # site = "tam"
    # lat = 22.7903
    # lon = 5.5292
    # alt = 1385

    # site = "brb"
    # lat = -15.601
    # lon = -47.713
    # alt = 1023

    site = "dwn"
    lat = -12.424
    lon = 130.8925
    alt = 32

    # folder = os.path.join("data", "bsrn", site)
    # process_data(site=site, folder=folder, lat=lat, lon=lon, alt=alt)

    # import and post process
    filename = os.path.join("data", "bsrn", f"{site}.csv")
    df = pd.read_csv(filename, parse_dates=True, index_col=0)

    filter_pct_clr = 0.05
    filter_npts_clr = 0.20

    # add solar time
    dtime = pd.to_timedelta(df.eq_of_time + (4 * lon), unit="m")
    df["solar_time"] = df.index + dtime

    df = df.set_index("solar_time")
    df = df.loc[df.index.hour > 8]  # filter solar time
    df["csv2"] = (df.cs_period & df.reno_cs)

    if filter_pct_clr is not None:
        # apply daily percent clear filter
        tmp_clr = df["csv2"].resample("D").mean()
        df["clr_pct"] = tmp_clr.reindex(df.index, method="ffill")
        df = df.loc[df.clr_pct >= filter_pct_clr].copy(deep=True)
    if filter_npts_clr is not None:
        # apply daily absolute number clear filter
        tmp_clr = df["csv2"].resample("D").count()
        thresh = np.quantile(
            tmp_clr.loc[tmp_clr > 0].to_numpy(), filter_npts_clr
        )
        df["clr_num"] = tmp_clr.reindex(df.index, method="ffill")
        df = df.loc[df.clr_num >= thresh].copy(deep=True)

    # reduce to cs_only
    df = df.loc[df.csv2].copy(deep=True)
    df["x"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["y"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))

    c1_const = 0.6
    c3_const = 0.15
    df['correction'] = c3_const * (np.exp(-1 * df.elev / 8500) - 1)
    df["e_act"] = df.y.to_numpy()
    df["y"] = df.y + df.correction - c1_const

    shape = df.shape[0]
    df = df.dropna()
    n_dropped = shape - df.shape[0]
    print("Nrows with NA", n_dropped)

    out = []
    for yr, group1 in df.groupby(df.index.year):
        for m, group2 in group1.groupby(group1.index.month):
            n_pts = group2.shape[0]
            train_y = group2.y.to_numpy().reshape(-1, 1)
            _, c2 = fit_linear(group2, set_intercept=0)
            pred_y = c2 * group2.x
            rmse = np.sqrt(mean_squared_error(train_y, pred_y))
            entry = dict(
                year=yr, month=m, day=1, c2=c2,
                rmse=rmse, n_pts=n_pts,
                avg_x=group2.x.mean(), avg_y=group2.y.mean(),
                avg_e=group2.e_act.mean(), avg_pw=group2.pw_hpa.mean(),
                avg_t=group2.t_a.mean(), med_t=group2.t_a.median(),
                avg_rh=group2.rh.mean(), med_rh=group2.rh.median(),
                avg_lw=group2.dw_ir.mean(), med_lw=group2.dw_ir.median()
            )
            out.append(entry)
    out = pd.DataFrame(out)
    out["date"] = pd.to_datetime(out[["year", "month", "day"]])
    out = out.set_index("date").sort_index()
    out = out.drop(columns=["year", "month", "day"])

    pdf = out.copy()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(alpha=0.3)
    ax.plot(pdf.index, pdf.c2, ".-")
    ax.set_ylim(1, 2)
    ax.set_xlim(pdf.index[0], pdf.index[-1])
    title = f"{site.upper()} fitted c2 values for c1=0.6, c3=0.15"
    ax.set_title(title, loc="left")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(alpha=0.3)
    ax.plot(pdf.index, pdf.avg_lw, ".-")
    ax.set_xlim(pdf.index[0], pdf.index[-1])
    title = f"{site.upper()} monthly average LW of clear sky samples"
    ax.set_title(title, loc="left")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()