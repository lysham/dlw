"""Download and process bsrn data."""

import os
import time
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt

from process import find_clearsky
from main import get_pw
from constants import SIGMA


def get_data(station_code="gob", years=np.arange(2013, 2023)):
    # List of stations: wiki.pangaea.de/wiki/BSRN#Sortable_Table_of_Stations
    folder = os.path.join("data", "bsrn", station_code)
    if not os.path.exists(folder):
        os.makedirs(folder)

    bsrn_usr = input("Enter username: ")
    bsrn_pwd = input("Enter password: ")

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
        df = df[keep_cols]
        df = df.rename(columns=dict(
            temp_air="t_a", relative_humidity="rh", pressure="pa_hpa",
            lwd="dw_ir", ghi="GHI", dni="DNI", dhi="DHI"
        ))
        tmp = pd.concat([tmp, df])
    df = tmp.copy()  # switch back to df

    df.t_a += 273.15  # convert celsius to kelvin
    df = df.loc[(df.GHI > 0) & (df.DNI > 0)]  # remove GHI and DNI < 0 values
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
    df = df.merge(tmp["reno_cs"], how="left", left_index=True, right_index=True)

    filename = os.path.join("data", "bsrn", f"{site}.csv")
    df.to_csv(filename)
    return None


if __name__ == "__main__":
    print()
    # GOB
    lat = 66.439
    lon = 15.041
    alt = 416
    # folder = os.path.join("data", "bsrn", "gob")
    # process_data(site="gob", folder=folder, lat=66.439, lon=15.041, alt=416)

    filename = os.path.join("data", "bsrn", f"gob.csv")
    df = pd.read_csv(filename, parse_dates=True, index_col=0)