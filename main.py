"""Gather and plot figshare data."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# location of data txt files stored locally
LONGMAN_FOLDER = os.path.join("data", "Longman_2018_figshare")

LW_STATION_IDS = [
    'HE282', 'HE283', 'HE284', 'HE285', 'HE286', 'HE287',
    'HE288', 'HN119', 'HN141', 'HN151', 'HN152', 'HN153',
    'HN161', 'HN162', 'HN164', 'HVO', 'HVT', 'MLO'
]  # HE288 not included in Tmin and Tmax


def import_data_from_txt(filename, suffix="", filter_id=True):
    save_lw_station_info = False

    filename = os.path.join(LONGMAN_FOLDER, filename)
    df = pd.read_csv(filename, na_values=np.nan, encoding_errors="ignore")
    # rename columns
    new_names = {"Station.Name": "station", "Sta..ID": "id", "Elev.": "elev"}
    df = df.rename(columns=new_names)

    if save_lw_station_info:
        # get list of stations with LW data and store station info
        lw_stations = list(df.station.values.flatten())
        station_info = df[["station", "SKN", "id", "LAT", "LON", "elev"]]
        filename = os.path.join("data", "lw_station_info.csv")
        station_info.to_csv(filename, index=False)

    df = df.set_index("id")  # make station ID new index
    if filter_id:
        # filter for stations in the LW station IDs list
        df = df.filter(items=LW_STATION_IDS, axis=0)

    # pivot table
    df = df.drop(columns=df.columns[:7])  # drop other info columns
    df = df.pivot_table(columns="id")  # make id the columns with ts and index
    df.index.names = ["ts"]  # rename the index column

    # convert index to datetime objects
    df.index = pd.to_datetime(df.index, format="X%Y.%m.%d")

    # add suffix to column names
    df = df.add_suffix(suffix)
    return df


def save_lw_station_data():
    # import LW, RH, and T data
    lw = import_data_from_txt("Lw_DataFile.txt", suffix="_lw")
    rh = import_data_from_txt("RH_DataFile.txt", suffix="_rh")
    tmin = import_data_from_txt("Tmin_Data_Not_Filled.txt", suffix="_tmin")
    tmax = import_data_from_txt("Tmax_Data_Not_Filled.txt", suffix="_tmax")
    # combine
    df = lw.join([rh, tmin, tmax], how="left")
    df.to_csv(os.path.join("data", "lw_station_data.csv"))
    return None


if __name__ == "__main__":
    print()
    # # LOAD LW STATION INFO
    # filename = os.path.join("data", "lw_station_info.csv")
    # stat_info = pd.read_csv(filename)

    filename = os.path.join("data", "lw_station_data.csv")
    df = pd.read_csv(filename)

    # gather data for one station only
    station = "HE283"
    df = df.filter(like=station)

    col_names = df.columns
    rename_columns = {}
    for col in df.columns:
        rename_columns[col] = col.split("_")[-1]
    df = df.rename(columns=rename_columns)

    # add column for Tavg = (Tmin + Tmax) / 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(df.index, df.lw, c="0.8")
    axes[1].plot(df.index, df.rh, c="0.8")
    axes[2].plot(df.index, df.tmin, c="slategray")
    axes[2].plot(df.index, df.tmax, c="maroon")
    plt.show()

    # make new plot of LW only with LW observed
    # LW at various CF for ET model correlation
    # LW at various CF for Li(26b) correlation

