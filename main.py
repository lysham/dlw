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


SIGMA = 5.6697e-8  # W/m^2


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


def giambelluca_w(p):
    # precipitable water vapor [cm^-1] defined by eq(4) in ET report (2014)
    c0 = -1.342063
    c1 = 7.661469e-5
    c2 = -1.652886e-9
    c3 = 1.314865e-14
    w = c0 + (c1 * p) + (c2 * np.power(p, 2)) + (c3 * np.power(p, 3))
    return w


def giambelluca_lwc(t, z):
    """
    Clear-sky longwave downwelling [W/m^2] defined by eqn(13) in ET report

    Parameters
    ----------
    t : float
        air temeprature [K]
    z : float
        elevation [m]

    Returns
    -------
    lwc : float
        estimated clear sky longwave downwelling [W/m^2]
    """
    p0 = 101500  # [Pa] sea level pressure
    p = p0 * np.exp(-z / 8500)  # atm pressure [Pa], eqn(5) in ET report
    w = giambelluca_w(p)
    # atmospheric emissivity, eqn(14) in ET report
    e_sky = 0.762 + (0.055 * np.log(w)) + (0.0031 * np.log(np.power(w, 2)))
    lwc = e_sky * SIGMA * np.power(t, 4)
    return lwc


def giambelluca_lw(cf, t, z):
    lwc = giambelluca_lwc(t, z)
    lw = lwc + (lwc * 0.202 * np.power(cf, 0.836))
    return lw


if __name__ == "__main__":
    print()
    # LOAD LW STATION INFO
    filename = os.path.join("data", "lw_station_info.csv")
    stat_info = pd.read_csv(filename)

    # LOAD LW STATION DATA
    filename = os.path.join("data", "lw_station_data.csv")
    df = pd.read_csv(filename, index_col=0)

    # reduce to one station only (STATION SPECIFIC ONWARD)
    station = "HE283"
    df = df.filter(like=station)

    # drop na
    df = df.dropna()

    # remove station name from column headers
    col_names = df.columns
    rename_columns = {}
    for col in df.columns:
        rename_columns[col] = col.split("_")[-1]
    df = df.rename(columns=rename_columns)

    # add column for Tavg = (Tmin + Tmax) / 2
    df = df.assign(tavg=(df.tmin+df.tmax)/2)

    # add lw values
    z = stat_info.loc[stat_info.id == station, "elev"].values[0]  # elevation

    # Unclear what the LW observed value represents:
    # best guess is a daily average of the hourly value (probably not though)

    # add LW=f(CF) correlation values, multiply by 3600 to get an hourly value
    df = df.assign(
        lw_cf000_gb=giambelluca_lw(cf=0.0, t=df.tavg, z=z) * -3600,
        lw_cf050_gb=giambelluca_lw(cf=0.5, t=df.tavg, z=z) * -3600,
        lw_cf100_gb=giambelluca_lw(cf=1.0, t=df.tavg, z=z) * -3600
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df.lw, c="#0b29e6", label="Observed")
    ax.plot(df.index, df.lw_cf000_gb, c="0.8", label="CF00")
    ax.plot(df.index, df.lw_cf050_gb, c="0.5", label="CF50")
    ax.plot(df.index, df.lw_cf100_gb, c="0.3", label="CF100")
    ax.legend()
    plt.show()

    # TODO notes
    # make new plot of LW only with LW observed
    # LW at various CF for ET model correlation
    # LW at various CF for Li(26b) correlation

