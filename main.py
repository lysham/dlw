"""Gather and plot figshare data."""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from constants import SIGMA

# location of data txt files stored locally
LONGMAN_FOLDER = os.path.join("data", "Longman_2018_figshare")

LW_STATION_IDS = [
    'HE282', 'HE283', 'HE284', 'HE285', 'HE286', 'HE287',
    'HE288', 'HN119', 'HN141', 'HN151', 'HN152', 'HN153',
    'HN161', 'HN162', 'HN164', 'HVO', 'HVT', 'MLO'
]  # HE288 not included in Tmin and Tmax


# list calibrated constants (Table 8 in Li et al. 2017)
CORR26A = [1.29, 0.8, 0.7, 0.78, 0.13]
CORR26B_DAY = [0.96, 1.2, 0.49, 1.09, 0.15]
CORR26B_ALL = [0.78, 1, 0.38, 0.95, 0.17]


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
    # precipitable water defined by eq(4) in ET report (2014)
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
    # print(f"Elev: {z:.0f}m, e_sky: {e_sky:.3f}")
    lwc = e_sky * SIGMA * np.power(t, 4)
    return lwc


def giambelluca_lw(cf, t, z):
    lwc = giambelluca_lwc(t, z)
    lw = lwc + (lwc * 0.202 * np.power(cf, 0.836))
    return lw


def get_pw(t, rh):
    """Get water vapor partial pressure.

    Parameters
    ----------
    t : float [K]
    rh : float [%]

    Returns
    -------
    p_w : float [Pa]
    """
    exp_term = 17.625 * (t - 273.15) / (t - 30.11)
    pw = 610.94 * (rh / 100) * np.exp(exp_term)  # Pa
    return pw


def get_esky_c(pw):
    # All day model
    e_sky = 0.618 + (0.056 * np.sqrt(pw))  # calibrated Brunt
    return e_sky


def li_lwc(t, rh):
    pw = get_pw(t, rh)
    pw /= 100  # hPa (mb)
    e_sky = get_esky_c(pw)
    lwc = e_sky * SIGMA * np.power(t, 4)
    return lwc


def li_lw(cf, t, rh, c=CORR26B_ALL, lwc=None):
    """

    Parameters
    ----------
    cf : float  or array-like
        CF or CMF given as [0, 1] value.
    t : float  or array-like
        Units K
    rh : float  or array-like
        Given as % value e.g. RH=30 is 30% humidity
    c : list
        Constants to be used in correlation
    lwc : float or array_like, optional
        Clear-sky longwave value [W/m^2]. If None, li_lwc is called.

    Returns
    -------
    lw : float or array-like
    """
    if lwc is None:
        lwc = li_lwc(t, rh)
    # rh = rh / 100  # convert to decimal
    term1 = lwc * (1 - (c[0] * np.power(cf, c[1])))
    term2 = c[2] * SIGMA * np.power(t, 4) * np.power(cf, c[3]) * np.power(rh, c[4])
    lw = term1 + term2
    return lw


def make_full_lw_plot():
    # make a plot of all LW data (18 stations on one timeseries)
    station_info = get_station_info()
    df = get_lw_station_data()

    stations = station_info.set_index("id").to_dict()["station"]
    df = df.filter(like="lw")
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.grid(axis="y", c="0.93")
    lines = ["-", "--", ":", "-."]
    markers = [".", "o", "v", "^", "*", "s", "<", ">"]
    i = 0
    for sid in stations.keys():
        # if df[[f"{sid}_lw"]].mean()[0] < 100:  # if filtering pos/neg
        ax.plot(
            df.index, df[[f"{sid}_lw"]], label=stations[sid], alpha=0.5,
            ls=lines[i // len(markers)], marker=markers[i % len(markers)]
        )
        i += 1
    ax.xaxis.set_major_locator(mpl.dates.YearLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax.set_ylabel("LW [W m$^{-2}$]")
    ax.set_xlim(df.loc[df.index.year == 2009].index[0], df.index[-1])
    # plt.show()
    filename = os.path.join("figures", "lw_data.png")
    fig.savefig(filename, dpi=300)
    return None


def plot_26b():
    # print("make figure for 26b")
    cf = np.linspace(0, 1, 11)
    conditions = [(30, 290), (30, 295), (40, 290)]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for rh, t in conditions:
        lw = li_lw(cf, t, rh)
        ax.plot(cf, lw, ".-", alpha=0.95, label=f"RH={rh}, T={t} K")
    ax.set_xlim(0, 1)
    ax.set_ylabel("LW [W m${-2}$]")
    ax.set_xlabel("CF")
    ax.legend()
    filename = os.path.join("figures", "corr26b.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_data_w_models(station, station_info, df):
    """Plot LW station data with T and RH for CF={0, 0.5, 1.0} using
    ET report 2014 and Li et al. correlations."""
    df = df.filter(like=station)

    # drop na
    df = df.dropna()

    # remove station name from column headers
    rename_columns = {}
    for col in df.columns:
        rename_columns[col] = col.split("_")[-1]
    df = df.rename(columns=rename_columns)

    # add column for Tavg = (Tmin + Tmax) / 2
    df = df.assign(tavg=((df.tmin+df.tmax)/2) + 273.15)  # tavg in [K]

    # add lw values, get elevation
    z = station_info.loc[station_info.id == station, "elev"].values[0]

    # Unclear what the LW observed value represents:
    # best guess is a daily average of the hourly value (probably not though)

    # add LW=f(CF) correlation values
    df = df.assign(
        lw_cf000_gb=giambelluca_lw(cf=0.0, t=df.tavg, z=z),
        lw_cf050_gb=giambelluca_lw(cf=0.5, t=df.tavg, z=z),
        lw_cf100_gb=giambelluca_lw(cf=1.0, t=df.tavg, z=z),
        lw_cf000_li=li_lw(cf=0, t=df.tavg, rh=df.rh),
        lw_cf050_li=li_lw(cf=0.5, t=df.tavg, rh=df.rh),
        lw_cf100_li=li_lw(cf=1.0, t=df.tavg, rh=df.rh),
    )

    # plot observations with correlations
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.grid(axis="y", c="0.95")
    ax.plot(df.index, df.lw, lw=2.0, c="#0b29e6", label="Observed")
    ax.plot(df.index, df.lw_cf000_gb, ls=":", c="0.8", label="CF=0.0")
    ax.plot(df.index, df.lw_cf050_gb, ls=":", c="0.5", label="CF=0.5")
    ax.plot(df.index, df.lw_cf100_gb, ls=":", c="0.3", label="CF=1.0")
    ax.plot(df.index, df.lw_cf000_li, c="#9f85c7", label="Li CF=0.0")
    ax.plot(df.index, df.lw_cf050_li, c="#804ecc", label="Li CF=0.5")
    ax.plot(df.index, df.lw_cf100_li, c="#5808cf", label="Li CF=1.0")
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    years = mpl.dates.YearLocator()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.set_title(station)
    ax.set_xlim(df.index[0], df.index[-1])
    plt.show()
    return None


def get_lw_station_data():
    # LOAD LW STATION DATA
    filename = os.path.join("data", "lw_station_data.csv")
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df


def get_station_info():
    # LOAD LW STATION INFO
    filename = os.path.join("data", "lw_station_info.csv")
    df = pd.read_csv(filename)
    return df


def compute_mbe(actual, model):
    return np.nanmean((model - actual), axis=0)


if __name__ == "__main__":
    print()

    # # import dataframes
    # station_info = get_station_info()
    # df = get_lw_station_data()
    #
    # plot_data_w_models("HN164", station_info, df)

    # make_full_lw_plot()
    # plot_26b()





