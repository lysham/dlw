"""Exploratory for now, using BE-Bra"""

import os
import pvlib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from constants import P_ATM, SIGMA
from main import get_pw


if __name__ == "__main__":
    folder = os.path.join("data", "fluxnet")

    # filename = os.path.join(folder, "FLX_BE-Bra_FLUXNET2015_SUBSET_HH_1996-2014_2-4.csv")
    # year = 2012
    # station = "BE-Bra"
    # lat, lon, elev = (51.3076, 4.5198, 16)
    # plot_title = f"{station} ({year})"

    filename = os.path.join(folder, "FLX_AT-Neu_FLUXNET2015_SUBSET_HH_2002-2012_1-4.csv")
    year = 2012
    station = "AT-Neu"
    lat, lon, elev = (47.1167, 11.3175, 970)
    plot_title = f"{station} ({year})"

    # Kind of line by line runs. Sometimes filter for a year. Sometimes keep multiple years.

    df_raw = pd.read_csv(filename, parse_dates=[0, 1])

    keep_cols = [
        'TIMESTAMP_START', 'TIMESTAMP_END', 'TA_F', 'TA_F_QC', 'SW_IN_POT',
        'SW_IN_F', 'SW_IN_F_QC', 'LW_IN_F', 'LW_IN_F_QC', 'VPD_F', 'VPD_F_QC',
        'PA_F', 'PA_F_QC', 'P_F', 'P_F_QC', 'RH'
    ]
    df = df_raw.copy()[keep_cols]

    df = df.set_index("TIMESTAMP_END")
    df = df.loc[df.index.year >= 2010]  # 2014 for BRA
    # braaschaat is GMT+1 (same with Neustift)
    df.index = df.index + pd.Timedelta(hours=-1)  # hacky

    # add clear sky values
    location = pvlib.location.Location(lat, lon, altitude=elev)
    cs_out = location.get_clearsky(df.index)
    tmp = location.get_solarposition(df.index)
    col_rename = {
        'SW_IN_F': 'GHI_m',
        'ghi': 'GHI_c', 'dni': 'DNI_c'
    }
    df = df.merge(cs_out, how='outer', left_index=True, right_index=True)
    df = df.merge(tmp.zenith, how='outer', left_index=True, right_index=True)
    df = df.loc[df.zenith < 90]
    df = df.rename(columns=col_rename)

    # find most appropriate sample frequency
    vals, counts = np.unique(np.diff(df.index.to_numpy()), return_counts=True)
    freq = vals[np.argmax(counts)]  # most frequent delta
    freq = freq / np.timedelta64(1, 's')  # convert to seconds
    # convert to even sample frequency
    tmp = df.asfreq(str(freq) + "S")
    # tmp["reno_cs"] = pvlib.clearsky.detect_clearsky(tmp.GHI_m, tmp.GHI_c)
    # evaluate (detect clear sky uses centered windows)
    tmp["reno_cs"] = pvlib.clearsky.detect_clearsky(
        tmp.GHI_m, tmp.GHI_c, window_length=90, mean_diff=79.144,
        max_diff=59.152, lower_line_length=-41.4, upper_line_length=77.78,
        var_diff=0.00745, slope_dev=68.579
    )
    # return to original index
    df = df.merge(tmp["reno_cs"], how="left", left_index=True, right_index=True)
    #
    # # OPTIONAL more for BE-BRA clear sky filter plot
    # pdf = df.loc[(df.index >= dt.datetime(2012, 7, 25)) & (df.index <= dt.datetime(2012, 7, 30))]
    # tmp = pdf.copy()  # need to preserve boolean array without NA
    # pdf = pdf.asfreq("30T", fill_value=np.nan)
    #
    # fig, ax = plt.subplots()
    # ax.plot(pdf.index, pdf.GHI_m, ".-", alpha=0.8, label="GHI$_m$")
    # ax.plot(pdf.index, pdf.GHI_c, lw=0.8, color="0.5", alpha=0.8, label="GHI$_c$")
    # ax.plot(tmp.loc[tmp.reno_cs].index, tmp.loc[tmp.reno_cs].GHI_m, ".", alpha=0.9, label="CS")
    # ax.set_ylabel("W/m$^2$")
    # ax.legend(ncol=3, bbox_to_anchor=(0, 1.02), loc="lower left")
    # ax.set_ylim(bottom=0)
    # fig.autofmt_xdate()
    # # plt.show()
    # fig.savefig(
    #     os.path.join("figures", f"{str.lower(station)}_CS.png"),
    #     dpi=300, bbox_inches="tight"
    # )

    # test correlation [28% clear dataset]
    df = df.loc[df.reno_cs]  # take only clear sky samples
    # PA_F [kPA], TA_F [degC], LW_IN_F [W/m^2], RH [%]
    df["t_a"] = df.TA_F + 273.15
    df["pw_hpa"] = get_pw(df.t_a, df.RH) / 100  # [hPa]
    df["dw_ir"] = df.LW_IN_F
    df["pw"] = (df.pw_hpa * 100) / P_ATM

    df["x"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["esky_c"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))  # target esky_c

    C1_CONST = 0.6
    C2_CONST = 1.652
    C3_CONST = 0.15
    df["y"] = C1_CONST + (C2_CONST * df.x) + (C3_CONST * (np.exp(-1 * elev / 8500) - 1))
    # rough toss of hour before 8, should really be before solar hour 8
    df = df.loc[df.index.hour >= 8]
    rmse = np.sqrt(mean_squared_error(df.esky_c, df.y))

    # plot esky predicted vs observed
    # fig, ax = plt.subplots()
    # ax.plot(df.esky_c, df.y, ".", alpha=0.3)
    # ax.axline((1, 1), slope=1, ls=":")
    # ax.set_xlabel("observed esky_c")
    # ax.set_ylabel("predicted esky_c")
    # plt.show()

    x = np.linspace(0.0001, 0.03, 40)
    y = C1_CONST + C2_CONST * np.sqrt(x)
    elev_correction = C3_CONST * (np.exp(-1 * elev / 8500) - 1)

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5)
    # ax.plot(df.pw, df.esky_c - elev_correction, ".", alpha=0.1, label=station)
    tmp = df.loc[df.index.year == 2012]
    for yr in [2010, 2011, 2012]:  # [2012, 2013, 2014]
        tmp = df.loc[df.index.year == yr]
        ax.plot(tmp.pw, tmp.esky_c - elev_correction, ".", alpha=0.1, label=yr)
    ax.plot(x, y, ls=":", color="k", label="0.6 + 1.652 sqrt(p$_w$)")
    ax.set_xlabel("p$_w$ [-]")
    ax.set_ylabel("emissivity [-]")
    ax.set_title(station)
    ax.set_axisbelow(True)
    lg = ax.legend()
    for lh in lg.legendHandles:
        lh.set_alpha(1)
    plt.show()

    fig.savefig(
        os.path.join("figures", f"{str.lower(station)}.png"),
        dpi=300, bbox_inches="tight"
    )