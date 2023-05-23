"""Explore correlation of constants with ENSO index"""

import os
import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.tsa.seasonal import seasonal_decompose

from corr26b import create_training_set, fit_linear, alt2sl, \
    reduce_to_equal_pts_per_site, three_c_fit


def preprocess_meiv2():
    filename = os.path.join("data", "enso_data", "meiv2_raw.txt")
    last_yr_in_file = 2023
    first_yr_in_file = 1979
    # skip endmatter and incomplete last year
    nrows = last_yr_in_file - first_yr_in_file
    columns = ["year", "DJ", "JF", "FM", "MA", "AM", "MJ",
                "JJ", "JA", "AS", "SO", "ON", "ND"]
    df = pd.read_csv(
        filename, sep="\s+", skiprows=1, nrows=nrows, names=columns
    )
    df = preprocess_helper(df, columns)
    filename = os.path.join("data", "enso_data", "meiv2.csv")
    df.to_csv(filename)
    return None


def preprocess_oni():
    filename = os.path.join("data", "enso_data", "oni_raw.csv")
    nrows = 2023 - 1950
    nrows += nrows // 10  # header line every 10 years
    skiprows = np.arange(11, nrows, 11)  # line 12, header=0, row1=index 0
    df = pd.read_csv(filename, skiprows=skiprows)
    # Drop last row for incomplete year (2023)
    df.drop(df.tail(1).index, inplace=True)
    # melt into single column file
    df = df.rename(columns={"Year": "year"})
    df = preprocess_helper(df, df.columns.to_list())
    filename = os.path.join("data", "enso_data", "oni.csv")
    df.to_csv(filename)
    return None


def preprocess_helper(df, columns):
    """Melt wide to long. Return DataFrame sorted on a datetime index.
    YYYY-MM-DD

    Parameters
    ----------
    df : DataFrame
        Includes column "year" and 12 succeeding columns of multi-month
        index values.
    columns : list
        Expect first value to be "year" followed by month strings of
        multi-month averages.

    Returns
    -------
    df
    """
    month_dict = {}  # create map of columns to integer month numbers
    for i in range(1, len(columns)):
        month_dict[columns[i]] = i
    df = pd.melt(
        df, id_vars=["year"], value_vars=columns[1:],
        var_name="month_str", value_name="value"
    )
    df["month"] = df["month_str"].map(month_dict)
    df["day"] = 1  # locate date at first day of assigned month
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["year", "month", "day"])
    return df


def import_oni_mei_data():
    filename = os.path.join("data", "enso_data", "oni.csv")
    oni = pd.read_csv(filename, index_col=0, parse_dates=True)
    filename = os.path.join("data", "enso_data", "meiv2.csv")
    mei = pd.read_csv(filename, index_col=0, parse_dates=True)
    return oni, mei


def c1c2_intime(pct_clr=0.05, npts_clr=0.2):
    start_time = time.time()
    fix_c1 = True
    fix_c3 = True

    c3_const = 0.13  # used if fix_c3 is True
    c1_const = 0.6  # used if fix_c1 is True

    years = np.arange(2000, 2023)  # till 2022

    c1 = np.zeros(len(years))
    c2 = np.zeros(len(years))
    c3 = np.zeros(len(years))

    i = 0
    for yr in years:
        df = create_training_set(
            year=[yr], temperature=False, cs_only=True,
            filter_pct_clr=pct_clr, filter_npts_clr=npts_clr, drive="server4"
        )
        df = reduce_to_equal_pts_per_site(df)

        if fix_c3:
            df = alt2sl(df, c3=c3_const)
            c3[i] = c3_const
            if fix_c1:
                df["y"] = df["y"] - c1_const
                c1[i], c2[i] = fit_linear(df, set_intercept=c1_const)
            else:
                c1[i], c2[i] = fit_linear(df, set_intercept=None)
        else:
            c1[i], c2[i], c3[i] = three_c_fit(df)

        i += 1
        print(yr, time.time() - start_time)

    df = pd.DataFrame(dict(year=years, c1=c1, c2=c2, c3=c3))
    if fix_c3:
        if fix_c1:
            name = f"c1={c1_const}_c3={c3_const}.csv"
        else:
            name = f"c1c2_c3={c3_const}.csv"
    else:
        name = "c1c2c3.csv"

    df.to_csv(os.path.join("data", "enso_c", name))
    return None


def c3_intime(c1_const, c2_const):
    start_time = time.time()
    years = np.arange(2000, 2023)  # till 2022
    c3 = np.zeros(len(years))

    i = 0
    for yr in years:
        df = create_training_set(
            year=[yr], temperature=False, cs_only=True,
            filter_pct_clr=0.2, filter_npts_clr=0.2, drive="server4"
        )
        df = reduce_to_equal_pts_per_site(df)
        df["y"] = df.y - (c1_const + (c2_const * df.x))
        df["x"] = np.exp(-1 * df.elev / 8500) - 1
        _, c3[i] = fit_linear(df, set_intercept=0)

        i += 1
        print(yr, time.time() - start_time)

    df = pd.DataFrame(dict(year=years, c1=c1_const, c2=c2_const, c3=c3))
    df.to_csv(os.path.join("data", "enso_c", f"c3_{c1_const}_{c2_const}.csv"))
    return None


def plot_index(filename, save_name="enso_index", c1=True, c2=True, c3=True):
    df = pd.read_csv(filename, index_col=0)

    df["month"] = 1
    df["day"] = 1  # locate date at first day of assigned month
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["year", "month", "day"])

    # c1_avg = df.c1.mean()
    # c2_avg = df.c2.mean()
    # c3_avg = df.c3.mean()

    mei_color = "#C54459"  # muted red
    oni_color = "#4C6BE6"  # powder blue
    mint = "#6CB289"
    gold = "#E0A500"

    oni, mei = import_oni_mei_data()
    mei = mei.loc[mei.index.year >= 2000]
    oni = oni.loc[oni.index.year >= 2000]

    # Get year averages
    tmp = {}
    oni["year"] = oni.index.year
    for yr, group in oni.groupby(oni.year):
        tmp[yr] = group.value.mean()
    oni["yr_avg"] = oni["year"].map(tmp)

    # Do the same for MEI
    tmp = {}
    mei["year"] = mei.index.year
    for yr, group in mei.groupby(oni.year):
        tmp[yr] = group.value.mean()
    mei["yr_avg"] = mei["year"].map(tmp)

    # fig, (ax0, ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.grid(axis="x", alpha=0.2)
    ax.axhline(0, ls="--", c="0.3", alpha=0.3)
    ax.plot(oni.index, oni.value, c=oni_color, alpha=0.4)
    ax.plot(mei.index, mei.value, c=mei_color, alpha=0.4)
    ax.step(oni.index, oni.yr_avg, label="ONI", c=oni_color, where="post")
    ax.step(mei.index, mei.yr_avg, label="MEIv2", c=mei_color, where="post")

    if c1:
        ax.step(df.index, df.c1, label="c1", c=mint, where="post")
    if c2:
        ax.step(df.index, df.c2, label="c2", c=gold, where="post")
    if c3:
        ax.step(df.index, df.c3, label="c3", c="0.5", where="post")

    ax.set_xlim(oni.index[0], oni.index[-1])
    ymin, ymax = ax.get_ylim()
    ylim = abs(ymin) if abs(ymin) > ymax else ymax
    ax.set_ylim(-1 * ylim, ylim)  # ensure symmetric around y=0
    ax.set_ylabel(r"$\leftarrow$ La Niña$\quad\quad$ El Niño $\rightarrow$")
    ax.legend()
    ax.set_axisbelow(True)
    plt.show()
    # filename = os.path.join("figures", save_name + ".png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)

    df["mei"] = mei.value.groupby(mei.index.year).median().to_numpy()
    df["oni"] = oni.value.groupby(oni.index.year).median().to_numpy()
    print(df.corr())
    # 0.57 correlation between c3 and mei, 0.51 for c3 and oni
    return None


if __name__ == "__main__":
    print()
    # preprocess_meiv2()
    # preprocess_oni()
    # c1c2_intime()  # TODO issue with constants
    # c3_intime(c1_const=0.6, c2_const=1.6)
    filename = os.path.join("data", "enso_c", f"c1c2c3.csv")
    # filename = os.path.join("data", "enso_c", f"c1=0.6_c3=0.13.csv")
    # plot_index(filename, save_name="tmp")
    df = pd.read_csv(filename, index_col=0)

    # # increments of 5 ye    ars[00 - 05, 06 - 10, 11 - 15, 16 - 22]
    # for y1, y2 in [(0, 5), (6, 10), (11, 15), (16, 22)]:
    #     df = create_training_set(
    #         year=np.arange(y1, y2 + 1) + 2000,
    #         temperature=False, cs_only=True,
    #         filter_pct_clr=0.05, filter_npts_clr=0.2, drive="server4"
    #     )
    #     filename = os.path.join(
    #         "data", "train_df",
    #         f"train_pct20_npts20_{str(y1+2000)[-2:]}_{str(y2+2000)[-2:]}.csv")
    #     df.to_csv(filename)
    #
    # start_time = time.time()
    # df = pd.DataFrame()
    # for y1, y2 in [(6, 10), (11, 15), (16, 22)]:
    #     filename = os.path.join(
    #         "data", "train_df",
    #         f"train_pct20_npts20_{str(y1+2000)[-2:]}_{str(y2+2000)[-2:]}.csv")
    #     tmp = pd.read_csv(filename, index_col=0, parse_dates=True)
    #     tmp = tmp.loc[(abs(tmp.t_a - tmp.afgl_t0) <= 2)]
    #     df = pd.concat([df, tmp])
    #     print(time.time() - start_time)
    #
    # df["year"] = df.index.year
    # df["month"] = df.index.month
    # df_train = df.copy(deep=True)
    #
    # keep_samples = 1000
    # # site = "GWC"
    # out = []  # c1, c2, c3, rmse, r2, npts
    # for yr, group1 in df_train.groupby(df_train.year):
    #     # group1 = group1.loc[group1.site == site]
    #     for m, group2 in group1.groupby(group1.month):
    #         n_pts = group2.shape[0]
    #         n_sites = len(np.unique(group2.site.to_numpy()))
    #         group2 = group2.sample(
    #             n=keep_samples, replace=True, random_state=21)
    #
    #         c1, c2, c3 = three_c_fit(group2)
    #         train_y = group2.y.to_numpy().reshape(-1, 1)
    #         if n_sites > 1:
    #             pred_y = c1 + (c2 * group2.x) + (c3 * group2.correction)
    #         else:
    #             pred_y = c1 + (c2 * group2.y)
    #         rmse = np.sqrt(mean_squared_error(train_y, pred_y))
    #         r2 = r2_score(train_y, pred_y)
    #         entry = dict(
    #             year=yr, month=m, day=1,
    #             c1=c1, c2=c2, c3=c3,
    #             r2=r2, rmse=rmse, n_pts=n_pts, n_sites=n_sites
    #         )
    #         out.append(entry)
    # # df_ref = df.copy(deep=True)
    # df = pd.DataFrame(out)
    # df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    # df = df.set_index("date").sort_index()
    # df = df.drop(columns=["year", "month", "day"])
    #
    # mei_color = "#C54459"  # muted red
    # oni_color = "#4C6BE6"  # powder blue
    # mint = "#6CB289"
    # gold = "#E0A500"
    #
    # oni, mei = import_oni_mei_data()
    # mei = mei.loc[mei.index.year >= 2006]
    # oni = oni.loc[oni.index.year >= 2006]
    #
    # month_window = 3
    # df = df.rolling(month_window, center=False).sum()
    #
    # fig, (ax0, ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # # fig, ax = plt.subplots(figsize=(16, 4))
    # ax0.step(df.index, df.rmse)
    # ax0.set_ylim(bottom=0)
    #
    # ax.grid(axis="x", alpha=0.2)
    # ax.axhline(0, ls="--", c="0.3", alpha=0.3)
    # ax.axhline(0.6, ls=":", c="0.8", alpha=0.3)
    # ax.plot(oni.index, oni.value, c=oni_color, alpha=0.4)
    # ax.plot(mei.index, mei.value, c=mei_color, alpha=0.4)
    #
    # ax.step(df.index, df.c1 / month_window, label="c1", c=mint, where="post")
    # ax.step(df.index, df.c2 / month_window, label="c2", c=gold, where="post")
    # ax.step(df.index, df.c3 / month_window, label="c3", c="0.5", where="post")
    #
    # ax.set_xlim(df.index[0], df.index[-1])
    # # ymin, ymax = ax.get_ylim()
    # # ylim = abs(ymin) if abs(ymin) > ymax else ymax
    # # ax.set_ylim(-1 * ylim, ylim)  # ensure symmetric around y=0
    # ax.set_ylim(-2, 2)
    # ax.set_ylabel(r"$\leftarrow$ La Niña$\quad\quad$ El Niño $\rightarrow$")
    # ax.legend()
    # ax.set_axisbelow(True)
    # ax.set_title(f"Month rolling window = {month_window}", loc="left")
    # plt.show()
    # #
    # # filename = os.path.join("figures", f"enso_m{month_window}_2.png")
    # # fig.savefig(filename, bbox_inches="tight", dpi=300)

    # # df["mei"] = mei.value.groupby(mei.index.year).median().to_numpy()
    # # df["oni"] = oni.value.groupby(oni.index.year).median().to_numpy()
    # df = pd.merge_asof(
    #     df, mei["value"], left_index=True, right_index=True, direction="nearest"
    # )
    # df = pd.merge_asof(
    #     df, oni["value"], left_index=True, right_index=True, direction="nearest"
    # )
    # print(df.corr())
    #
    # analysis = df[['c1']].copy()
    # analysis.dropna(inplace=True)
    # decompose_result_mult = seasonal_decompose(analysis, model="additive", period=12)
    # # decompose_result_mult.plot()
    # # plt.show()
    # x1 = decompose_result_mult.trend
    #
    # analysis = df[['c2']].copy()
    # analysis.dropna(inplace=True)
    # decompose_result_mult = seasonal_decompose(analysis, model="additive", period=12)
    # # decompose_result_mult.plot()
    # # plt.show()
    # x2 = decompose_result_mult.trend
    #
    # analysis = df[['c3']].copy()
    # analysis.dropna(inplace=True)
    # decompose_result_mult = seasonal_decompose(analysis, model="additive", period=12)
    # # decompose_result_mult.plot()
    # # plt.show()
    # x3 = decompose_result_mult.trend
    #
    # trend = decompose_result_mult.trend
    # seasonal = decompose_result_mult.seasonal
    # residual = decompose_result_mult.resid
    #
    # fig, ax = plt.subplots(figsize=(16, 4))
    # ax.grid(alpha=0.2)
    # ax.axhline(0, ls="--", c="0.3", alpha=0.3)
    # ax.axhline(0.6, ls=":", c="0.8", alpha=0.3)
    # ax.plot(oni.index, oni.value, c=oni_color, alpha=0.4)
    # ax.plot(mei.index, mei.value, c=mei_color, alpha=0.4)
    # ax.fill_between(
    #     mei.index, -1, 1, where=abs(mei.value) > 0.5,
    #     fc="0.8", alpha=0.5,
    #     transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    # )
    # # ax.plot(x1 - x1.mean(), c="k")
    # ax.plot(x2 - x2.mean(), c=gold, label="c2")
    # ax.plot(x3 - x3.mean(), c="0.5", label="c3")
    # ax.set_ylabel(r"$\leftarrow$ La Niña$\quad\quad$ El Niño $\rightarrow$")
    # ax.set_xlim(oni.index[0], oni.index[-1])
    # ax.legend()
    # plt.show()
    # filename = os.path.join("figures", f"enso_detrend_v2_2.png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)