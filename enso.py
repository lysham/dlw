"""Explore correlation of constants with ENSO index"""

import os
import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def c1c2_intime():
    start_time = time.time()
    fix_c1 = False
    fix_c3 = True

    c3_const = 0.20  # used if fix_c3 is True
    c1_const = 0.55  # used if fix_c1 is True

    years = np.arange(2000, 2023)  # till 2022

    c1 = np.zeros(len(years))
    c2 = np.zeros(len(years))
    c3 = np.zeros(len(years))

    i = 0
    for yr in years:
        df = create_training_set(
            year=[yr], all_sites=True, temperature=True, cs_only=True,
            pct_clr_min=0.3, drive="server4"
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


if __name__ == "__main__":
    print()
    # preprocess_meiv2()
    # preprocess_oni()
    # c1c2_intime()  # TODO issue with constants

    filename = os.path.join("data", "enso_c", f"c1c2c3.csv")
    df = pd.read_csv(filename, index_col=0)

    df["month"] = 1
    df["day"] = 1  # locate date at first day of assigned month
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["year", "month", "day"])

    c1_avg = df.c1.mean()
    c2_avg = df.c2.mean()
    # df["c1_dev"] = df.c1 - c1_avg
    # df["c2_dev"] = df.c2 - c2_avg

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

    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax0.step(df.index, df.c1, label="c1", c=mint, where="post")
    ax0.step(df.index, df.c2, label="c2", c=gold, where="post")
    ax0.step(df.index, df.c3, label="c3", c="0.5", where="post")
    ax0.legend()

    ax.grid(axis="x", alpha=0.2)
    ax.axhline(0, ls="--", c="0.3", alpha=0.3)
    ax.plot(oni.index, oni.value, c=oni_color, alpha=0.4)
    ax.plot(mei.index, mei.value, c=mei_color, alpha=0.4)
    ax.step(oni.index, oni.yr_avg, label="ONI", c=oni_color, where="post")
    ax.step(mei.index, mei.yr_avg, label="MEIv2", c=mei_color, where="post")

    ax.set_xlim(oni.index[0], oni.index[-1])
    ymin, ymax = ax.get_ylim()
    ylim = abs(ymin) if abs(ymin) > ymax else ymax
    ax.set_ylim(-1 * ylim, ylim)  # ensure symmetric around y=0
    ax.set_ylabel(r"$\leftarrow$ La Niña$\quad\quad$ El Niño $\rightarrow$")
    ax.legend()
    ax.set_axisbelow(True)
    plt.show()
    # filename = os.path.join("figures", "enso_index.png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)

    df["mei"] = mei.value.groupby(mei.index.year).median().to_numpy()
    df["oni"] = oni.value.groupby(oni.index.year).median().to_numpy()
    df.corr()  # 0.57 correlation between c3 and mei, 0.51 for c3 and oni