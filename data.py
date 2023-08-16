"""Export data for open research."""

import time
import os
import pandas as pd
import numpy as np
from corr26b import create_training_set
from figures import C3_CONST
from constants import SURF_SITE_CODES

from sklearn.linear_model import LinearRegression


def create_data_h5():
    col_rename = {
        'site': 'site',
        'dw_ir': 'dlw_m',
        'GHI_m': 'ghi_m',
        'DNI_m': 'dni_m',
        'diffuse': 'dhi_m',
        'rh': 'rh_m',
        'pa_hpa': 'pa_m',
        't_a': 't_m',
        'zen': 'sza',
        'GHI_c': 'ghi_c',
        'DNI_c': 'dni_c',
        'dhi': 'dhi_c',
        'cs_period': 'cs1',
        'reno_cs': 'cs2',
        'elev': 'site_elev',
        'clr_pct': 'clr_pct',
        'clr_num': 'clr_num',
        'pw_hpa': 'pw_hpa',
        'correction': 'alt_correction',
        'tra_set': 'tra',
        'val_set': 'val',
        'x': 'sqrt_pw',
        'y': 'e_sky',
    }
    order = list(col_rename.keys())

    start_time = time.time()
    filename = os.path.join("data", "tra.csv")
    tra = pd.read_csv(filename, index_col=0, parse_dates=True)
    tra["stime"] = tra.index.to_numpy()

    filename = os.path.join("data", "val.csv")
    val = pd.read_csv(filename, index_col=0, parse_dates=True)
    val["stime"] = val.index.to_numpy()

    # create instead of importing so that filters can be removed
    for site in SURF_SITE_CODES:
        df = create_training_set(
            year=[2010, 2011, 2012, 2013, 2014, 2015],
            temperature=False, cs_only=False, sites=[site],
            filter_solar_time=False, filter_pct_clr=0, filter_npts_clr=0,
            drive="server4"
        )
        df['correction'] = C3_CONST * (np.exp(-1 * df.elev / 8500) - 1)
        print(time.time() - start_time)

        # add columns to indicate if sample is part of training or validation
        df["stime"] = df.index.to_numpy()
        compare_on = ["site", "zen", "dw_ir", "stime"]

        # training set matching
        tra_match_idx = df[compare_on].merge(
            tra[compare_on], on=compare_on, how="left", indicator=True
        )["_merge"] == "both"
        df["tra_set"] = tra_match_idx.to_numpy()

        # validation set matching
        val_match_idx = df[compare_on].merge(
            val[compare_on], on=compare_on, how="left", indicator=True
        )["_merge"] == "both"
        df["val_set"] = val_match_idx.to_numpy()

        # fix column names
        drop_columns = ["temp", "csv2", "afgl_t0", "afgl_p0", "P_rep", "stime"]
        df = df.drop(columns=drop_columns)
        df = df[order]  # reorder columns in order of keys above
        df = df.rename(columns=col_rename)  # rename columns as above

        if site == "BOU":  # rename BOU -> TBL
            df["site"] = "TBL"
            site = "TBL"

        df = df.sort_index()  # sort by index

        filename = os.path.join("data", "export", "data.h5")
        df.to_hdf(filename, key=site, mode="a")
        print(site, time.time() - start_time)
    return None


def reconstruct_tra_val():
    # recreate training and validation sets
    filename = os.path.join("data", "export", "data.h5")
    training = []
    validation = []
    surfrad_sites = ['BON', 'DRA', 'FPK', 'GWC', 'PSU', 'SXF', 'TBL']
    for site in surfrad_sites:  # loop through sites
        df = pd.read_hdf(filename, key=site)
        training.append(df.loc[df.tra])  # append samples marked as training
        validation.append(df.loc[df.val])  # append samples marked as validation
    # join respective set samples across sites
    training = pd.concat(training, ignore_index=True)
    validation = pd.concat(validation, ignore_index=True)
    return training, validation


def reproduce_results(training):
    c1 = 0.6  # set intercept (c1 constant)
    x = training.sqrt_pw.to_numpy().reshape(-1, 1)
    y = training.e_sky - training.alt_correction - c1  # adjust for altitude and c1
    y = y.to_numpy().reshape(-1, 1)

    model = LinearRegression(fit_intercept=False)
    model.fit(x, y)

    c2 = model.coef_[0][0]
    print(f"c1={c1:.3f}, c2={c2:.3f}")
    # output: c1=0.600, c2=1.652
    return None


if __name__ == "__main__":
    print()
    # create_data_h5()

    # training, validation = reconstruct_tra_val()
    # reproduce_results(training)