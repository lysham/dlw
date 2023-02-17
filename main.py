"""Gather and plot figshare data."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TXT_FILES = [  # not complete list
    "Lw_DataFile.txt",
    "RH_DataFile.txt",
    "Tmin_Data_Not_Filled.txt"
]

LONGMAN_FOLDER = os.path.join("data", "Longman_2018_figshare")


if __name__ == "__main__":
    filename = os.path.join(LONGMAN_FOLDER, TXT_FILES[0])
    df = pd.read_csv(filename)

    fig, ax = plt.subplots()
    y = df.iloc[1, 8:].to_numpy()
    x = range(len(y))
    ax.plot(x, y, ".-")
    plt.show()