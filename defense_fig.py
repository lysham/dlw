"""Misc additional figures for defense presentation"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cf

from constants import SIGMA
from fraction import planck_lambda


COLORS = {
    "cornflowerblue": "#6495ED",
    "orange": "#F37748",
    "yellow": "#FFCA3A",
    "darkyellow": "#E0A500"
}


def bb_spectra():
    # electromagnetic spectrum figure
    wvl_sun = np.geomspace(0.15, 15, 100)  # micron
    bb_sun = planck_lambda(wvl_sun, t=5800)  # / (SIGMA * np.power(5670, 4))
    wvl_earth = np.geomspace(3, 100, 100)  # micron
    bb_earth = planck_lambda(wvl_earth, t=288)

    # scale down bb_sun by distance between earth and sun
    r_earth = 6378000  # m
    r_earthsun = 1.495e11  # m
    d_sun = 1.39e9  # m

    factor = np.power(d_sun, 2) / (4 * np.power(r_earthsun, 2))
    bb_sun = bb_sun * factor

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(wvl_sun / 1000000, bb_sun * wvl_sun / 4, lw=2, c=COLORS["darkyellow"])
    ax.plot(wvl_earth / 1000000, bb_earth * wvl_earth, lw=2, c=COLORS["orange"])
    ax.set_xscale("log")
    ax.set_xlabel("$\lambda$ [m]")
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(True)
    ax.set_yticks([])
    s = "Blackbody\nradiation\nat 5800 K\n(distance 1 a.u.)"
    ax.text(0.15, 0.6, s, transform=ax.transAxes, ha="right", va="center")
    s = "Blackbody\nradiation\nat 288 K"
    ax.text(0.8, 0.5, s, transform=ax.transAxes, ha="left", va="center")
    ax.set_ylim(bottom=0)
    ax.axvline(0.000004, ls="--", c="0.5")
    # ylabel = "W/m$^2$/$\mu$m"
    # ax.set_ylabel(ylabel)
    # plt.show()
    filename = os.path.join(folder, "bb_spectra.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def station_locations():
    # copied from GOES_class/archive/figures_for_senate.py
    # show all network location on map projection
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.STATES)
    ax.set_extent([-125, -65, 25, 52], crs=ccrs.PlateCarree())
    loc12 = [
        "BON", "DRA", "FPK",
        "GWN", "PSU", "SXF",
        "TBL", "ABQ", "BIS",
        "HNX", "SLC", "STE"
    ]

    LOC2COORD = {
        "BON": {"lat": 40.05192, "lon": -88.37309, "alt": 230},
        "DRA": {"lat": 36.62373, "lon": -116.01947, "alt": 1007},
        "FPK": {"lat": 48.30783, "lon": -105.10170, "alt": 634},
        "GWN": {"lat": 34.2547, "lon": -89.8729, "alt": 98},
        "PSU": {"lat": 40.72012, "lon": -77.93085, "alt": 376},
        "SXF": {"lat": 43.73403, "lon": -96.62328, "alt": 473},
        "TBL": {"lat": 40.12498, "lon": -105.23680, "alt": 1689},
        "ABQ": {"lat": 35.03796, "lon": -106.62211, "alt": 1617},
        "BIS": {"lat": 46.77179, "lon": -100.75955, "alt": 503},
        "HNX": {"lat": 36.31357, "lon": -119.63164, "alt": 73},
        "SEA": {"lat": 47.68685, "lon": -122.25667, "alt": 1288},
        "SLC": {"lat": 40.77220, "lon": -111.95495, "alt": 20},
        "STE": {"lat": 38.97203, "lon": -77.48690, "alt": 85},
    }

    for loc in loc12:
        lat = LOC2COORD[loc]["lat"]
        lon = LOC2COORD[loc]["lon"]
        ax.plot(lon, lat, "bo", ms=5, transform=ccrs.Geodetic())
        if loc == "HNX":
            ax.text(
                np.floor(lon) - 0.5, np.floor(lat), loc,
                transform=ccrs.Geodetic(),
                bbox=dict(boxstyle="square", ec="0.0", fc="1.0"),
                ha="right", va="top"
            )
        elif loc == "STE":
            ax.text(
                np.ceil(lon), np.floor(lat), loc,
                transform=ccrs.Geodetic(),
                bbox=dict(boxstyle="square", ec="0.0", fc="1.0"),
                ha="right", va="top"
            )
        else:
            ax.text(
                np.ceil(lon) + 1, np.floor(lat), loc, transform=ccrs.Geodetic(),
                bbox=dict(boxstyle="square", ec="0.0", fc="1.0")
            )
    ax.set_title("Selected SURFRAD and SOLRAD station locations")
    filename = os.path.join(folder, "station_locations.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    print()
    folder = os.path.join("figures", "defense")
    if not os.path.exists(folder):
        os.makedirs(folder)
