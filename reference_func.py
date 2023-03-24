"""Keep functions not in use but helpful for referencing"""

from main import *
from sklearn.metrics import mean_squared_error
from corr26b import get_tsky, join_surfrad_asos, shakespeare
from constants import SURF_ASOS, SURFRAD


def look_at_jyj():
    # Code below from python notebooks looking at 26b correlation
    filename = os.path.join("data", "jyj_2017_data", "JYJ_traindataforcollapse")
    train = pd.read_pickle(filename)
    t_a = train['temp'].values + 273.15
    rh_vals = train['rh'].values
    lw_meas = train['LWmeas'].values

    # fig, ax = plt.subplots()
    # ax.plot(train.dw_ir, train.LWmeas, ".", alpha=0.3)
    # ax.axline((1, 1), slope=1, c="0.0")
    # plt.show()

    # import test values
    filename = os.path.join("data", "jyj_2017_data", "JYJ_testdataforcollapse")
    test = pd.read_pickle(filename)
    t_a_ = test['temp'].values + 273.15
    rh_ = test['rh'].values
    pa_ = test['pressure'].values * 100
    lw_meas_ = test['LWmeas'].values

    t = 292.85
    rh = 25.1
    ir_mea = 291.8
    ir, t_sky = get_tsky(t, ir_mea)
    # ir = SIGMA * t_sky ** 4
    pw = get_pw(t, rh)
    esky = get_esky_c(pw)
    t_sky2 = esky ** (1 / 4) * t
    print(t_sky, t_sky2, t_sky - t_sky2)
    print(ir)
    return None


def isd_history():
    # Import and store ASOS station info
    # only need to run once to create isd_history.csv
    file_address = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv'
    df = pd.read_csv(file_address)  # total of 29705 stations worldwide
    filename = os.path.join("data", "isd_history.csv")
    df = df.rename(columns={"STATION NAME": "STATION_NAME", "ELEV(M)": "ELEV"})
    df.to_csv(filename, index=False)
    return None


def make_surf_asos_df():
    df = pd.DataFrame.from_dict(SURF_ASOS, orient="index")
    return df


def check_for_cf_in_asos_data():
    # check which stations have CF data
    folder = os.path.join("data", "asos_2012")
    all_files = os.listdir(folder)
    stations = []
    for x in os.listdir(folder):
        f = x.split(".")
        if len(f) > 1:
            if len(f[0]) > 10:
                stations.append(x)

    for f in stations:
        filename = os.path.join(folder, f)
        df = pd.read_csv(filename, skiprows=1, index_col=0, parse_dates=True)
        cols = df.columns
        print(f, cols)
        if "CF2" in cols:
            print(f, "CF2 exists", df.CF2.mean())
    return None


def plot_shakespeare_comparison():
    # Explore shakespeare paper approach
    # produces "LW_clr_{site}.png" and "esky_clr_{site}.png"
    site = "FPK"
    lat1 = SURFRAD[site]["lat"]
    lon1 = SURFRAD[site]["lon"]

    h1, spline = shakespeare(lat1, lon1)

    df = join_surfrad_asos(site)
    df = df[[
        "rh", "pressure", "t_a", "pw", "dw_ir", "lw_s",
        "esky_c", "lw_c", "zen", "cs_period"
    ]]
    df = df.rename(columns={"pressure": "pa_hpa", "pw": "pw_hpa"})
    df["w"] = 0.62198 * df.pw_hpa / (df.pa_hpa - df.pw_hpa)
    df["q"] = df.w / (1 + df.w)
    p0 = 101325  # Pa
    df["p_ratio"] = (df.pa_hpa * 100) / p0
    df["he"] = (h1 / np.cos(40.3 * np.pi / 180)) * (df.p_ratio ** 1.8)
    df = df.drop(columns=["p_ratio"])
    # solve for tau at each q and he
    tau = []
    for q1, he1 in zip(df.q.values, df.he.values):
        tau.append(spline.ev(q1, he1).item())
    df["tau"] = tau

    # calc emissivity
    df["esky_t"] = 1 - np.exp(-1 * df.tau)
    df["lw_c_t"] = df.esky_t * SIGMA * np.power(df.t_a, 4)

    df["esky_day"] = 0.598 + (0.057 * np.sqrt(df.pw_hpa))

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5)
    ax.scatter(df.esky_t, df.esky_day, marker=".", alpha=0.1, label="(22a)")
    ax.scatter(df.esky_t, df.esky_c, marker=".", alpha=0.1, label="(22c)")
    ax.axline((0.8, 0.8), slope=1, c="0.7", ls="--")
    ax.set_ylabel("calibrated Brunt [-]")
    ax.set_xlabel("Shakespeare method [-]")
    ax.set_xlim(0.5, 0.95)
    ax.set_ylim(0.5, 0.95)
    ax.set_axisbelow(True)
    ax.set_title(f"{site} (npts={df.shape[0]:,})")
    leg = ax.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    filename = os.path.join("figures", f"esky_clr_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    # plt.show()

    pdf = df.loc[df.cs_period].copy()
    pdf["lw_c_22a"] = pdf.esky_day * SIGMA * np.power(pdf.t_a, 4)

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5)
    ax.scatter(pdf.lw_c_t, pdf.lw_c_22a, marker=".", alpha=0.5, label="22a")
    ax.scatter(pdf.lw_c_t, pdf.lw_c, marker=".", alpha=0.5, label="22c")
    ax.axline((300, 300), slope=1, c="0.7", ls="--")
    ax.set_ylabel("calibrated Brunt [W/m$^2$]")
    ax.set_xlabel("Shakespeare method [W/m$^2$]")
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(ymin, ymax)
    ax.set_axisbelow(True)
    leg = ax.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_title(f"{site} daytime clear (npts={pdf.shape[0]:,})")
    filename = os.path.join("figures", f"LW_clr_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

    print(pdf.shape[0])
    rmse = np.sqrt(mean_squared_error(pdf.lw_s, pdf.lw_c))
    print(rmse)
    rmse = np.sqrt(mean_squared_error(pdf.lw_s, pdf.lw_c_22a))
    print(rmse)
    rmse = np.sqrt(mean_squared_error(pdf.dw_ir, pdf.lw_c_t))
    print(rmse)
    rmse = np.sqrt(mean_squared_error(pdf.lw_s, pdf.lw_c_t))
    print(rmse)
    return None


if __name__ == "__main__":
    print()