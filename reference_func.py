"""Keep functions not in use but helpful for referencing"""

from main import *
import scipy
import pvlib
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from corr26b import get_tsky, join_surfrad_asos, shakespeare, \
    shakespeare_comparison, import_cs_compare_csv, fit_linear
from fraction import fe_lt, fi_lt

from constants import *
from fig3 import get_atm_p


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
    df = shakespeare_comparison(site=site)

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


def plot_gridded_H():
    # Figure of gridded H
    filename = os.path.join("data", "shakespeare", "data.mat")
    f = scipy.io.loadmat(filename)
    h = np.flip(f["H"], axis=1)
    xx = np.rot90(h)

    fig, ax = plt.subplots(figsize=(15, 5))
    c = ax.imshow(xx, norm="log", vmin=1000, vmax=25000)
    fig.colorbar(c, label="H [m]")
    ax.set_xticks([])
    ax.set_yticks([])
    filename = os.path.join("figures", "gridded_H.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def print_H_and_He_values():
    # explore h and h_e values
    site = "BON"
    lat1 = SURFRAD[site]["lat"]
    lon1 = SURFRAD[site]["lon"]
    h1, spline = shakespeare(lat1, lon1)
    pa = 900e2  # Pa
    p_ratio = pa / P_ATM
    he = (h1 / np.cos(40.3 * np.pi / 180)) * (p_ratio ** 1.8)
    he_p0 = (h1 / np.cos(40.3 * np.pi / 180))
    print(site)
    print(f"H={h1:.2f}, He_900={he:.2f}, He_1bar={he_p0:.2f}")
    return None


def plot_li_vs_sh():
    # produce TvPw_LivSH_{site} and TvRH_LivSh_{site}
    site = "BOU"
    df = shakespeare_comparison(site=site)

    # FIGURE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    plt.subplots_adjust(hspace=0.05)
    ax = axes[0]
    ax.grid(alpha=0.3)
    c0 = ax.scatter(
        df.rh, df.t_a, c=df.esky_c - df.esky_t,
        alpha=0.7, marker=".", vmin=0.01, vmax=0.08
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("RH [-]")
    ax.set_ylim(255, 315)
    ax.set_ylabel("T [K]")
    ax.set_axisbelow(True)
    s = r"$\Delta \varepsilon = \varepsilon_{Li} - \varepsilon_{Sh}$ [-]"
    fig.colorbar(
        c0, ax=ax, label=s, extend="both")
    ax.set_title(f"{site}, difference in esky_c", loc="left")

    ax = axes[1]
    ax.grid(alpha=0.3)
    c1 = ax.scatter(
        df.rh, df.t_a, c=df.lw_c, marker=".", alpha=0.7, vmin=150, vmax=450)
    ax.set_xlim(0, 100)
    ax.set_xlabel("RH [-]")
    ax.set_ylim(255, 315)
    ax.set_title("LW_c from Li", loc="left")
    fig.colorbar(c1, ax=axes[1], label="LW$_{c}$ [W/m$^2$]", extend="both")
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.grid(alpha=0.3)
    c = ax.scatter(
        df.rh, df.t_a, c=df.lw_c - df.lw_c_t,
        alpha=0.7, marker=".", vmin=5, vmax=30
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("RH [-]")
    ax.set_ylim(255, 315)
    ax.set_axisbelow(True)
    fig.colorbar(
        c, ax=axes[2], label=r"$\Delta$ LW$_c$ [W/m$^2$]", extend="both")
    ax.set_title("Difference in LW_c (Li - Sh)")
    filename = os.path.join("figures", f"TvRH_LivSh_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    # Plot of e_sky differences as a function of P_w
    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    c = ax.scatter(df.pw_hpa, df.t_a, marker=".", alpha=0.7, c=df.esky_c - df.esky_t)
    ax.set_xlabel("P$_w$ [hPa]")
    ax.set_ylabel("T [K]")
    ax.set_title(f"{site}")
    ax.set_axisbelow(True)
    fig.colorbar(c, label=r"$\Delta \varepsilon$")
    filename = os.path.join("figures", f"TvPw_LivSh_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def esky_clr_vs_params():
    site = "BON"
    df = shakespeare_comparison(site=site)

    df["e_act"] = df.dw_ir / (SIGMA * np.power(df.t_a, 4))
    df["e_act_s"] = df.lw_s / (SIGMA * np.power(df.t_a, 4))

    f = df.loc[~df.cs_period]
    t = df.loc[df.cs_period]  # clear skies only
    df = t.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["pw_hpa"]
    ax.scatter(x, df.e_act, marker=".", c="0.8", alpha=0.9,
               label=r"$\varepsilon_{act}$")
    ax.scatter(x, df.e_act_s, marker="^", c="0.8", alpha=0.9,
               label=r"$\varepsilon_{act,s}$")
    ax.scatter(x, df.esky_c, marker=".", label=r"$\varepsilon_{B}$")
    ax.scatter(x, df.esky_t, marker=".", label=r"$\varepsilon_{\tau}$")
    ax.set_ylabel(r"$\varepsilon$ [-]")
    ax.set_xlabel(r"$p_w$ [hPa]")
    leg = ax.legend(loc="lower right")
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_ylim(0.6, 1)
    ax.set_title(f"{site} clr")
    filename = os.path.join("figures", f"e_v_pw_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    rmse = np.sqrt(mean_squared_error(df.esky_c, df.e_act_s))
    print(rmse)
    rmse = np.sqrt(mean_squared_error(df.esky_t, df.e_act))
    print(rmse)

    rmse = np.sqrt(mean_squared_error(df.lw_s, df.lw_c))
    print(rmse)
    rmse = np.sqrt(mean_squared_error(df.dw_ir, df.lw_c_t))
    print(rmse)

    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    c = ax.scatter(df.dw_ir, df.t_a, c=df.e_act, marker=".")
    ax.set_xlim(100, 600)
    ax.set_ylim(250, 320)
    ax.set_ylabel("Temperature [K]")
    ax.set_xlabel("DLW [W/m$^2$]")
    ax.set_axisbelow(True)
    ax.set_title(f"{site} clr")
    fig.colorbar(c, label=r"$\varepsilon = DLW / \sigma T^4$")
    filename = os.path.join("figures", f"TvsDLW_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(alpha=0.3)
    ax.scatter(f.t_a, f.e_act, label="non-CS")
    ax.scatter(t.t_a, t.e_act, label="CS")
    ax.set_ylabel(r"$\varepsilon = DLW / \sigma T^4$")
    ax.set_xlabel("Temperature [K]")
    ax.legend()
    ax.set_axisbelow(True)
    filename = os.path.join("figures", f"e_v_T_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(alpha=0.3)
    ax.scatter(f.pw_hpa, f.e_act, label="non-CS")
    ax.scatter(t.pw_hpa, t.e_act, label="CS")
    ax.set_ylabel(r"$\varepsilon = DLW / \sigma T^4$")
    ax.set_xlabel(r"$p_w$ [hPa]")
    ax.legend()
    ax.set_axisbelow(True)
    filename = os.path.join("figures", f"e_v_pw_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def print_lw_clr_err_by_site():
    df = import_cs_compare_csv("cs_compare_2012.csv")
    # print out errors by site
    b_rmse = []
    b_mbe = []
    t_rmse = []
    t_mbe = []
    for s, q in df.groupby(df.site):
        # if s != "BOU":
        rmse = np.sqrt(mean_squared_error(q.lw_s, q.lw_c))
        mbe = compute_mbe(q.lw_s.values, q.lw_c.values)
        b_rmse.append(rmse)
        b_mbe.append(mbe)
        print(s, "(Brunt then tau)")
        print(f"rmse={rmse:.2f}, mbe={mbe:.2f}")
        rmse = np.sqrt(mean_squared_error(q.dw_ir, q.lw_c_t))
        mbe = compute_mbe(q.dw_ir.values, q.lw_c_t.values)
        print(f"rmse={rmse:.2f}, mbe={mbe:.2f}\n")
        t_rmse.append(rmse)
        t_mbe.append(mbe)
    print(np.mean(np.array(b_rmse)), np.mean(np.array(b_mbe)))
    print(np.mean(np.array(t_rmse)), np.mean(np.array(t_mbe)))
    return None


def afgl_from_longwave_github():
    filename = os.path.join("data", "raw.csv")
    colnames = ["alt_km", "pres_mb", "temp_k", "density_cm-3", "h2o_ppmv",
                "o3_ppmv", "n2o_ppmv", "co_ppmv", "ch4_ppmv"]
    df = pd.read_csv(filename, names=colnames, index_col=False)
    filename = os.path.join("data", "afgl_midlatitude_summer.csv")
    df.to_csv(filename, index=False)
    return None


def plot_leinhard_fig1():
    lt_cm = np.geomspace(0.01, 1.5, 100)
    fi = []
    fe = []
    for x in lt_cm:
        fe.append(fe_lt(x * 1e4))
        fi.append(fi_lt(x * 1e4))
    fi = np.array(fi)
    fe = np.array(fe)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(alpha=0.3)
    ax.plot(lt_cm, fi, c="b", label=r"$f_i$")
    ax.plot(lt_cm, fe, c="r", label=r"$f_e$")
    ax.plot(lt_cm, fi - fe, c="g", label=r"$f_i - f_e$")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\lambda$T (cm K)")
    ax.legend()
    filename = os.path.join("figures", "lienhard_f1.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


# 4/7/23
def plot_lwadjustment_vs_t_ir():
    # originally generated in corr26b.py
    filename = os.path.join("data", "tsky_table_3_50.csv")
    f = pd.read_csv(filename)

    n = 20
    ta = np.linspace(240, 315, n)
    ir = np.linspace(300, 400, n)
    lw_corr = np.zeros((n, n))
    lw_adj = np.zeros((n, n))
    e_adj = np.zeros((n, n))
    for i in range(len(ta)):
        for j in range(len(ir)):
            # ts = get_tsky(ta[i], ir[j])[1]
            ts = np.interp(ir[j], f['ir_meas'].values, f['tsky'].values)
            lw_corr[i, j] = SIGMA * np.power(ts, 4)
            lw_adj[i, j] = lw_corr[i, j] / ir[j]
            e_meas = ir[j] / (SIGMA * np.power(ta[i], 4))
            e_corr = lw_corr[i, j] / (SIGMA * np.power(ta[i], 4))
            e_adj[i, j] = e_corr / e_meas

    # lw_corr / np.tile(ir, 5).reshape(5, -1)
    fig, ax = plt.subplots()
    c = ax.imshow(lw_adj)
    ax.set_xlabel("LW meas [W/m^2]")
    ax.set_ylabel("Ta [K]")
    ylabels = [f"{x:.2f}" for x in ta]
    xlabels = [f"{x:.2f}" for x in ir]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels, rotation=90)
    fig.colorbar(c, label="LW corr / LW meas [-]")
    # filename = os.path.join("figures", "LWadjustment_vs_T_IR.png")
    filename = os.path.join("figures", "LWadjustment_vs_IR.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    # # Essentially the exact same figure
    # fig, ax = plt.subplots()
    # c = ax.imshow(e_adj)
    # ax.set_xlabel("LW meas [W/m^2]")
    # ax.set_ylabel("Ta [K]")
    # ylabels = [f"{x:.2f}" for x in ta]
    # xlabels = [f"{x:.2f}" for x in ir]
    # ax.set_xticks(range(n))
    # ax.set_yticks(range(n))
    # ax.set_yticklabels(ylabels)
    # ax.set_xticklabels(xlabels, rotation=90)
    # fig.colorbar(c, label="e corr / e meas [-]")
    return None


def plot_berdahl_fig1_tdp():
    tdp = np.linspace(-12, 23, 15)  # degC
    pw = tdp2pw(tdp + 273.15)  # Pa
    pw_hpa = pw / 100

    li_adj = 0.585 + 0.057 * np.sqrt(pw_hpa)
    bermar = 0.564 + 0.059 * np.sqrt(pw_hpa)
    alados = 0.612 + 0.044 * np.sqrt(pw_hpa)
    sellers = 0.605 + 0.048 * np.sqrt(pw_hpa)
    li_new = 0.589 + 0.055 * np.sqrt(pw_hpa)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(alpha=0.1)
    ax.plot(tdp, li_adj, label="Li (adjusted)")
    ax.plot(tdp, sellers, label="Sellers, 1965")
    ax.plot(tdp, bermar, label="Berdahl and Martin, 1984")
    ax.plot(tdp, alados, label="Alados, 2012")
    ax.plot(tdp, li_new, ls="--", c="0.8", label="0.589 + 0.055sqrt(Pw)")
    ax.legend()
    ax.set_ylim(0.6, 0.9)
    ax.set_xlabel("Dew point temperature [deg C]")
    ax.set_ylabel("Emittance [-]")
    ax.set_axisbelow(True)
    # plt.show()
    filename = os.path.join("figures", "berdahl_fig1_tdp.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_f3_f4_comparisons():
    f3 = pd.read_csv(os.path.join("data", "tsky_table_3_50.csv"))
    f4 = pd.read_csv(os.path.join("data", "tsky_table_4_50.csv"))
    f3["fl"] = f3.ir_meas / (SIGMA * np.power(f3.tsky, 4))
    f4["fl"] = f4.ir_meas / (SIGMA * np.power(f4.tsky, 4))
    f3 = f3.set_index("ir_meas")
    f4 = f4.set_index("ir_meas")
    df = f3.join(f4, lsuffix="_f3", rsuffix="_f4")

    df["d_tsky"] = np.abs(df.tsky_f3 - df.tsky_f4)
    df["d_fl"] = np.abs(df.fl_f3 - df.fl_f4)
    df["d_lw"] = np.abs((SIGMA * np.power(df.tsky_f3, 4)) -
                        (SIGMA * np.power(df.tsky_f4, 4)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(alpha=0.3)
    dlw3 = SIGMA * np.power(df.tsky_f3, 4)
    dlw4 = SIGMA * np.power(df.tsky_f4, 4)
    ax.plot(df.index, dlw3, label="3-50")
    ax.plot(df.index, dlw4, ls="--", label="4-50")
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_xlabel("IR measured [W/m$^2$]")
    # ax.set_ylabel("T_sky [K]")
    ax.set_ylabel("DLW [W/m$^2$]")
    # ax.set_ylabel(r"$\Delta$ T_sky [K]")
    # ax.set_ylabel(r"$\Delta$ DLW [W/m$^2$]")
    ax.legend()
    filename = os.path.join("figures", "dlw_vs_ir.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_lookuptable_350():
    f3 = pd.read_csv(os.path.join("data", "tsky_table_3_50.csv"))
    f4 = pd.read_csv(os.path.join("data", "tsky_table_4_50.csv"))
    f3["fl"] = f3.ir_meas / (SIGMA * np.power(f3.tsky, 4))
    f4["fl"] = f4.ir_meas / (SIGMA * np.power(f4.tsky, 4))
    f3 = f3.set_index("ir_meas")
    f4 = f4.set_index("ir_meas")

    fig, ax = plt.subplots()
    ax.plot(f3.index, f3.tsky, c="b", lw=0.5)
    ax.plot(f4.index, f4.tsky, c="0.2", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylabel("Tsky [K]", color="b")
    ax2 = ax.twinx()
    ax2.plot(f3.index, f3.fl, c="r", lw=0.5)
    ax2.plot(f4.index, f4.fl, c="0.2", lw=0.5, ls="--", alpha=0.5)
    ax2.set_ylabel("fraction [-]", color="r")
    ax.set_xlabel("IR measured")
    filename = os.path.join("figures", "lookuptable_350.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
    return None


def plot_lwe_vs_tod():
    df = import_cs_compare_csv("cs_compare_2012.csv")

    df["e_eff"] = df.esky_c + df.de_p + df.de_t
    df["lw_eff"] = df.e_eff * SIGMA * np.power(df.t_a, 4)
    df["lw_err_eff"] = df.lw_eff - df.lw_s

    df["e_dp"] = df.esky_c + df.de_p
    df["lw_dp"] = df.e_dp * SIGMA * np.power(df.t_a, 4)
    df["lw_err_dp"] = df.lw_dp - df.lw_s

    gwc = df.loc[df.site == "GWC"].copy()
    gwc = gwc.loc[abs(gwc.t_a - 294.2) < 1].copy()

    gwc["solar_time_hour"] = pd.DatetimeIndex(gwc.solar_time.copy()).hour
    tmp = pd.DatetimeIndex(gwc.solar_time.copy())
    gwc["solar_tod"] = tmp.hour + (tmp.minute / 60) + (tmp.second / 3600)
    gwc = gwc.set_index("solar_tod")
    gwc = gwc.sample(frac=0.05, random_state=33)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    # s = gwc.loc[gwc.zen < 80]
    cb = ax.scatter(gwc.index, gwc.lw_err_b, c=gwc.t_a,
                    alpha=0.8, vmin=265, vmax=315)
    # s = gwc.loc[gwc.zen >= 80]
    # ax.scatter(s.index, s.lw_err_b, alpha=0.3, label="80 <= zen < 85")
    ax.axhline(0, c="0.8", ls="--")
    ax.set_xlabel("solar time of day")
    ax.set_ylabel("LW error after altitude correction [W/m$^2$] ")
    ax.set_ylim(-40, 40)
    ax.set_title(f"GWC, downsampled (pts={gwc.shape[0]:,})")
    fig.colorbar(cb, label="Ta [K]")
    # ax.legend()
    plt.show()
    filename = os.path.join("figures", "gwc_lwe_vs_tod_Ta.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    gwc = gwc.loc[gwc.zen < 80]
    gwc["noon_err"] = abs(gwc.index - 12)
    print(gwc[["lw_err_e", "noon_err", "pw_hpa", "pa_hpa"]].corr())

    return None


def plot_cs2013_lwerr_vs_elev(df):
    fig, axes = plt.subplots(7, 1, figsize=(5, 8), sharex=True)
    i = 0
    for s, alt in ELEVATIONS:
        ax = axes[i]
        ax.grid(axis="x", alpha=0.3)
        pdf = df.loc[df.site == s]
        ax.axvline(0, c="k", alpha=0.9, zorder=0)
        # ax.hist(pdf.lw_err_e, bins=50, alpha=0.9)
        ax.hist(pdf.lw_err_eff, bins=50, alpha=0.9)
        rmse = np.sqrt(mean_squared_error(pdf.lw_s3, pdf.lw_eff))
        ax.set_title(f"{s} [{alt:}m] (rmse={rmse:.2f})", loc="left")
        i += 1
        ax.set_axisbelow(True)
    ax.set_xlabel("LW_pred - LW_target")
    plt.tight_layout()
    # filename = os.path.join("figures", "cs2013_LWerr_vs_elev.png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_co2_h2o_lbl():
    df = pd.DataFrame()  # create specific co2 and h20 df
    colnames = ['pw', 'H2O', 'pCO2', 'pO3', 'pAerosols',
                'pN2O', 'pCH4', 'pO2', 'pN2', 'pOverlaps']
    for i in range(N_BANDS):
        f = f"fig5_esky_ij_b{i + 1}.csv"
        filename = os.path.join("data", "ijhmt_2019_data", f)
        tmp = pd.read_csv(filename, names=colnames, header=0)
        # in fig3.py can just do `tmp = import_ijhmt_df(f)`
        tmp["CO2"] = tmp.pCO2 - tmp.H2O
        if i == 0:
            tmp = tmp[["pw", "H2O", "CO2"]]
        else:
            tmp = tmp[["H2O", "CO2"]]
        tmp = tmp.rename(columns={"H2O": f"h2o_b{i + 1}",
                                  "CO2": f"co2_b{i + 1}"})
        df = pd.concat([df, tmp], axis=1)
    pw = df.pw.to_numpy()
    co2 = df.filter(like="co2").sum(axis=1).to_numpy()
    h2o = df.filter(like="h2o").sum(axis=1).to_numpy()

    fig, ax = plt.subplots()
    ax.plot(pw, co2)
    ax.plot(pw, h2o)
    ax.plot(pw, co2 + h2o)
    plt.show()
    return None


def esky_format(x, c1, c2, c3):
    return c1 + (c2 * np.power(x, c3))


def curve_fit(df):
    # curve fit
    train_y = df.pOverlaps.to_numpy()
    train_x = df.pw.to_numpy()
    out = curve_fit(
        f=esky_format, xdata=train_x, ydata=train_y,
        p0=[0.5, 0.0, 0.5], bounds=(-100, 100)
    )
    c1, c2, c3 = out[0]
    c1 = c1.round(4)
    c2 = c2.round(4)
    c3 = c3.round(4)
    pred_y = esky_format(train_x, c1, c2, c3)
    rmse = np.sqrt(mean_squared_error(train_y, pred_y))
    print("(c1, c2, c3): ", c1, c2, c3)
    r2 = r2_score(df.total, df.pred_y)
    print(rmse.round(5), r2.round(5))
    return None


def plot_gwc_2012_zen80_c1c2_vs_dta():
    df = import_cs_compare_csv("cs_compare_2012.csv", site="GWC")
    # df = import_cs_compare_csv("cs_compare_2013.csv")
    # df = pd.concat([df, tmp])
    # df = df.sample(frac=0.25, random_state=96)
    df = df.loc[df.zen < 80]

    df["y"] = df.e_act_s #+ df.de_p  # - 0.6376
    df["pp"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    # de_p = df.de_p.values[0]
    # x_dt = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 8, 16])
    x_dt = np.linspace(0.5, 20, 30)
    y_c1 = []
    y_c2 = []
    y = []  # with a fixed c1
    total_pts = df.shape[0]
    frac_pts = []

    err_full = []
    err_ltd = []
    err_full_c2 = []
    err_ltd_c2 = []
    target_y = df.lw_s.to_numpy()
    df["x"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    for x in x_dt:
        s = df.loc[abs(df.t_a - 294.2) < x].copy()
        frac_pts.append(s.shape[0] / total_pts)
        # s = s.sample(1500, random_state=20)

        c1, c2 = fit_linear(s, print_out=False)
        y_c1.append(c1)
        y_c2.append(c2)
        pred_y = c1 + (c2 * s.pp)
        model = pred_y * SIGMA * np.power(s.t_a, 4)
        err_ltd.append(np.sqrt(mean_squared_error(s.lw_s.to_numpy(), model)))
        pred_y = c1 + (c2 * df.pp)
        model = pred_y * SIGMA * np.power(df.t_a, 4)
        err_full.append(np.sqrt(mean_squared_error(target_y, model)))

        s["y"] = s.y - 0.6376
        c1, c2 = fit_linear(s, set_intercept=0.6376)
        y.append(c2)
        pred_y = c1 + (c2 * s.pp)
        model = pred_y * SIGMA * np.power(s.t_a, 4)
        err_ltd_c2.append(np.sqrt(mean_squared_error(s.lw_s.to_numpy(), model)))
        pred_y = c1 + (c2 * df.pp)
        model = pred_y * SIGMA * np.power(df.t_a, 4)
        err_full_c2.append(np.sqrt(mean_squared_error(target_y, model)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 2], sharex=True)
    ax = axes[0]
    c1 = "tab:blue"
    c2 = "tab:red"
    ax.plot(x_dt, y_c1, ".-", c=c1)
    ax.axhline(0.6326, ls=":", c=c1)
    ax2 = ax.twinx()
    ax2.plot(x_dt, y_c2, ".-", c=c2, lw=2)
    ax2.axhline(1.6026, ls=":", c=c2, lw=2)
    ax2.plot(x_dt, y, "*--", c="firebrick", alpha=0.8)
    ax.set_ylabel("c1", color=c1)
    ax2.set_ylabel("c2", color=c2)
    ax.tick_params(axis="y", colors=c1)
    ax2.tick_params(axis="y", colors=c2)
    # fig, ax = plt.subplots()
    ax = axes[1]
    model = df.esky_c * SIGMA * np.power(df.t_a, 4)
    err = np.sqrt(mean_squared_error(df.lw_s, model))
    c1 = "#3A5743"
    c2 = "#AF5D63"
    ax.plot(x_dt, err_ltd, ".-", c=c1)
    ax.plot(x_dt, err_full, ".-", c=c2)
    ax.plot(x_dt, err_ltd_c2, "*--", c=c1)
    ax.plot(x_dt, err_full_c2, "*--", c=c2)
    ax.axhline(err, ls=":", c="k", alpha=0.6)
    ax2 = ax.twinx()
    ax2.bar(x_dt, frac_pts, color="tab:gray", alpha=0.5, width=0.2)
    ax2.set_ylabel("Normalized size of sample [-]", color="tab:gray")
    ax2.tick_params(axis="y", colors="tab:gray")
    ax2.set_ylim(0, 1)
    ax.set_xlabel(r"T$_a$ = 294.2 +/- $\Delta$ [K]")
    ax.set_ylabel("LW err [W/m$^2$]")

    # plt.show()
    filename = os.path.join("figures", "gwc_2012_zen80_c1c2_vs_dta.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def plot_z_vs_HandP():
    z = np.linspace(0, 2000, 20)
    p = get_atm_p(z) / 100  # hPa

    site_p = []
    site_z = []
    sites = []
    h1_p = []
    site_h = []
    for s, elev in ELEVATIONS:
        site_p.append(get_atm_p(elev) / 100)
        site_z.append(elev)
        sites.append(s)

        lat1 = SURFRAD[s]["lat"]
        lon1 = SURFRAD[s]["lon"]
        h1, spline = shakespeare(lat1, lon1)
        site_h.append(h1)
        h1_p.append(P_ATM * np.exp(-1 * elev / h1) / 100)

    site_h = np.array(site_h)
    site_z = np.array(site_z)
    site_he = (site_h / np.cos(40.3 * np.pi / 180)) * np.exp(-1.8 * site_z / site_h)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    ax = axes[0]
    ax.grid(alpha=0.3)
    ax.axvline(8500, c="0.7", lw=2, zorder=0)
    ax.scatter(8500*np.ones(len(sites)), site_z, s=10, label="H=8500m")
    ax.scatter(site_h, site_z, s=15, label="H(lat,lon)")
    ax.scatter(site_he, site_z, s=15, marker="*", label=r"H$_e$(lat,lon)")
    for i in range(len(sites)):
        ax.text(8500 - 50, site_z[i] + 5, s=sites[i], fontsize=11, ha="right")
    ax.set_ylabel("z [m]")
    ax.set_xlabel("scale height [m]")
    ax.set_xlim(left=0)
    ax.legend()

    ax = axes[1]
    ax.grid(alpha=0.3)
    ax.plot(p, z, c="0.7", lw=2, label="P=P0 e^(-z/H)", zorder=0)
    p_adj = P_ATM * np.exp(-1.8 * z / 8500) / 100
    ax.plot(p_adj, z, c="0.7", ls="--", label="P=P0 e^(-1.8z/H)", zorder=0)
    for i in range(len(sites)):
        ax.text(site_p[i] + 5, site_z[i] + 5, s=sites[i], fontsize=11)
    ax.scatter(site_p, site_z, s=10, label="P=P(H=8500m)")
    ax.scatter(h1_p, site_z, s=10, label="P=P(H(lat,lon))")
    ax.set_ylim(0, 2000)
    ax.set_xlim(right=1100)
    ax.set_xlabel("pressure [hPa]")
    ax.legend(loc="lower left")
    plt.tight_layout()

    filename = os.path.join("figures", "z_vs_HandP.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def generate_scale_heights_dict():
    height_dict = {}
    i = 0
    for site in SURF_SITE_CODES:
        lat1 = SURFRAD[site]["lat"]
        lon1 = SURFRAD[site]["lon"]
        h1, spline = shakespeare(lat1, lon1)
        height_dict[site] = h1
        i += 1
    print(height_dict)
    return None


def test_altitude_correction():
    """Compare Berdahl correction with H=8500m vs H=H(lat,lon)"""
    df = import_cs_compare_csv("cs_compare_2012.csv")
    df = df.sample(1000)
    df["pp"] = 1.01325 * (np.exp(-1 * df.elev / 8500))
    df["e_err"] = df.e_act_s - df.esky_c
    plt.scatter(df.pp, df.e_err, alpha=0.5)

    y = df.e_err.to_numpy().reshape(-1, 1)

    # c1 * p0 * (exp(-z/H) - 1)
    X8 = 1.01325 * (np.exp(-1 * df.elev / 8500) - 1).to_numpy().reshape(-1, 1)
    Xh = 1.01325 * (np.exp(-1 * df.elev / df.h) - 1).to_numpy().reshape(-1, 1)

    # c1 * (p0 exp(-z/H) - 100)
    # df["P_h"] = P_ATM * np.exp(-1 * df.elev / df.h)
    # X8 = ((df.P_rep.to_numpy() / 100) - 1000).reshape(-1, 1)
    # Xh = ((df.P_h.to_numpy() / 100) - 1000).reshape(-1, 1)

    for X in [X8, Xh]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mbe = compute_mbe(y_test, y_pred)[0]
        r2 = r2_score(y_test, y_pred)
        print(f"\nc1: {model.coef_[0][0]:.6f}")
        print(f"RMSE: {rmse:.8f}")
        print(f"MBE: {mbe:.8f}")
        print(f"R2: {r2:.8f}")
    print(f"\ntrain: {X_train.shape[0]:,}, test: {X_test.shape[0]:,}")
    return None


def esky_format_exp(x, c1, c2):
    return c1 * np.exp(-1 * c2 * x)


def compare_exp_fit():
    """
    Compare fits for emissivity
    e=c1+c2sqrt(pw), t=c1+c2pw, 1-e=c1exp(-c2pw)
    """
    df = import_cs_compare_csv("cs_compare_2013.csv")
    df = df.loc[abs(df.t_a - 294.2 < 2)].copy()
    df["pp"] = df.pw_hpa * 100 / P_ATM
    df["e_eff"] = df.e_act - df.de_p  # bring to sea level
    X_lin = np.sqrt(df.pp.to_numpy().reshape(-1, 1))
    X_tau = df.pp.to_numpy().reshape(-1, 1)
    X_exp = df.pp.to_numpy().reshape(-1, 1)
    y = df.e_eff.to_numpy().reshape(-1, 1)

    # EMISSIVITY
    X_specific = X_lin
    y_specific = y
    # LINEAR TAU
    X_specific = X_tau
    y_specific = -1 * np.log(1 - y)
    # EXPONENTIAL TAU - need to have esky_format_exp() function
    X_specific = X_exp
    y_specific = 1 - y

    # copy/paste with linear tau or emissivity
    X_train, X_test, y_train, y_test = train_test_split(
        X_specific, y_specific, test_size=0.33, random_state=42)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    c2 = model.coef_[0][0].round(4)
    c1 = model.intercept_[0].round(4)
    print(c1, c2)
    # Evaluate on training
    y_pred = c1 + (c2 * X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    # Evaluate on test
    y_pred = c1 + (c2 * X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    print(X_train.shape, X_test.shape)

    # copy/paste with exp_fit X
    X_train, X_test, y_train, y_test = train_test_split(
        X_specific.reshape(-1), y_specific.reshape(-1),
        test_size=0.33, random_state=42)
    out = curve_fit(
        f=esky_format_exp, xdata=X_train, ydata=y_train,
        p0=[0.3, 7], bounds=(-100, 100)
    )
    c1 = out[0][0].round(4)
    c2 = out[0][1].round(4)
    print(c1, c2)
    y_pred = esky_format_exp(X_train, c1, c2)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    y_pred = esky_format_exp(X_test, c1, c2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

    # ---------------
    # IJHMT data (same structure as cs_compare file)
    filename = os.path.join("data", "ijhmt_2019_data", "fig3_esky_i.csv")
    colnames = ['pw', 'H2O', 'pCO2', 'pO3', 'pAerosols',
                'pN2O', 'pCH4', 'pO2', 'pN2', 'pOverlaps']
    df = pd.read_csv(filename, names=colnames, header=0)
    # emissivity
    X_train = np.sqrt(df.pw.to_numpy().reshape(-1, 1))
    y_train = df.pOverlaps.to_numpy().reshape(-1, 1)
    # linear tau
    X_train = df.pw.to_numpy().reshape(-1, 1)
    y_train = -1 * np.log(1 - y_train)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    c2 = model.coef_[0][0].round(4)
    c1 = model.intercept_[0].round(4)
    y_pred = c1 + (c2 * X_train)
    print(c1, c2)
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # exponential tau
    X_train = df.pw.to_numpy().reshape(-1)
    y_train = 1 - df.pOverlaps.to_numpy().reshape(-1)

    out = curve_fit(
        f=esky_format_exp, xdata=X_train, ydata=y_train,
        p0=[0.0, 0.5], bounds=(-100, 100)
    )
    c1 = out[0][0].round(4)
    c2 = out[0][1].round(4)
    y_pred = esky_format_exp(X_train, c1, c2)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    return None


if __name__ == "__main__":
    print()