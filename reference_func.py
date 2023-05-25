"""Keep functions not in use but helpful for referencing"""

from main import *
import time
import scipy
import pvlib
import datetime as dt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from corr26b import get_tsky, join_surfrad_asos, shakespeare, \
    shakespeare_comparison, import_cs_compare_csv, fit_linear, three_c_fit, \
    add_solar_time, add_afgl_t0_p0, create_training_set, \
    reduce_to_equal_pts_per_site
from fraction import fe_lt, fi_lt

from constants import *
from fig3 import get_atm_p
from process import *


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
    filename = os.path.join("data", "profiles", "AFGL_midlatitude_winter.csv")
    colnames = ["alt_km", "pres_mb", "temp_k", "density_cm-3", "h2o_ppmv",
                "o3_ppmv", "n2o_ppmv", "co_ppmv", "ch4_ppmv"]
    df = pd.read_csv(filename, names=colnames, index_col=False)
    filename = os.path.join("data", "afgl_midlatitude_winter.csv")
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


def plot_p_vs_elev():
    # run in fig3.py
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

    filename = os.path.join("data", "afgl_midlatitude_summer.csv")
    af_sum = pd.read_csv(filename)
    y_sum_p = np.interp(site_z / 1000, af_sum.alt_km.values, af_sum.pres_mb.values)

    fig, ax = plt.subplots(figsize=(5, 8), sharey=True)
    ax.grid(alpha=0.3)
    ax.plot(p, z, c="0.7", lw=2, label="P=P0 e^(-z/H)", zorder=0)
    ax.plot(y_sum_p, site_z, "k")
    for i in range(len(sites)):
        ax.text(site_p[i] + 5, site_z[i] + 5, s=sites[i], fontsize=11)
    ax.scatter(site_p, site_z, s=10, label="P=P(H=8500m)")
    ax.set_ylim(0, 2000)
    ax.set_xlim(right=1100)
    ax.set_xlabel("pressure [hPa]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    filename = os.path.join("figures", "P_vs_elev_afgl.png")
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


def plot_lwerr_bin(df, mod, x, nbins=4, site=None, save_fig=False):
    if mod == "t":
        y_mod = "lw_c_t"
        y_true = "dw_ir"
        err = "lw_err_t"
        xlabel = r"$\Delta LW = LW_{\tau} - LW$ [W/m$^2$]"
    elif mod == "b":
        y_mod = "lw_c"
        y_true = "lw_s"
        err = "lw_err_b"
        xlabel = r"$\Delta LW = LW_{B} - LW_{s}$ [W/m$^2$]"
    if site is not None:
        df = df.loc[df.site == site].copy()
    if x == "pw":
        df["w_bin"] = pd.qcut(df.pw_hpa, nbins, labels=False)
    elif x == "rh":
        df["w_bin"] = pd.qcut(df.rh, nbins, labels=False)
    elif x == "tk":
        df["w_bin"] = pd.qcut(df.t_a, nbins, labels=False)
    elif x == "pa":
        df["w_bin"] = pd.qcut(df.pa_hpa, nbins, labels=False)
    # FIGURE
    fig, axes = plt.subplots(
        nbins, 1, figsize=(8, 10), sharex=True, sharey=True)
    for i in range(nbins):
        ax = axes[i]
        ax.grid(axis="x")
        q = df.loc[df.w_bin == i].copy()
        ax.hist(q[[err]], bins=30)
        ax.set_xlim(-30, 30)
        if x == "pw":
            title = f"Q{i+1}: p$_w$ [{q.pw_hpa.min():.2f}-{q.pw_hpa.max():.2f}]"
        elif x == "rh":
            title = f"Q{i + 1}: RH [{q.rh.min():.2f}-{q.rh.max():.2f}]"
        elif x == "tk":
            title = f"Q{i + 1}: T [{q.t_a.min():.2f}-{q.t_a.max():.2f}]"
        elif x == "pa":
            title = f"Q{i + 1}: P [{q.pa_hpa.min():.2f}-{q.pa_hpa.max():.2f}]"
        # ax.set_title(title, loc="left")
        ax.text(
            0.02, 0.8, s=title, backgroundcolor="1.0", size="medium",
            transform=ax.transAxes
        )
        ax.set_axisbelow(True)
        rmse = np.sqrt(mean_squared_error(q[[y_true]], q[[y_mod]]))
        mbe = compute_mbe(q[[y_true]].values, q[[y_mod]].values)[0]
        err_str = f"RMSE={rmse:.2f} W/m$^2$ \n MBE={mbe:.2f} W/m$^2$"
        ax.text(
            0.02, 0.3, s=err_str, backgroundcolor="1.0",
            transform=ax.transAxes
        )
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    if save_fig:
        if site is None:
            f = f"LWerr_{x}_bin={nbins}_{mod}.png"
        else:
            f = f"LWerr_{x}_bin={nbins}_{mod}_{site}.png"
        filename = os.path.join("figures", f)
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    return None


def plot_lwerr_bin_main():
    # GRAPH histograms of error by quartile of some humidity metric
    df = import_cs_compare_csv("cs_compare_2012.csv")
    nbins = 8
    xvar = "pw"  # ["pw", "rh", "tk", "pa"]
    # mod = "t"  # ["t", "b"] model type (tau or Brunt)
    plot_lwerr_bin(df, "b", xvar, nbins=nbins, save_fig=1)
    return None


def compare_esky_fits(p="hpa", lw="s", tra_yr=2012, val_yr=2013, rm_loc=None):
    """Train and evaluate the Brunt model fit (22a) for daytime clear sky
    samples with variations for the input parameter (p), the LW measurement
    used to determine sky emissivity (lw), the training and validation years,
    and the number of sites to incorporate in training and validation.

    Parameters
    ----------
    p : ["hpa", "scaled"], optional
        Use an input value of p_w in hPa ("hpa") or a normalzed value of
        p_w (p_w / p_0) ("scaled") as the input parameter for the Brunt
        emissivity regression.
    lw : "s" or None, optional
        If "s" then used emissivity determined from corrected LW measurements (LW_s)
    tra_yr : int, optional
        Data year to use on training.
    val_yr : int, optional
        Data year to use on evaluating fit.
    rm_loc : str or list, optional
        The defualt value means that no locations are removed and all data
        from the cs_compare csv will be used in training and validation.
        User can input a string corresponding to the SURFRAD site code or a
        list of site codes to indicate that the location(s) be removed from
        both training and validation sets.

    Returns
    -------
    None
    """
    # Compare e_sky_c fits on different pressure and LW variables
    # compare_esky_fits(p="scaled", lw="", tra_yr=2012, val_yr=2013, rm_loc=None)

    tra = import_cs_compare_csv(f"cs_compare_{tra_yr}.csv")
    val = import_cs_compare_csv(f"cs_compare_{val_yr}.csv")
    if rm_loc is not None:
        if isinstance(rm_loc, list):
            for s in rm_loc:  # remove multiple loc
                tra = tra.loc[tra.site != s]
                val = val.loc[val.site != s]
        elif isinstance(rm_loc, str):
            tra = tra.loc[tra.site != rm_loc]  # remove single loc
            val = val.loc[val.site != rm_loc]
    print(f"TRA samples: {tra.shape[0]:,}")
    print(f"VAL samples: {val.shape[0]:,}\n")

    if p == "hpa":
        tra["pp"] = np.sqrt(tra.pw_hpa)
        val["pp"] = np.sqrt(val.pw_hpa)
    elif p == "scaled":
        tra["pp"] = np.sqrt((tra.pw_hpa * 100) / P_ATM)
        val["pp"] = np.sqrt((val.pw_hpa * 100) / P_ATM)

    if lw == "s":  # corrected to LW_s
        train_y = tra.e_act_s.to_numpy()
        val_y = val.e_act_s.to_numpy()
    else:
        train_y = tra.e_act.to_numpy()
        val_y = val.e_act.to_numpy()

    train_x = tra[["pp"]].to_numpy()
    model = LinearRegression(fit_intercept=True)
    model.fit(train_x, train_y)
    c2 = model.coef_[0].round(4)
    c1 = model.intercept_.round(4)
    pred_y = c1 + (c2 * tra.pp)
    rmse = np.sqrt(mean_squared_error(train_y, pred_y))
    mbe = compute_mbe(train_y, pred_y)
    print("(c1, c2): ", c1, c2)
    print(f"Training RMSE:{rmse:.4f} and MBE:{mbe:.4f}\n")
    pred_y = c1 + (c2 * val.pp)
    rmse = np.sqrt(mean_squared_error(val_y, pred_y))
    print(f"Validation RMSE:{rmse:.4f}")
    return None


def iterative_alt_corrections():
    # check iterative solve of c1, c2, and altitude correction const
    niter = 6  # number of times to fit c1, c2, c3
    constants = np.zeros((3, 1 + niter * 2))
    rmses = []

    # Form training and validation sets
    tra = pd.DataFrame()
    val = pd.DataFrame()
    # rs 11, 90 for first trial
    # rs = 323 for second
    for yr in [2011, 2012, 2013]:
        tmp = import_cs_compare_csv(f"cs_compare_{yr}.csv")
        tmp = tmp.loc[abs(tmp.t_a - 294.2) < 2].copy()
        tmp = tmp.set_index("solar_time")
        tmp = tmp.loc[tmp.index.hour > 8]  # solar time > 8am
        tmp = tmp[["pw_hpa", "elev", "e_act"]].copy()
        tmp = tmp.sample(frac=0.2, random_state=323)  # reduce sample

        # Split DataFrame into 70% and 30%
        tmp_70 = tmp.sample(frac=0.7, random_state=323)
        tmp_30 = tmp.drop(tmp_70.index)

        tra = pd.concat([tra, tmp_70])
        val = pd.concat([val, tmp_30])

    tra["pp"] = np.sqrt(tra.pw_hpa * 100 / P_ATM)
    val["pp"] = np.sqrt(val.pw_hpa * 100 / P_ATM)
    train_x = tra.pp.to_numpy().reshape(-1, 1)
    P_ATM_bar = P_ATM / 100000

    # initial fit a linear model
    model = LinearRegression(fit_intercept=True)
    model.fit(train_x, tra.e_act.to_numpy().reshape(-1, 1))
    c2 = model.coef_[0].round(4)[0]
    c1 = model.intercept_.round(4)[0]
    pred_y = c1 + (c2 * val.pp)
    rmse = np.sqrt(mean_squared_error(val.e_act.to_numpy(), pred_y))
    print(c1, c2, f"(RMSE: {rmse:.4f})")
    i = 0
    constants[0, i] = c1
    constants[1, i] = c2
    c3 = 0
    constants[2, i] = 0  # no correction at first
    rmses.append(rmse.round(5))
    i += 1

    for counter in range(niter):
        # now fit an altitude correction
        model = LinearRegression(fit_intercept=False)
        de_p = c3 * P_ATM_bar * (np.exp(-1 * tra.elev.to_numpy() / 8500) - 1)
        y = (c1 + (c2 * tra.pp.to_numpy())) + de_p
        y_err = y - tra.e_act.to_numpy()
        x = P_ATM_bar * (np.exp(-1 * tra.elev.to_numpy() / 8500) - 1)
        model.fit(x.reshape(-1, 1), -1 * y_err.reshape(-1, 1))
        c3 = model.coef_[0].round(4)[0]
        de_p = c3 * P_ATM_bar * (np.exp(-1 * val.elev.to_numpy() / 8500) - 1)
        pred_y = (c1 + (c2 * val.pp)) + de_p
        rmse = np.sqrt(mean_squared_error(val.e_act.to_numpy(), pred_y))
        print(c3, f"(RMSE: {rmse:.4f})")
        constants[0, i] = c1
        constants[1, i] = c2
        constants[2, i] = c3
        rmses.append(rmse.round(5))
        i += 1

        # fit linear model with altitude correction
        model2 = LinearRegression(fit_intercept=True)
        de_p = c3 * P_ATM_bar * (np.exp(-1 * tra.elev.to_numpy() / 8500) - 1)
        y = tra.e_act.to_numpy() - de_p  # bring to sea level for fit
        model2.fit(train_x, y.reshape(-1, 1))
        c2 = model2.coef_[0].round(4)[0]
        c1 = model2.intercept_.round(4)[0]
        de_p = c3 * P_ATM_bar * (np.exp(-1 * val.elev.to_numpy() / 8500) - 1)
        pred_y = c1 + (c2 * val.pp) + de_p
        rmse = np.sqrt(mean_squared_error(val.e_act.to_numpy(), pred_y))
        print(c1, c2, f"(RMSE: {rmse:.4f})")
        constants[0, i] = c1
        constants[1, i] = c2
        constants[2, i] = c3
        rmses.append(rmse.round(5))
        i += 1

    x = np.arange(len(rmses))
    rmses = np.array(rmses)

    fig, axes = plt.subplots(4, 1, figsize=(10, 6), height_ratios=[2, 1, 1, 1], sharex=True)
    axes[0].plot(x, rmses, ".-")
    axes[1].plot(x, constants[0, :], ".-")
    axes[2].plot(x, constants[1, :], ".-")
    axes[3].plot(x, constants[2, :], ".-")
    axes[0].set_ylabel("RMSE on VAL")
    axes[1].set_ylabel("c1")
    axes[2].set_ylabel("c2")
    axes[3].set_ylabel("c3")
    axes[3].set_xticks(x)
    axes[2].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))
    axes[3].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    filename = os.path.join("figures", "iterative_alt_corrections3_filter.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_convergence():
    df = create_training_set(
        year=[2010, 2011, 2012, 2013], all_sites=True, site=None,
        temperature=False, cs_only=True, pct_clr_min=0.0
    )
    test = df.loc[df.index.year == 2013].copy()  # make test set
    df = df.loc[df.index.year != 2013].copy()

    sizes = np.geomspace(100, 100000, 20)
    n_iter = 50  # per sample size
    c1_vals = np.zeros((len(sizes), n_iter))
    c2_vals = np.zeros((len(sizes), n_iter))
    c3_vals = np.zeros((len(sizes), n_iter))
    rmses = np.zeros((len(sizes), n_iter))
    for i in range(len(sizes)):
        for j in range(n_iter):
            train = df.sample(n=int(sizes[i]))
            c1, c2, c3 = three_c_fit(train)
            c1_vals[i, j] = c1
            c2_vals[i, j] = c2
            c3_vals[i, j] = c3

            # evaluate on test
            de_p = c3 * (np.exp(-1 * test.elev / 8500) - 1)
            pred_e = c1 + (c2 * test.x) + de_p
            pred_y = SIGMA * np.power(test.t_a, 4) * pred_e
            rmse = np.sqrt(mean_squared_error(
                test.dw_ir.to_numpy(), pred_y.to_numpy()))
            rmses[i, j] = rmse

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.set_xscale("log")
    ax.fill_between(sizes, c1_vals.min(axis=1), c1_vals.max(axis=1), alpha=0.5)
    ax.plot(sizes, c1_vals.mean(axis=1))
    ax.set_ylabel("c1")
    ax.set_ylim(0.56, 0.64)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_xscale("log")
    ax.fill_between(sizes, c2_vals.min(axis=1), c2_vals.max(axis=1), alpha=0.5)
    ax.plot(sizes, c2_vals.mean(axis=1))
    ax.set_ylabel("c2")
    ax.set_ylim(1.2, 1.8)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.set_xscale("log")
    ax.fill_between(sizes, rmses.min(axis=1), rmses.max(axis=1), alpha=0.5)
    ax.plot(sizes, rmses.mean(axis=1))
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("RMSE [W/m$^2$]")
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_xlim(sizes[0], sizes[-1])

    filename = os.path.join("figures", "convergence.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_t0_p0_per_site():
    overlay_profile = False
    filter_ta = True
    alpha_background = 0.2 if overlay_profile else 1.0
    pm_p_mb = 20  # plus minus pressure (mb)

    if overlay_profile:
        filename = os.path.join("data", "afgl_midlatitude_summer.csv")
        af_sum = pd.read_csv(filename)
        filename = os.path.join("data", "afgl_midlatitude_winter.csv")
        af_win = pd.read_csv(filename)

    df = create_training_set(
        year=[2011, 2012], all_sites=True, site=None, temperature=False,
        cs_only=True, pct_clr_min=0.3
    )
    df = reduce_to_equal_pts_per_site(df)

    if filter_ta:
        df = df.loc[abs(df.t_a - df.afgl_t0) <= 2].copy()
    pdf = df.sample(2000, random_state=22)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = pdf.loc[pdf.site == s]
        afgl_t = group.afgl_t0.values[0]
        afgl_p = group.afgl_p0.values[0]

        x = np.linspace(group.t_a.min(), group.t_a.max(), 2)
        # x = np.linspace(afgl_t - 2, afgl_t + 2, 3)
        ax.fill_between(
            x, afgl_p - pm_p_mb, afgl_p + pm_p_mb,
            fc=SEVEN_COLORS[i], alpha=0.2 * alpha_background, zorder=0)
        ax.axhline(afgl_p, c=SEVEN_COLORS[i], label=s, zorder=1,
                   alpha=alpha_background)
        ax.axvline(afgl_t, c=SEVEN_COLORS[i], zorder=1,
                   alpha=alpha_background)
        ax.scatter(
            group.t_a, group.pa_hpa, marker=".", alpha=0.8 * alpha_background,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)
        i += 1
    ymin, ymax = ax.get_ylim()
    if overlay_profile:
        alt_x = np.linspace(0, 2, 10)  # km
        y_sum_p = np.interp(alt_x, af_sum.alt_km.values, af_sum.pres_mb.values)
        x_sum_t = np.interp(alt_x, af_sum.alt_km.values, af_sum.temp_k.values)
        ax.plot(x_sum_t, y_sum_p, ls="--", c="0.3", label="AFGL\nsummer")
        y_sum_p = np.interp(alt_x, af_win.alt_km.values, af_win.pres_mb.values)
        x_sum_t = np.interp(alt_x, af_win.alt_km.values, af_win.temp_k.values)
        ax.plot(x_sum_t, y_sum_p, ls="-.", c="0.3", label="AFGL\nwinter")
        ax.set_ylim(ymin, ymax)
    lgd = ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    for lh in lgd.legendHandles:
        lh.set_alpha(1)
    ax.set_xlabel("T$_a$ [K]")
    ax.set_ylabel("P [mb]")
    ax.invert_yaxis()
    plt.tight_layout()
    suffix = "_afgl" if overlay_profile else ""
    suffix += "_filterTa" if filter_ta else ""
    filename = os.path.join("figures", f"ts_ps_per_site{suffix}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_ts_ps_season():
    # variation on t0/p0 original, without season distinction
    overlay_profile = True
    filter_ta = False
    alpha_background = 0.2 if overlay_profile else 1.0
    pm_p_mb = 20  # plus minus pressure (mb)
    ms = 15  # marker size

    if overlay_profile:
        filename = os.path.join("data", "afgl_midlatitude_summer.csv")
        af_sum = pd.read_csv(filename)
        filename = os.path.join("data", "afgl_midlatitude_winter.csv")
        af_win = pd.read_csv(filename)

    df = create_training_set(
        year=[2010, 2011, 2012], temperature=False, cs_only=True,
        filter_pct_clr=0.05, filter_npts_clr=0.2, drive="server4"
    )
    df = reduce_to_equal_pts_per_site(df)

    if filter_ta:
        df = df.loc[abs(df.t_a - df.afgl_t0) <= 2].copy()

    # filter per season Winter
    # pdf = df.sample(2000, random_state=22)
    month_season = {}
    for i in range(1, 13):
        if i in [12, 1, 2]:
            month_season[i] = "winter"
        elif i in [3, 4, 5]:
            month_season[i] = "spring"
        elif i in [6, 7, 8]:
            month_season[i] = "summer"
        elif i in [9, 10, 11]:
            month_season[i] = "fall"
    df["month"] = df.index.month
    df["season"] = df["month"].map(month_season)

    pdf = df.loc[df.season == "winter"].sample(200, random_state=22)
    tmp = df.loc[df.season == "summer"].sample(200, random_state=22)
    pdf = pd.concat([pdf, tmp])
    tmp = df.loc[(df.season == "spring")].sample(100, random_state=22)
    pdf = pd.concat([pdf, tmp])
    tmp = df.loc[(df.season == "fall")].sample(100, random_state=22)
    pdf = pd.concat([pdf, tmp])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = pdf.loc[pdf.site == s]
        afgl_t = group.afgl_t0.values[0]
        afgl_p = group.afgl_p0.values[0]

        x = np.linspace(group.t_a.min(), group.t_a.max(), 2)
        # x = np.linspace(afgl_t - 2, afgl_t + 2, 3)
        ax.fill_between(
            x, afgl_p - pm_p_mb, afgl_p + pm_p_mb,
            fc=SEVEN_COLORS[i], alpha=0.2 * alpha_background, zorder=0)
        ax.axhline(afgl_p, c=SEVEN_COLORS[i], label=s, zorder=1,
                   alpha=alpha_background)
        ax.axvline(afgl_t, c=SEVEN_COLORS[i], zorder=1,
                   alpha=alpha_background)

        ax.scatter(
            group.loc[group.season == "fall"].t_a,
            group.loc[group.season == "fall"].pa_hpa, marker="*", s=ms,
            alpha=0.8 * alpha_background * 0.5,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)
        ax.scatter(
            group.loc[group.season == "spring"].t_a,
            group.loc[group.season == "spring"].pa_hpa, marker="*", s=ms,
            alpha=0.8 * alpha_background * 0.5,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)

        ax.scatter(
            group.loc[group.season == "summer"].t_a,
            group.loc[group.season == "summer"].pa_hpa, marker="o", s=ms,
            alpha=0.8 * alpha_background,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)
        ax.scatter(
            group.loc[group.season == "winter"].t_a,
            group.loc[group.season == "winter"].pa_hpa, marker="^", s=ms,
            alpha=0.8 * alpha_background,
            c=SEVEN_COLORS[i], ec="0.5", zorder=10)
        i += 1
    ymin, ymax = ax.get_ylim()
    if overlay_profile:
        alt_x = np.linspace(0, 2, 10)  # km
        y_sum_p = np.interp(alt_x, af_sum.alt_km.values, af_sum.pres_mb.values)
        x_sum_t = np.interp(alt_x, af_sum.alt_km.values, af_sum.temp_k.values)
        ax.plot(x_sum_t, y_sum_p, ls="--", c="0.3", label="AFGL\nsummer")
        y_sum_p = np.interp(alt_x, af_win.alt_km.values, af_win.pres_mb.values)
        x_sum_t = np.interp(alt_x, af_win.alt_km.values, af_win.temp_k.values)
        ax.plot(x_sum_t, y_sum_p, ls="-.", c="0.3", label="AFGL\nwinter")
        ax.set_ylim(ymin, ymax)

    ax.plot([], [], marker="o", color="0.2", ls="none", label="summer")
    ax.plot([], [], marker="^", color="0.2", ls="none", label="winter")
    ax.plot([], [], marker="*", color="0.2", ls="none", label="spring/fall")
    lgd = ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    for lh in lgd.legend_handles:
        lh.set_alpha(1)
    ax.set_xlabel("T$_a$ [K]")
    ax.set_ylabel("P [mb]")
    ax.invert_yaxis()
    plt.tight_layout()
    suffix = "_afgl" if overlay_profile else ""
    suffix += "_filterTa" if filter_ta else ""
    # filename = os.path.join("figures", f"ts_ps_season{suffix}.png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def rh_boxplot():
    df = import_cs_compare_csv("cs_compare_2012.csv")
    data = []
    for s in SURF_SITE_CODES:
        data.append(df.loc[df.site == s, "rh"].to_numpy())

    fig, ax = plt.subplots()
    ax.boxplot(
        data, labels=SURF_SITE_CODES, patch_artist=True,
        boxprops={'fill': True, 'facecolor': 'white'},
        medianprops={'color': 'steelblue'},
    )
    plt.show()
    filename = os.path.join("figures", "rh_boxplot.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def lwe_vs_tod_rh():
    """
    Can be modified to show error over datetime. Colormap can be changed to
    values other than RH but labels should be changed accordingly.
    """
    # LW ERROR plots vs TOD
    s = "GWC"
    collapse_tod = True  # True: all data by TOD, False: data by datetime

    df = import_cs_compare_csv("cs_compare_2012.csv", site=s)
    df["standard_time"] = df.index.to_numpy()
    df = df.set_index("solar_time")
    df = df.loc[df.index.hour > 8].copy()  # remove data before 8am solar
    df = add_afgl_t0_p0(df)

    tmp = np.log(df.pw_hpa * 100 / 610.94)
    df["tdp"] = 273.15 + ((243.04 * tmp) / (17.625 - tmp))
    df["dtdp"] = df.t_a - df.tdp

    print(df.shape)  # this filter is causing issuess
    df = df.loc[(abs(df.t_a - df.afgl_t0) <= 2) &
                (abs(df.pa_hpa - df.afgl_p0) <= 50)].copy()
    # df = df.loc[abs(df.t_a - df.afgl_t0) <= 2].copy()
    # df = df.loc[abs(df.t_a - 294.2) <= 2]
    print(df.shape)

    # find linear fit
    df["x"] = np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["y"] = df.e_act
    c1, c2, c3 = three_c_fit(df)
    print(c1, c2, c3)
    df["de_p"] = c3 * P_ATM_BAR * (np.exp(-1 * df.elev / 8500) - 1)
    df["y"] = (c1 + c2 * df.x)  # - df.de_p  # bring to sea level

    df["ir_pred"] = SIGMA * np.power(df.t_a, 4) * df.y
    df["lw_err"] = df.ir_pred - df.dw_ir
    df["abs_lw_err"] = abs(df.ir_pred - df.dw_ir)

    if df.shape[0] > 10000:
        pdf = df.sample(5000, random_state=23)
    else:
        pdf = df.copy()

    # Changes index to solar time of day
    if collapse_tod:
        pdf["solar_tod"] = \
            pdf.index.hour + (pdf.index.minute / 60) + (pdf.index.second / 3600)
        pdf = pdf.set_index("solar_tod")
    # df = df.sample(frac=0.25, random_state=33)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    cb = ax.scatter(
        pdf.index, pdf.lw_err, c=pdf.rh, alpha=0.8, marker=".",
        vmin=0, vmax=100
    )
    ax.axhline(0, c="0.8", ls="--")
    ax.set_xlabel("solar time of day")
    ax.set_ylabel("LW error [W/m$^2$] ")
    ax.set_ylim(-40, 40)
    title = f"{s} (pts={pdf.shape[0]:,}), ST>8, T~T$_0$, P~P$_0$"
    title += f", [{c1}, {c2}]"
    ax.set_title(title, loc="left")
    fig.colorbar(cb, label="RH [%]")
    plt.show()
    if collapse_tod:
        filename = os.path.join("figures", f"{s.lower()}_lwe_vs_tod_rh.png")
    else:
        filename = os.path.join("figures", f"{s.lower()}_2012_day_error.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def day_site_date_plot():
    # EXPLORE SINGLE DAY
    s = "BOU"  # site
    c1, c2 = 0.5478, 1.9195  # constants for BOU
    # c1, c2 = 0.5861, 1.6461  # constants for PSU
    # c1, c2 = 0.6271, 1.3963  # constants for GWC
    plot_date = dt.date(2012, 11, 19)  # date to plot

    df = shakespeare_comparison(s, plot_date.year)
    df["site"] = s
    df = add_solar_time(df)
    df = df.set_index("solar_time")

    df = df.loc[df.index.date == plot_date].copy()
    df["y"] = c1 + c2 * np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["lw_pred"] = df.y * SIGMA * np.power(df.t_a, 4)

    # FIGURE
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.05)
    ax = axes[0]
    ax.scatter(df.index, df.dw_ir, marker=".", label="measured LW")
    ax.scatter(df.index, df.lw_pred, marker=".", c="0.6", label="modeled LW")
    ax.legend(loc="lower right")
    date_str = plot_date.strftime("%m-%d-%Y")
    title = f"{s} {date_str}"
    ax.set_title(title, loc="left")

    ax = axes[1]
    ax.plot(df.index, df.DNI_m, label="DNI")
    ax.plot(df.index, df.GHI_m, label="GHI")
    ax.fill_between(
        df.index, 0, df.GHI_m, where=df.cs_period, fc="0.7", alpha=0.4,
        label="CS"
    )
    ax.set_ylim(0, 1200)
    ax.legend()

    ax = axes[2]
    ax2 = ax.twinx()
    ln1 = ax.plot(df.index, df.t_a, marker=".", label="T [K]")
    ln2 = ax2.plot(df.index, df.rh, marker=".", c="k", label="RH [%]")
    ax2.set_ylim(0, 100)
    lns = ln1 + ln2
    labels = [x.get_label() for x in lns]
    ax.legend(lns, labels, framealpha=0.95, loc="upper right")
    plt.show()

    date_str = plot_date.strftime("%Y%m%d")
    filename = os.path.join("figures", f"day_{s}_{date_str}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def lw_err_per_day():
    # EXPLORE stricter CS period filter
    s = "PSU"
    # c1, c2 = 0.6271, 1.3963  # constants for GWC
    # c1, c2 = 0.5478, 1.9195  # constants for BOU
    c1, c2 = 0.5861, 1.6461  # constants for PSU
    df = shakespeare_comparison(s, 2012)
    df["site"] = s
    df = add_solar_time(df)
    df = df.set_index("solar_time")
    df["y"] = c1 + c2 * np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["lw_pred"] = df.y * SIGMA * np.power(df.t_a, 4)

    df = df[["cs_period", "dw_ir", "t_a", "lw_pred"]].copy()
    df["lw_err"] = df.lw_pred - df.dw_ir
    pdf = df.resample("D")["cs_period", "lw_err"].mean()

    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.7)
    ax.scatter(pdf.cs_period, pdf.lw_err, marker=".")
    ax.set_xlim(0, 1)
    ax.set_xlabel("daily clear sky fraction")
    ax.set_ylabel("mean daily LW error")
    title = f"{s} 2012 (ndays={pdf.shape[0]}) [{c1}, {c2}]"
    ax.set_title(title, loc="left")
    filename = os.path.join("figures", f"{s}_lw_err_per_day.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def three_step_fit(df):
    # DEPRECATED
    # run a linear regression, then altitude correction, then final regression
    # df must have columns: x, y, elev
    train_x = df.x.to_numpy().reshape(-1, 1)
    train_y = df.y.to_numpy().reshape(-1, 1)
    model = LinearRegression(fit_intercept=True)
    model.fit(train_x, train_y)
    c1 = model.intercept_.round(4)[0]
    c2 = model.coef_[0].round(4)[0]
    pred_y = c1 + (c2 * train_x)

    model2 = LinearRegression(fit_intercept=False)
    elev = df.elev.to_numpy().reshape(-1, 1)
    train_x = (P_ATM / 100000) * (np.exp(-1 * elev / 8500) - 1)
    train_y = pred_y - train_y  # model - actual
    model2.fit(train_x, train_y)
    c3 = model2.coef_[0].round(4)[0]

    # now linear fit one more time
    de_p = c3 * (P_ATM / 100000) * (np.exp(-1 * elev / 8500) - 1)
    train_x = df.x.to_numpy().reshape(-1, 1)
    train_y = df.y.to_numpy().reshape(-1, 1) + de_p
    model.fit(train_x, train_y)
    c1 = model.intercept_.round(4)[0]
    c2 = model.coef_[0].round(4)[0]

    return c1, c2, c3


def plot_uncertainty():
    # visualize impact of +/- 5 W/m^2 error in LW measurement
    x = np.linspace(250, 310, 9)  # Ta
    y = np.linspace(260, 400, 9)  # LW
    y = np.flip(y)  # smallest value at bottom
    z = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            base_e = y[i] / (SIGMA * np.power(x[j], 4))
            if base_e > 1:
                z[i, j] = 0
            else:
                z[i, j] = (0.05 * y[i]) / (SIGMA * np.power(x[j], 4))

    fig, ax = plt.subplots()
    cb = ax.imshow(z, vmin=0, vmax=0.05)
    fig.colorbar(cb, label=r"$\pm \varepsilon$ [-]")
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticklabels(x, rotation=45)
    ax.set_yticklabels(y)
    ax.set_xlabel("T$_a$ [K]")
    ax.set_ylabel("LW [W/m$^2$]")
    filename = os.path.join("figures", "uncertainty.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


# 5/15/2023
def compare_clearsky_id_3min():
    # built in main of process.py
    # pre-2009, SURFRAD sites seem to be on 3-min frequency
    keep_cols = ["zen", "GHI_m", "DNI_m", "GHI_c", "DNI_c", "cs_period"]
    # df8 = import_site_year("BON", "2008", drive="server4")  # 2008
    # df8 = df8[keep_cols]
    df9 = import_site_year("BON", "2009", drive="server4")  # 2009
    df9 = df9[keep_cols]
    df9 = df9.rename(columns={"cs_period": "cs_orig"})

    # df = df9.resample('3T', label="right").mean()
    start_time = time.time()
    df = df9.groupby(df9.index.floor("3T")).mean()
    df = find_clearsky(df, window=10, min_sample=2)
    df["cs_min2"] = df.cs_period.astype(int)
    print(time.time() - start_time)

    df = find_clearsky(df, window=30, min_sample=3)
    df["cs_w30"] = df.cs_period.astype(int)
    print(time.time() - start_time)

    df[["cs_orig", "cs_min2", "cs_w30"]].describe()
    df[["cs_orig", "cs_min2", "cs_w30"]].corr()

    df9 = find_clearsky(df9, window=10, min_sample=2)
    df9["cs_min2"] = df9.cs_period.astype(int)
    df9["cs_orig"] = df9.cs_orig.astype(int)
    df9[["cs_orig", "cs_min2"]].describe()
    print(time.time() - start_time)
    return None


def plot_clear_site():
    # plot daily values of pct_clr and npts_clr per site
    # from corr26b
    s = "GWC"
    # c1, c2 = 0.5861, 1.6461  # constants for PSU
    df = create_training_set(
        year=[2012, 2013, 2014], sites=s, filter_pct_clr=0.0,
        filter_npts_clr=0.0,
        temperature=False, cs_only=False, drive="server4")
    c1, c2 = fit_linear(df.loc[df.csv2])
    df["y"] = c1 + c2 * np.sqrt(df.pw_hpa * 100 / P_ATM)
    df["lw_pred"] = df.y * SIGMA * np.power(df.t_a, 4)

    df["lw_err"] = df.lw_pred - df.dw_ir
    df["csv2"] = df.csv2.astype("bool")
    pdf = df[["csv2", "lw_err"]].resample("D").mean()
    x = df.resample("D")["csv2"].mean()
    y = df.resample("D")["csv2"].count()

    tmp_clr = df["csv2"].resample("D").count()
    thresh = np.quantile(
        tmp_clr.loc[tmp_clr > 0].to_numpy(), 0.2
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.grid(True, alpha=0.7)
    # ax.scatter(pdf.csv2, pdf.lw_err, marker=".")
    ax.scatter(x, y, marker=".", alpha=0.5)
    ax.axvline(0.05, c="k")
    ax.axhline(thresh, c="k")
    title = f"{s} (ndays={pdf.shape[0]})"
    ax.set_title(title, loc="left")
    ax.set_xlim(0, 1)
    ax.set_xlabel("daily clear sky fraction")
    ax.set_ylabel("daily clear sky samples")
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    filename = os.path.join("figures", f"clear_{s}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_surface_c1c2(elev, azim):
    # run in corr26b
    df = create_training_set(
        year=[2010, 2011, 2012, 2013], filter_pct_clr=0.05,
        filter_npts_clr=0.2, temperature=True, cs_only=True, drive="server4"
    )
    test = df.loc[df.index.year == 2012].copy()  # make test set

    c1_x = np.linspace(0, 1, 100)
    c2_x = np.linspace(0, 4, 200)

    # Zoom in
    # c1_x = np.linspace(0.5, 0.65, 80)
    # c2_x = np.linspace(1.7, 2, 50)

    z = np.zeros((len(c1_x), len(c2_x)))
    for i in range(len(c1_x)):
        for j in range(len(c2_x)):
            pred_y = c1_x[i] + c2_x[j] * test.x
            z[i, j] = np.sqrt(mean_squared_error(test.y, pred_y))

    X, Y = np.meshgrid(c2_x, c1_x)

    # elev = 30
    # azim = 40
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=elev, azim=azim)
    cb = ax.plot_surface(
        X, Y, z, cmap=mpl.cm.coolwarm,
        norm=mpl.colors.LogNorm(),
        linewidth=0, antialiased=True
    )
    ax.contourf(
        c2_x, c1_x, z, zdir='z', offset=0, cmap=mpl.cm.coolwarm, alpha=0.7,
        norm=mpl.colors.LogNorm()
    )
    fig.colorbar(cb, shrink=0.6, label="rmse")
    ax.set_ylabel(f"c1 [{c1_x[0]}, {c1_x[-1]}]")
    ax.set_xlabel(f"c2 [{c2_x[0]}, {c2_x[-1]}]")
    ax.set_zlabel("rmse")
    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
    ax.xaxis.set_major_formatter('{x:.02f}')
    ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(5))
    ax.yaxis.set_major_formatter('{x:.02f}')
    title = f"view note: elev={elev}, azim={azim}"
    ax.set_title(title, loc="left")
    plt.tight_layout()
    plt.show()
    return None


def rmse_v_c1_sites_3d():
    df = create_training_set(
        year=[2011, 2012, 2013], filter_pct_clr=0.05,
        filter_npts_clr=0.2, temperature=True, cs_only=True, drive="server4")

    yticks = np.flip(np.arange(7))  # ticks for y-axis
    c1_x = np.linspace(0.4, 0.8, 50)
    c2 = np.zeros((len(ELEVATIONS), len(c1_x)))
    rmse = np.zeros((len(ELEVATIONS), len(c1_x)))
    i = 0
    for site in ELEVATIONS:
        tmp = df.loc[df.site == site[0]]
        train = tmp.loc[tmp.index.year != 2013]
        test = tmp.loc[tmp.index.year == 2013]
        train_x = train.x.to_numpy().reshape(-1, 1)
        train_y = train.y.to_numpy().reshape(-1, 1)
        for j in range(len(c1_x)):
            model = LinearRegression(fit_intercept=False)
            model.fit(train_x, train_y - c1_x[j])
            pred_y = c1_x[j] + (model.coef_[0].round(4)[0] * test.x)
            rmse[i, j] = np.sqrt(mean_squared_error(test.y, pred_y))
            c2[i, j] = model.coef_[0].round(4)[0]
        i += 1

    elev = -90
    azim = 180
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((3, 8, 3))
    ylabels = []
    min_c1 = []
    min_c2 = []
    for i in range(len(ELEVATIONS)):
        cb = ax.scatter(
            c1_x, rmse[i, :], zs=yticks[i], zdir="y",
            c=c2[i, :], cmap=mpl.cm.Spectral, vmin=0, vmax=3,
            alpha=0.9, marker="."
        )
        ylabels.append(ELEVATIONS[i][0])
        idx = rmse[i, :].argmin()
        ax.scatter(c1_x[idx], yticks[i], 0, edgecolor="0.3", facecolor="1.0", marker=".")
        text = f"({c1_x[idx]:.02f}, {c2[i, idx]:.02f})"
        min_c1.append(c1_x[idx])
        min_c2.append(c2[i, idx])
        # ax.text(
        #     c1_x[idx] - 0.05, yticks[i] + 0.05, 0, s=text,
        #     fontsize="x-small", color="0.2"
        # )
    ax.set_xlabel(f"c1 [{c1_x[0]}, {c1_x[-1]}]")
    ax.set_zlabel('RMSE', rotation=90)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_zlim(0, 0.08)
    ax.zaxis.set_major_locator(mpl.ticker.LinearLocator(5))
    ax.zaxis.set_major_formatter('{x:.02f}')
    title = f"view note: elev={elev}, azim={azim}"
    ax.set_title(title, loc="left")
    # fig.colorbar(cb, pad=0.15, shrink=0.6, label="c2", extend="both")
    plt.show()

    # ELEVATION: look at minimized c1 vs station elevation
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(alpha=0.3)
    xx = np.zeros(len(min_c1))
    yy = np.zeros(len(min_c1))
    for i in range(len(min_c1)):
        elev = ELEVATIONS[i][1]
        xx[i] = elev
        yy[i] = min_c1[i]
        # ax.scatter(elev, min_c1[i], c="0.3")
        ax.text(elev + 5, min_c1[i] + 0.001, ELEVATIONS[i][0])
    ax.scatter(xx, yy, c="0.3")
    x = np.linspace(0, 1670, 20)
    y = 0.2 * (np.exp(-1 * x / 8500) - 1)
    ax.plot(x, y + 0.63)
    ax.axhline(0.6, ls="--")
    ax.set_xlabel("Station elevation (m)")
    ax.set_ylabel("best fit c1")
    ax.set_xlim(left=0)
    ax.set_axisbelow(True)
    plt.show()

    y2 = yy - yy.max()  # normalize
    model = LinearRegression(fit_intercept=False)
    model.fit(xx.reshape(-1, 1), y2.reshape(-1, 1))
    c2 = model.coef_[0].round(4)  # slope of best fit

    return None


def rmse_v_c1_single_site():
    # run in corr26b
    df = create_training_set(
        year=[2011, 2012, 2013], filter_pct_clr=0.05,
        filter_npts_clr=0.2, temperature=True, cs_only=True, drive="server4")
    c1_x = np.linspace(0.4, 0.8, 50)
    site = "GWC"
    tmp = df.loc[df.site == site]
    train = tmp.loc[tmp.index.year != 2013]
    test = tmp.loc[tmp.index.year == 2013]
    train_x = train.x.to_numpy().reshape(-1, 1)
    train_y = train.y.to_numpy().reshape(-1, 1)
    rmse = np.zeros(len(c1_x))
    c2 = np.zeros(len(c1_x))
    for j in range(len(c1_x)):
        model = LinearRegression(fit_intercept=False)
        model.fit(train_x, train_y - c1_x[j])
        pred_y = c1_x[j] + (model.coef_[0].round(4)[0] * test.x)
        rmse[j] = np.sqrt(mean_squared_error(test.y, pred_y))
        c2[j] = model.coef_[0].round(4)[0]

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5)
    cb = ax.scatter(c1_x, rmse, c=c2, vmin=0, vmax=3, cmap=mpl.cm.Spectral)
    fig.colorbar(cb, label=f"c2", extend="both")
    ax.set_ylabel("RMSE")
    ax.set_xlabel(f"c1 [{c1_x[0]}, {c1_x[-1]}]")
    ax.set_axisbelow(True)
    ax.set_xlim(c1_x[0], c1_x[-1])
    idx = rmse.argmin()
    text = f"minimum RMSE @ ({c1_x[idx]:.04f}, {c2[idx]})"
    ax.text(
        0.95, 0.95, text, transform=ax.transAxes, ha="right", va="top",
        backgroundcolor="1.0")
    title = f"{site} (train={train.shape[0]:,} pts, test={test.shape[0]:,} pts)"
    ax.set_title(title, loc="left")
    plt.tight_layout()
    plt.show()
    return None


def rmse_contour_c1_vs_c2():
    # contourf plots, fixed c1, c2, c3. RMSE evaluated collectively.
    df = create_training_set(
        year=[2010, 2011, 2012, 2013], filter_pct_clr=0.05,
        filter_npts_clr=0.2, temperature=True, cs_only=True, drive="server4"
    )
    df['correction'] = np.exp(-1 * df.elev / 8500) - 1
    test = df.copy()
    # train = df.loc[df.index.year != 2012].copy()
    # test = df.loc[df.index.year == 2012].copy()  # make test set

    c1_x = np.linspace(0.3, 0.8, 25)  # 100
    c2_x = np.linspace(1, 3, 50)  # 200
    c3 = 0.5

    z = np.zeros((len(c1_x), len(c2_x)))
    for i in range(len(c1_x)):
        for j in range(len(c2_x)):
            pred_y = c1_x[i] + c2_x[j] * test.x
            correction = c3 * test.correction
            z[i, j] = np.sqrt(mean_squared_error(test.y, pred_y + correction))

    # cnorm = mpl.colors.LogNorm(vmin=0.01, vmax=1)
    xi, yi = np.unravel_index(z.argmin(), z.shape)
    cnorm = mpl.colors.Normalize(vmin=0, vmax=0.4)
    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    cb = ax.contourf(
        c2_x, c1_x, z, cmap=mpl.cm.coolwarm, norm=cnorm
    )
    ax.scatter(c2_x[yi], c1_x[xi], c="k", marker="^")
    text = f"({c1_x[xi]:.4f}, {c2_x[yi]:.4f}, {c3})"
    ax.text(c2_x[yi] + .05, c1_x[xi] + 0.01, text)
    fig.colorbar(cb, label="RMSE")
    ax.set_ylabel(f"c1 [{c1_x[0]}, {c1_x[-1]}]")
    ax.set_xlabel(f"c2 [{c2_x[0]}, {c2_x[-1]}]")
    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
    ax.xaxis.set_major_formatter('{x:.02f}')
    ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(6))
    ax.yaxis.set_major_formatter('{x:.02f}')
    title = f"c3={c3} (RMSE: min={z.min():.3f}, avg={z.mean():.3f})"
    ax.set_title(title, loc="left")
    plt.tight_layout()
    plt.show()
    return None


def fitted_c2_vs_c1_fixed_c3():
    df = create_training_set(
        year=[2010, 2011, 2012, 2013], filter_pct_clr=0.05,
        filter_npts_clr=0.2, temperature=False, cs_only=True, drive="server4")
    df['correction'] = np.exp(-1 * df.elev / 8500) - 1

    c3 = 0.5
    df["yc"] = df.y - (c3 * df.correction)
    yticks = np.flip(np.arange(7))  # ticks for y-axis
    c1_x = np.linspace(0.4, 0.8, 50)
    c2 = np.zeros((len(ELEVATIONS), len(c1_x)))
    rmse = np.zeros((len(ELEVATIONS), len(c1_x)))
    i = 0
    for site in ELEVATIONS:
        tmp = df.loc[df.site == site[0]]
        train = tmp.loc[tmp.index.year != 2013]
        test = tmp.loc[tmp.index.year == 2013]
        train_x = train.x.to_numpy().reshape(-1, 1)
        train_y = train.yc.to_numpy().reshape(-1, 1)
        for j in range(len(c1_x)):
            model = LinearRegression(fit_intercept=False)
            model.fit(train_x, train_y - c1_x[j])
            pred_y = c1_x[j] + (model.coef_[0].round(4)[0] * test.x)
            rmse[i, j] = np.sqrt(mean_squared_error(test.yc, pred_y))
            c2[i, j] = model.coef_[0].round(4)[0]
        i += 1

    # fig, ax = plt.subplots(figsize=(8, 3))
    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    for i in range(len(ELEVATIONS)):
        site = ELEVATIONS[i][0]
        y_pad = rmse[i, :] * 5
        ax.fill_between(
            c1_x, c2[i, :] - y_pad, c2[i, :] + y_pad,
            fc=COLOR7_DICT[site], alpha=0.2
        )
        ax.plot(c1_x, c2[i, :], label=site, c=COLOR7_DICT[site])
    ax.legend()
    ax.set_xlabel("given c1")
    ax.set_ylabel("fitted c2")
    # ax.set_xlim(0.5, 0.7)
    # ax.set_ylim(0, 3)
    ax.set_xlim(0.4, 0.8)
    ax.set_ylim(-2, 6)
    ax.set_title(f"c3={c3}", loc="left")
    ax.set_axisbelow(True)
    plt.show()
    return None


def cs_count_per_site():
    s = "SXF"
    out = []
    for yr in np.arange(2003, 2023):
        df = import_site_year(s, yr, drive="server4")
        entry = dict(
            year=yr,
            cs_base=df.cs_period.sum(),
            cs_reno=df.reno_cs.sum(),
            cs_union=df[["cs_period", "reno_cs"]].all(axis=1).sum(),
            cs_intersect=df[["cs_period", "reno_cs"]].any(axis=1).sum(),
            total=df.shape[0]
        )
        out.append(entry)
    df = pd.DataFrame(out)
    filename = os.path.join("data", "cs_count", f"{s}.csv")
    df.to_csv(filename)
    return None


def plot_cs_count_per_site():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(alpha=0.3)
    total = np.zeros(len(np.arange(2003, 2023)))
    cs_sum = np.zeros(len(total))
    for s in SURF_SITE_CODES:
        filename = os.path.join("data", "cs_count", f"{s}.csv")
        df = pd.read_csv(filename, index_col=0)
        ax.step(
            df.year, df.cs_union / df.total, where="post", c=COLOR7_DICT[s],
            label=s,
        )
        total += df.loc[df.year >= 2003, "total"].to_numpy()
        cs_sum += df.loc[df.year >= 2003, "cs_union"].to_numpy()
    ax.plot(np.arange(2003, 2023), cs_sum / total, ".--", c="0.3",
            label="total")

    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 0.8)
    ax.set_xlim(2000, 2023)
    ax.set_ylabel("Fraction of samples labelled clear per year")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
    return None


if __name__ == "__main__":
    print()