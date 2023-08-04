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
from statsmodels.tsa.seasonal import seasonal_decompose

from constants import *
from fig3 import get_atm_p, import_ijhmt_df, ijhmt_to_tau, ijhmt_to_individual_e
from process import *
from enso import get_train, import_oni_mei_data, create_monthly_df
from figures import FILTER_NPTS_CLR, FILTER_PCT_CLR, training_data, COLORS, \
    C1_CONST, C2_CONST, pw2rh, rh2pw


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


def plot_enso_detrend():
    mei_color = "#C54459"  # muted red
    oni_color = "#4C6BE6"  # powder blue
    mint = "#6CB289"
    gold = "#E0A500"
    oni, mei = import_oni_mei_data()

    df = get_train()  # retrieve and concatenate create_train csv files
    df = create_monthly_df(df, keep_samples=1000, c1_const=0.6, c3_const=0.1)

    df["mei"] = mei.value.groupby(mei.index.year).median().to_numpy()
    df["oni"] = oni.value.groupby(oni.index.year).median().to_numpy()
    df = pd.merge_asof(
        df, mei["value"], left_index=True, right_index=True, direction="nearest"
    )
    df = pd.merge_asof(
        df, oni["value"], left_index=True, right_index=True, direction="nearest"
    )
    print(df.corr())

    analysis = df[['c1']].copy()
    analysis.dropna(inplace=True)
    decompose_result_mult = seasonal_decompose(analysis, model="additive", period=12)
    # decompose_result_mult.plot()
    # plt.show()
    x1 = decompose_result_mult.trend

    analysis = df[['c2']].copy()
    analysis.dropna(inplace=True)
    decompose_result_mult = seasonal_decompose(analysis, model="additive", period=12)
    # decompose_result_mult.plot()
    # plt.show()
    x2 = decompose_result_mult.trend

    analysis = df[['c3']].copy()
    analysis.dropna(inplace=True)
    decompose_result_mult = seasonal_decompose(analysis, model="additive", period=12)
    # decompose_result_mult.plot()
    # plt.show()
    x3 = decompose_result_mult.trend

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.grid(alpha=0.2)
    ax.axhline(0, ls="--", c="0.3", alpha=0.3)
    ax.axhline(0.6, ls=":", c="0.8", alpha=0.3)
    ax.plot(oni.index, oni.value, c=oni_color, alpha=0.4)
    ax.plot(mei.index, mei.value, c=mei_color, alpha=0.4)
    ax.fill_between(
        mei.index, -1, 1, where=abs(mei.value) > 0.5,
        fc="0.8", alpha=0.5,
        transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    )
    # ax.plot(x1 - x1.mean(), c="k")
    ax.plot(x2 - x2.mean(), c=gold, label="c2")
    ax.plot(x3 - x3.mean(), c="0.5", label="c3")
    ax.set_ylabel(r"$\leftarrow$ La Niña$\quad\quad$ El Niño $\rightarrow$")
    ax.set_xlim(oni.index[0], oni.index[-1])
    ax.legend()
    plt.show()
    filename = os.path.join("figures", f"enso_detrend_v2_2.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_c2_intime():
    # run in enso
    mei_color = "#C54459"  # muted red
    gold = "#E0A500"
    df = get_train()
    df = create_monthly_df(df, keep_samples=1000, c1_const=0.6, c3_const=0.1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.grid(True, alpha=0.3)
    ax.plot(df.index, df.c2, color=gold, label="monthly")
    pdf = df.rolling(6, center=True).mean()
    ax.plot(pdf.index, pdf.c2, label="6mo avg")
    pdf = df.rolling(12, center=True).mean()
    ax.plot(pdf.index, pdf.c2, c=mei_color, label="12mo avg")
    ax.set_axisbelow(True)
    ax.set_xlim(df.index[0], df.index[-1])
    ax.xaxis.set_major_locator(mpl.dates.YearLocator())
    ax.legend()
    ax.set_title("Fitted c2 values (all sites) for c1=0.6, c3=0.1", loc="left")
    plt.tight_layout()
    plt.show()
    return None


def plot_enso_plus_single_var():
    mei_color = "#C54459"  # muted red
    oni_color = "#4C6BE6"  # powder blue
    mint = "#6CB289"
    gold = "#E0A500"

    df = get_train()  # retrieve and concatenate create_train csv files
    df = create_monthly_df(df, keep_samples=1000, c1_const=0.6, c3_const=0.1)

    oni, mei = import_oni_mei_data()
    # mei = mei.loc[mei.index.year >= 2006]
    # oni = oni.loc[oni.index.year >= 2006]

    # month_window = 3
    # df = df.rolling(month_window, center=False).sum()

    grid_alpha = 0.5
    # fig, ax = plt.subplots(figsize=(12, 4))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[1, 2])
    plt.subplots_adjust(hspace=0.1)
    # fig, ax = plt.subplots(figsize=(16, 4))
    ax = axes[0]
    # ax.step(df.index, df.rmse)
    ax.step(df.index, df.avg_rh)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=grid_alpha)
    ax.set_axisbelow(True)
    # ax.set_ylabel("RMSE")
    ax.set_ylabel("RH")

    # ax = axes[1]
    # ax.step(df.index, df.n_pts)
    # ax.set_ylim(bottom=0)
    # ax.grid(alpha=grid_alpha)
    # ax.set_axisbelow(True)
    # ax.set_ylabel("sample size")

    ax = axes[-1]
    ax.grid(alpha=grid_alpha)
    ax.axhline(0, ls="--", c="0.3", alpha=0.3)
    # ax.axhline(0.6, ls=":", c="0.8", alpha=0.3)
    ax.plot(oni.index, oni.value, c=oni_color, alpha=0.4)
    ax.plot(mei.index, mei.value, c=mei_color, alpha=0.4)

    # ax.step(df.index, df.c1 / month_window, label="c1", c=mint, where="post")
    ax.step(df.index, df.c2, label="c2", c=gold, where="post")
    # ax.step(df.index, df.c3 / month_window, label="c3", c="0.5", where="post")

    ax.set_xlim(df.index[0], df.index[-1])
    # ymin, ymax = ax.get_ylim()
    # ylim = abs(ymin) if abs(ymin) > ymax else ymax
    # ax.set_ylim(-1 * ylim, ylim)  # ensure symmetric around y=0
    ax.set_ylim(-2, 2)
    ax.set_ylabel(r"$\leftarrow$ La Niña$\quad\quad$ El Niño $\rightarrow$")
    # ax.legend()
    ax.xaxis.set_major_locator(mpl.dates.YearLocator())
    ax.set_axisbelow(True)
    # ax.set_title(f"Month rolling window = {month_window}", loc="left")
    plt.show()

    # filename = os.path.join("figures", f"enso_m{month_window}_2.png")
    # fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def data_12yr_e_vs_pw_site():
    # e vs pw and dlw vs rh
    site = "GWC"
    ms = 10  # marker size
    df_ref = create_training_set(
        year=[2010 + i for i in range(12)], sites=[site],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR,
        drive="server4"
    )

    cnorm = mpl.colors.Normalize(vmin=280, vmax=310)
    cmap = mpl.cm.coolwarm
    cnorm2 = mpl.colors.Normalize(vmin=1, vmax=12)
    cmap2 = mpl.cm.twilight

    # make both plots inside the for loop
    fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex=True, sharey=True)
    fig2, axes2 = plt.subplots(4, 3, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(12):
        df = df_ref.loc[df_ref.index.year == 2010 + i]
        df = df.sample(n=1000)
        ax = axes[0 + i//3, 0 + i % 3]
        ax2 = axes2[0 + i // 3, 0 + i % 3]
        ax.grid(alpha=0.3)
        cb = ax.scatter(
            df.pw_hpa, df.y, c=df.t_a, cmap=cmap, norm=cnorm,
            s=ms, alpha=0.5
        )
        ax.set_title(f"{2010 + i}", loc="left", fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel("emissivity [-]")
            ax2.set_ylabel("DLW [W/m$^2$]")
        if i // 3 == 3:
            ax.set_xlabel("p$_w$ [hPa]")
            ax2.set_xlabel("RH [%]")

        ax2.grid(alpha=0.3)
        cb2 = ax2.scatter(
            df.rh, df.dw_ir, c=df.index.month, cmap=cmap2, norm=cnorm2,
            s=ms, alpha=0.5
        )
        ax2.set_title(f"{2010 + i}", loc="left", fontsize=10)
    ax.set_ylim(0.5, 1)
    ax.set_xlim(0, 25)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(cb, cax=cbar_ax, extend="both", label="T [K]")
    cbar.solids.set(alpha=1)
    filename = os.path.join("figures", f"data_12yr_e_vs_pw_{site}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    ax2.set_ylim(100, 550)
    ax2.set_xlim(0, 100)
    fig2.subplots_adjust(right=0.85)
    cbar_ax = fig2.add_axes([0.91, 0.1, 0.03, 0.8])
    cbar = fig2.colorbar(cb2, cax=cbar_ax, label="month")
    cbar.solids.set(alpha=1)
    filename = os.path.join("figures", f"data_12yr_dlw_vs_rh_{site}.png")
    fig2.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def one_day():
    # EXPLORE SINGLE DAY
    site = "SXF"
    plot_date = dt.date(2021, 2, 28)  # date to plot

    df = create_training_set(
        year=[plot_date.year], sites=[site],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR, drive="server4"
    )

    # ff = df.loc[(df.pw_hpa < 5) & (df.y > 0.8)]
    pdf = df.loc[df.index.date == plot_date].copy()
    # FIGURE
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), height_ratios=[2, 1])
    ax = axes[0]
    ax.scatter(pdf.index, pdf.t_a, marker=".", alpha=0.5, color="blue")
    ax2 = ax.twinx()
    ax2.grid(True)
    ax2.scatter(pdf.index, pdf.y, marker=".", c="0.5")
    date_str = plot_date.strftime("%m-%d-%Y")
    title = f"{site} {date_str}"
    ax.set_title(title, loc="left")
    ax.set_ylabel("T [K]", fontdict=dict(color="blue"))
    ax2.set_ylabel("emissivity", fontdict=dict(color="0.5"))

    ax = axes[1]
    ax.plot(pdf.index, pdf.GHI_m, label="GHI")
    ax.plot(pdf.index, pdf.DNI_m, label="DNI")
    ax.plot(pdf.index, pdf.dw_ir, label="LW")
    ax.legend(ncol=3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    return None


def dra_multiplexer():
    # compare DRA pre- and post- April 11 2023
    site = "DRA"
    ms = 10  # marker size
    df_ref = create_training_set(
        year=[2023], sites=[site],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR,
        drive="server4"
    )

    date = dt.date(2023, 4, 11)
    pre = df_ref.loc[df_ref.index.date < date]  # comparison not working
    post = df_ref.loc[df_ref.index.date > date]

    pre = pre.sample(1000, random_state=12)
    post = post.sample(1000, random_state=12)

    cnorm = mpl.colors.Normalize(vmin=280, vmax=310)
    cmap = mpl.cm.coolwarm

    fig, ax = plt.subplots(figsize=(5, 5), sharex=True, sharey=True)
    ax.grid(alpha=0.3)
    ax.scatter(pre.pw_hpa, pre.y, s=ms, alpha=0.5, label="pre-")
    ax.scatter(post.pw_hpa, post.y, s=ms, alpha=0.5, label="post-")
    ax.set_title("2023 clear sky samples pre- and post- April 11")
    ax.legend()
    ax.set_ylim(0.5, 1)
    ax.set_xlim(0, 15)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    ax.set_axisbelow(True)
    plt.show()
    return None


def epri_presentation():
    df = training_data()
    df = reduce_to_equal_pts_per_site(df, min_pts=100)
    df['y'] = df.y - df.correction  # bring all sample to sea level
    ms = 15  # marker size for data samples

    t = 288  # standard temperature for scaling measurement error
    yerr = 5 / (SIGMA * np.power(t, 4))  # +/-5 W/m^2 error
    figsize = (10, 4)
    # set axis bounds of both figures
    xmin, xmax = (0.2, 35)
    ymin, ymax = (0.5, 1.0)

    # define fitted correlation
    x = np.geomspace(xmin+0.00001, xmax, 40)  # hPa
    y = 0.6 + 1.653 * np.sqrt(x * 100 / P_ATM)  # emissivity

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.grid(alpha=0.3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(
        df.pw_hpa, df.y, marker="o", s=ms,
        alpha=0.3, c="0.3", ec="0.5", lw=0.5, zorder=0
    )
    ax.plot(x, y, lw=1.5, ls="--", c="0.0", zorder=2,
            label="$c_1+c_2\sqrt{p_w}$")
    ax.legend(ncol=1, loc="lower right", fontsize=14)
    ax.set_axisbelow(True)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    plt.show()
    filename = os.path.join("figures", f"example.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    # ---- polished figure of e vs pw with and without 2020 DRA

    # df = training_data()  # import data
    df = create_training_set(
        year=[2010, 2015, 2020],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR, drive="server4"
    )
    df['correction'] = 0.15 * (np.exp(-1 * df.elev / 8500) - 1)
    df = reduce_to_equal_pts_per_site(df, min_pts=200)
    df['y'] = df.y - df.correction

    # --- evaluate error before and after accounting for 2020 data
    df20 = df.copy()
    # get main training set for comparison
    dfx = training_data()
    dfx = reduce_to_equal_pts_per_site(dfx, min_pts=200)
    dfx['y'] = dfx.y - dfx.correction  # bring all sample to sea level
    # evaluate comparative LW error
    emissivity = 0.619 + 1.512 * np.sqrt(df20.pw_hpa * 100 / P_ATM)
    act = df20.dw_ir.to_numpy().reshape(-1, 1)
    lw = emissivity * np.power(df20.t_a, 4) * SIGMA
    rmse = np.sqrt(mean_squared_error(act, lw.to_numpy().reshape(-1, 1)))
    print("w/2020 RMSE: ", rmse)
    emissivity = 0.604 + 1.605 * np.sqrt(dfx.pw_hpa * 100 / P_ATM)
    act = dfx.dw_ir.to_numpy().reshape(-1, 1)
    lw = emissivity * np.power(dfx.t_a, 4) * SIGMA
    rmse = np.sqrt(mean_squared_error(act, lw.to_numpy().reshape(-1, 1)))
    print("w/o 2020 RMSE: ", rmse)
    # ---

    ms = 15

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.grid(True, alpha=0.3)
    i = 0
    for site in ELEVATIONS:  # plot in sorted order
        s = site[0]
        group = df.loc[df.site == s]
        if s == "SXF":
            # for 2010-2015-2020 plot (remove 2/3/2010 and 11/23/2010 for SXF)
            group = group.loc[group.index.date != dt.date(2010, 2, 3)]
            group = group.loc[group.index.date != dt.date(2010, 11, 23)]
        ax.scatter(
            group.pw_hpa, group.y, marker="o", s=ms,
            alpha=0.8, c=SEVEN_COLORS[i], ec="0.5", lw=0.5, zorder=10
        )
        ax.scatter([], [], marker="o", s=3*ms, alpha=1, c=SEVEN_COLORS[i],
                   ec="0.5", lw=0.5,  label=s)  # dummy for legend
        i += 1
    xmin, xmax = (0, 35)
    x = np.geomspace(0.00001, xmax, 40)
    y = 0.6 + 1.653 * np.sqrt(x * 100 / P_ATM)
    label = r"$c_1 + c_2 \sqrt{p_w}$"
    ax.plot(x, y, c="0.3", lw=1.5, ls="--", label=label, zorder=10)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    ax.legend(ncol=4, bbox_to_anchor=(0.99, 0.05), loc="lower right", fontsize=8)
    plt.tight_layout()
    filename = os.path.join("figures", f"e_response_2020.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)

    # ---- DRA plot
    site = "DRA"
    ms = 10  # marker size
    df_ref = create_training_set(
        year=[2023], sites=[site],
        temperature=False, cs_only=True,
        filter_pct_clr=FILTER_PCT_CLR,
        filter_npts_clr=FILTER_NPTS_CLR,
        drive="server4"
    )

    date = dt.date(2023, 4, 11)
    pre = df_ref.loc[df_ref.index.date < date]
    post = df_ref.loc[df_ref.index.date > date]

    pre = pre.sample(800, random_state=12)
    post = post.sample(800, random_state=12)

    fig, ax = plt.subplots(figsize=(5, 4), sharex=True, sharey=True)
    ax.grid(alpha=0.3)
    ax.scatter(pre.pw_hpa, pre.y, s=ms, c="0.3", alpha=0.3,
               label="pre- April 11")
    ax.scatter(post.pw_hpa, post.y, s=ms, c=COLORS["cornflowerblue"],
               alpha=0.5, label="post- April 11")
    ax.set_title("2023 clear sky samples", loc="left")
    ax.legend()
    ax.set_ylim(0.5, 1)
    ax.set_xlim(0, 12)
    ax.set_xlabel("p$_w$ [hPa]")
    ax.set_ylabel("emissivity [-]")
    ax.set_axisbelow(True)
    plt.tight_layout()
    filename = os.path.join("figures", "dra_multiplexer.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def from_fig3_compare_linspace_geomspace():
    xmin, xmax = (0.1, 30)  # hPa
    x = np.geomspace(xmin+0.00001, xmax, 40) * 100 / P_ATM  # normalized
    b2 = 0.1170 + 0.0662 * np.tanh(270.4686 * x)
    b3 = 0.1457 + 0.0417 * np.power(x, 0.0992)
    b4 = 0.1057 + 5.8689 * np.power(x, 0.9633)
    b157 = 0.1725 + 0.0766 + 0.0019 + 0.0026
    y = b2 + b3 + b4 + b157

    y_fit = 0.6173 + 1.681 * np.sqrt(x)
    y_fit_geom = 0.6192 + 1.6651 * np.sqrt(x)

    fig, ax = plt.subplots()
    ax.grid(axis="y")
    ax.plot(x * 100, y, c="k", ls=":")
    ax.plot(x * 100, y_fit, c="g", alpha=0.5)
    ax.plot(x * 100, y_fit_geom, c="r", alpha=0.4)
    ax2 = ax.twinx()
    ax2.axhline(0, c="0.7")
    ax2.plot(x * 100, y_fit - y, c="g", ls="--")
    ax2.plot(x * 100, y_fit_geom - y, c="r", ls="--")
    plt.show()
    return None


def fig5_e_tau():
    # create figure 5 type plot for emissivity and optical depth (tau)

    # import data
    plot_tau = True  # True plot tau, False plot emissivity
    if plot_tau:
        filename = os.path.join('data', 'tab2_tau.csv')
        ylabel = "optical depth [-]"
        figname = "fig5_tau.png"
    else:
        filename = os.path.join('data', 'tab2.csv')
        ylabel = "emissivity [-]"
        figname = "fig5.png"

    df = pd.read_csv(filename, na_values=["-"])
    gases = df.columns[2:-1]

    x = np.geomspace(0.1, 25, 40) * 100  # Pa
    pw = x / P_ATM
    xticks = [0.5, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(1, 7, figsize=(10, 4), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0)
    i = 0
    for band, group in df.groupby(df.band):
        ax = axes[i]
        ax.set_xlabel("$p_w$ x 100")
        ax.set_title(band.upper(), loc="left")
        y_ref = np.zeros(len(pw))
        for gas in gases:
            c1, c2, c3 = group[gas].values
            if np.isnan(c2) and np.isnan(c3):
                if not np.isnan(c1):  # c1 is constant
                    y = c1 * np.ones(len(pw))
                    # tau_constant = -1 * np.log(1 - c1)
                    # print(band, gas, tau_constant.round(4))
                else:
                    y = np.zeros(len(pw))
            else:
                if plot_tau:
                    y = c1 + c2 * np.power(pw, c3)
                else:
                    if band == "b2":
                        y = c1 + c2 * np.tanh(c3 * pw)
                    else:
                        y = c1 + c2 * np.power(pw, c3)
            ax.plot(pw*100, y_ref + y, label=gas)  # plot emissivity
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=8)
            y_ref += y
        i += 1
    ax.legend(fontsize="8")
    ax.set_ylim(bottom=-0.05)
    ax.set_xlim(0.1, 2.49)
    axes[0].set_ylabel(ylabel)
    filename = os.path.join("figures", figname)
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def define_tau_method_1_vs_2():
    df = import_ijhmt_df("fig3_esky_i.csv")
    df = df.set_index("pw")
    gases = df.columns

    # method 1: convert to additive transmissivity then disaggregate
    # method 2: disaggregate emissivity then convert to transmissivity

    # 1
    df1 = 1 - df  # each column now aggregated transmissivities
    col1 = df1.H2O.to_numpy()
    df1 = df1.div(df1.shift(axis=1), axis=1)  # shift axis 1 then divide
    df1["H2O"] = col1

    # 2
    col1 = df.H2O.to_numpy()
    df2 = df.diff(axis=1)  # each column is individual emissivity
    df2["H2O"] = col1  # first column already individual emissivity
    df2 = 1 - df2

    pdf = df2.copy()
    title = "method 2"

    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    x = pdf.index.to_numpy()
    species = list(LI_TABLE1.keys())[:-1]
    # species = species[::-1]  # plot in reverse
    j = 0
    y_ref = np.ones(len(x))
    for i in species:
        if i == "H2O":
            y = i
        elif i == "aerosols" or i == "overlaps":
            y = f"p{i[0].upper() + i[1:]}"
        else:
            y = f"p{i}"
        ax.plot(x, pdf[y].to_numpy(), label=i, c=cmaplist[-(j + 1)])
        y_ref = y_ref * pdf[y].to_numpy()
        j += 1
    ax.plot(x, y_ref, label="total", c="0.0", ls="--")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("pw [-]")
    ax.set_ylabel("transmissivity")
    ax.set_title(title)
    ax.legend()
    plt.show()
    return None


def plot_band_tau_vs_dopt():
    df = ijhmt_to_tau("fig3_esky_i.csv")
    x = df.index.to_numpy()
    y = df["CO2"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.set_title("O$_3$ transmissivity", loc="left")
    # ax2.set_title("O$_3$ optical depth", loc="left")
    ax1.plot(x, y, label="wide-band")
    ax2.plot(x, -1 * np.log(y))
    for i in [4, 5, 6]:
        df = ijhmt_to_tau(f"fig5_esky_ij_b{i}.csv")
        yb = df["CO2"].to_numpy()
        ax1.plot(x, yb, label=f"B{i}")
        ax2.plot(x, -1 * np.log(yb))
    ax1.set_xlabel("$p_w$ [-]")
    ax2.set_xlabel("$p_w$ [-]")
    ax1.set_ylabel("transmissivity [-]")
    ax2.set_ylabel(r"$d_{\rm{opt}}$ [-]")
    ax1.grid()
    ax2.grid()
    ax1.legend()
    plt.tight_layout()
    plt.show()

    # plot optical depth for one gas per band
    fig, ax = plt.subplots()
    for i in np.arange(1, 8):
        df = ijhmt_to_tau(f"fig5_esky_ij_b{i}.csv")
        x = df.index.to_numpy()
        y = df["O3"].to_numpy()
        ax.plot(x, -1 * np.log(y), label=f"band {i}")
    ax.set_ylabel(r"$d_{\rm{opt}}$ [-]")
    ax.set_xlabel("$p_w$ [-]")
    ax.legend()
    plt.show()
    return None


def plot_lbl_match():  # TODO update ijhmt
    # compare LC2019 with SR2021
    df = ijhmt_to_individual_e("fig3_esky_i.csv")
    x = df.index.to_numpy()
    tmp = df.copy(deep=True)
    tmp = tmp.drop(columns=["Aerosols", "total"])
    tmp["total"] = tmp.cumsum(axis=1).iloc[:, -1]

    # transmissivity - plot total tau against Shakespeare
    site = "GWC"
    lat1 = SURFRAD[site]["lat"]
    lon1 = SURFRAD[site]["lon"]
    h1, spline = shakespeare(lat1, lon1)
    pw = x * P_ATM  # Pa
    w = 0.62198 * pw / (P_ATM - pw)
    q = w / (1 + w)
    p_rep = P_ATM * np.exp(-1 * SURFRAD[site]["alt"] / 8500)
    p_ratio = p_rep / P_ATM
    he = (h1 / np.cos(40.3 * np.pi / 180)) * np.power(p_ratio, 1.8)
    d_opt = spline.ev(q, he)
    tau_shp = np.exp(-1 * d_opt)

    sr2021 = 1 - tau_shp
    y_fit = C1_CONST + C2_CONST * np.sqrt(x)

    y_lbl_orig = 0.6173 + 1.6940 * np.power(x, 0.5035)

    obs_err5 = 5 / (SIGMA * np.power(294.2, 4))
    obs_err10 = 10 / (SIGMA * np.power(294.2, 4))

    fig, ax = plt.subplots()
    ax.grid(alpha=0.3)
    ax.plot(x, y_fit, lw=2, ls="-", c="0.0", label="fit")
    ax.fill_between(x, y_fit - obs_err10, y_fit + obs_err10, fc="0.8", alpha=0.5, label="+/-10W/m$^2$")
    ax.fill_between(x, y_fit - obs_err5, y_fit + obs_err5, fc="0.6", alpha=0.5, label="+/-5W/m$^2$")
    ax.plot(x, df.total.to_numpy(), c=COLORS["persianred"], ls="-", zorder=2,
            label="LC2019")
    ax.plot(x, df.H2O + df.CO2, c=COLORS["persianred"], ls=":", lw=2,
            label="LC2019 (H2O+CO2)")
    ax.plot(x, tmp.total.to_numpy(), c=COLORS["persianindigo"], ls="--",
            label="LC2019 (no aerosols)")
    ax.plot(x, y_lbl_orig, c=COLORS["barnred"], ls="-",
            label="LBL (original)")
    ax.plot(x, sr2021, c=COLORS["cornflowerblue"], ls=":", label="SR2021")
    ax.plot(x, sr2021 + 0.02, c=COLORS["cornflowerblue"], lw=2, ls="-",
            label="SR2021 + 0.02")
    ax.legend(loc="lower right")
    ax.set_xlabel("pw")
    ax.set_ylabel("emissivity")
    ax.set_axisbelow(True)
    ax.set_xlim(x[0], x[-1])
    filename = os.path.join("figures", "lbl_match.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def broadband_contribution_e_and_tau():
    # emissivity and transmissivity with RH reference axes
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    species = list(LI_TABLE1.keys())[:-1]

    tau = ijhmt_to_tau()
    eps = ijhmt_to_individual_e()
    x = tau.index.to_numpy()

    # set margins, size of figure
    fig_x0 = 0.05
    fig_y0 = 0.35
    fig_width = 0.42
    fig_height = 0.65
    wspace = 1 - (2 * fig_width) - (2 * fig_x0)
    if wspace < 0:
        print("warning: overlapping figures")

    fig = plt.figure(figsize=(6, 3.5))
    ax0 = fig.add_axes((fig_x0, fig_y0, fig_width, fig_height))
    ax1 = fig.add_axes(
        (fig_x0 + fig_width + wspace, fig_y0, fig_width, fig_height))
    j = 0
    y_e = np.zeros(len(x))
    y_t = np.ones(len(x))
    for gas in species:
        y = eps[gas].to_numpy()
        ax0.fill_between(x, y_e, y_e + y, label=LBL_LABELS[gas], fc=cmaplist[j])
        y_e = y_e + y
        y = tau[gas].to_numpy()
        ax1.fill_between(x, y_t, y_t * y, label=LBL_LABELS[gas], fc=cmaplist[j])
        y_t = y_t * y
        j += 1
    # set axis limits, labels, grid
    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1)
    ax0.set_xlim(x[0], x[-1])
    ax1.set_xlim(x[0], x[-1])
    ax0.legend(ncol=2, loc="lower center")
    ax0.grid(alpha=0.3)
    ax0.set_axisbelow(True)
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)
    ax0.set_xlabel("p$_w$ [-]")
    ax1.set_xlabel("p$_w$ [-]")
    ax0.set_title(r"(a) $\varepsilon_{i}$", loc="left")
    ax1.set_title(r"(b) $\tau_{i}$", loc="left")

    # add secondary axes for relative humidity reference
    ax2 = fig.add_axes((fig_x0, 0.2, fig_width, 0.0))
    ax2.yaxis.set_visible(False)  # hide the yaxis
    rh_lbls = [20, 40, 60, 80, 100]
    t = 290
    ax2.set_xlim(pw2rh(x[0], t), pw2rh(x[-1], t))
    ax2.set_xticks(np.array(rh_lbls), labels=rh_lbls)
    ax2.set_xlabel(f"RH [%] at {t} K")

    ax3 = fig.add_axes((fig_x0, 0.05, fig_width, 0.0))
    ax3.yaxis.set_visible(False)  # hide the yaxis
    t = 300
    ax3.set_xlim(pw2rh(x[0], t), pw2rh(x[-1], t))
    ax3.set_xticks(np.array(rh_lbls), labels=rh_lbls)
    ax3.set_xlabel(f"RH [%] at {t} K")

    ax4 = fig.add_axes((fig_x0 + fig_width + wspace, 0.2, fig_width, 0.0))
    ax4.yaxis.set_visible(False)  # hide the yaxis
    t = 290
    ax4.set_xlim(pw2rh(x[0], t), pw2rh(x[-1], t))
    ax4.set_xticks(np.array(rh_lbls), labels=rh_lbls)
    ax4.set_xlabel(f"RH [%] at {t} K")

    ax4 = fig.add_axes((fig_x0 + fig_width + wspace, 0.05, fig_width, 0.0))
    ax4.yaxis.set_visible(False)  # hide the yaxis
    t = 300
    ax4.set_xlim(pw2rh(x[0], t), pw2rh(x[-1], t))
    ax4.set_xticks(np.array(rh_lbls), labels=rh_lbls)
    ax4.set_xlabel(f"RH [%] at {t} K")

    filename = os.path.join("figures", "broadband_contribution.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


# reduce functions in fig3.py


def plot_fig3():
    # graph fig3 from data Mengying provided
    df = import_ijhmt_df("fig3_esky_i.csv")
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(N_SPECIES)]
    fig, ax = plt.subplots()
    x = df.pw.to_numpy()
    species = list(LI_TABLE1.keys())[:-1]
    species = species[::-1]
    j = 0
    for i in species:
        if i == "H2O":
            y = i
        else:
            y = f"p{i}"
        ax.fill_between(x, 0, df[y].to_numpy(), label=i, fc=cmaplist[-(j + 1)])
        j += 1
    ax.set_ylim(bottom=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("pw")
    ax.set_ylabel(r"$\varepsilon$")
    plt.show()
    return None


def plot_fig3_ondata(s, sample):
    """Plot esky_c fits over sampled clear sky site data with altitude
    correction applied. Colormap by solar time.

    Parameters
    ----------
    s : str
        SURFRAD site code
    sample : [0.05, 0.25]
        If larger than 10%, temperature is filtered to +/- 2K.

    Returns
    -------
    None
    """
    site = import_cs_compare_csv("cs_compare_2012.csv", site=s)
    site = site.loc[site.zen < 80].copy()
    site["pp"] = site.pw_hpa * 100 / P_ATM

    site = site.sample(frac=sample, random_state=96)
    if sample > 0.1:  # apply filter to temperature
        site = site.loc[abs(site.t_a - 294.2) < 2]
        title = f"{s} 2012 {sample:.0%} sample, zen<80, +/-2K"
    else:
        title = f"{s} 2012 {sample:.0%} sample, zen<80"
    tmp = pd.DatetimeIndex(site.solar_time.copy())
    site["solar_tod"] = tmp.hour + (tmp.minute / 60) + (tmp.second / 3600)
    de_p = site.de_p.values[0]

    df = import_ijhmt_df("fig3_esky_i.csv")
    x = df.pw.to_numpy()
    df["total"] = df.pOverlaps
    df["pred_y"] = 0.6376 + (1.6026 * np.sqrt(x))
    df["best"] = 0.6376 + (1.6191 * np.sqrt(x))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    c = ax.scatter(
        site.pp, site.e_act_s, c=site.solar_tod, cmap="seismic",
        vmin=5, vmax=19, alpha=0.8,
    )
    ax.plot(x, df.total + de_p, label="LBL")
    ax.plot(x, df.pred_y + de_p, ls="--", label="(0.6376, 1.6026)")
    ax.plot(x, df.best + de_p, ls=":", label="(0.6376, 1.6191)")
    ax.legend()
    fig.colorbar(c, label="solar time")
    ax.set_ylabel("effective sky emissivity [-]")
    ax.set_xlabel("p$_w$ [-]")
    ax.set_xlim(0, 0.03)
    ax.set_ylim(0.60, 1.0)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    filename = os.path.join("figures", "fig3_ondata.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    return None


def plot_fig3_quantiles(site=None, yr=None, all_sites=False, tau=False,
                        temperature=False, shk=False, pct_clr_min=0.3,
                        pressure_bins=20, violin=False):
    # DISCLAIMER: tau may need to be adjusted since s["x"] is now defined
    #   using three_c_fit to solve for c1,c2, and c3 at once

    # plot_fig3_quantiles(
    #     yr=[2010, 2011, 2012, 2013], all_sites=True, tau=False,
    #     temperature=True, pct_clr_min=0.5, pressure_bins=5, violin=True
    # )
    # plot_fig3_quantiles(
    #     yr=[2015], tau=False,
    #     temperature=False, pct_clr_min=0.05, pressure_bins=10, violin=True
    # )

    # Create dataset for training
    s = create_training_set(
        year=yr, temperature=temperature,
        cs_only=True, filter_pct_clr=0.05, filter_npts_clr=0.2, drive="server4"
    )
    s = reduce_to_equal_pts_per_site(s)  # reduce to num pts in common

    # import ijhmt data to plot
    df = import_ijhmt_df("fig3_esky_i.csv")
    x_lbl = df.pw.to_numpy()
    df["total"] = df.pOverlaps

    if tau:
        s["transmissivity"] = 1 - s.y  # create set will create y col (e_act)
        s["y"] = -1 * np.log(s.transmissivity)
        y_lbl = -1 * np.log(1 - df.total)
        ylabel = "optical depth [-]"
        # ymin, ymax = 0.8, 3.0
    else:
        y_lbl = df.total
        ylabel = "effective sky emissivity [-]"
        # ymin, ymax = 0.60, 1.0
    labels = ["Q05-95", "Q25-75", "Q40-60"]
    quantiles = [0.05, 0.25, 0.4, 0.6, 0.75, 0.95]
    pressure_bins = pressure_bins

    # Prepare data for quantiles plot
    c1, c2, c3, xq, yq = prep_plot_data_for_quantiles_plot(
        s, pressure_bins, quantiles, violin=violin
    )
    y_fit = c1 + c2 * np.sqrt(x_lbl)

    if shk:  # if using only one site, apply shakespeare model
        pw = (x_lbl * P_ATM) / 100  # convert back to partial pressure [hPa]
        pa_hpa = s.P_rep.values[0] / 100
        w = 0.62198 * pw / (pa_hpa - pw)
        q_values = w / (1 + w)
        he = s.he.values[0]

        lat1 = SURFRAD[site]["lat"]
        lon1 = SURFRAD[site]["lon"]
        h1, spline = shakespeare(lat1, lon1)

        tau_shakespeare = []
        for q1 in q_values:
            tau_shakespeare.append(spline.ev(q1, he).item())
        # don't need to make any altitude adjustments in shakespeare
        if tau:  # optical depth
            y_sp = np.array(tau_shakespeare)
        else:  # emissivity
            y_sp = (1 - np.exp(-1 * np.array(tau_shakespeare)))

    # set title and filename
    suffix = "_tau" if tau else ""
    suffix += "_ta" if temperature else ""
    suffix += f"_clr{pct_clr_min*100:.0f}"
    suffix += f"_{pressure_bins}"
    suffix += f"_violin" if violin else ""
    title_suffix = r", T~T$_0$" if temperature else ""
    title_suffix += f" day_clr>{pct_clr_min:.0%}"
    if len(yr) == 1:
        str_name = f"{yr[0]}"
        name = yr[0]
    else:
        str_name = f"{yr[0]} - {yr[-1]}"
        name = f"{yr[0]}_{yr[-1]}"

    if all_sites:
        filename = os.path.join("figures", f"fig3{name}_q{suffix}.png")
        title = f"SURFRAD {str_name} (n={s.shape[0]:,}) ST>8" + title_suffix
    else:
        filename = os.path.join("figures", f"fig3{name}_{site}_q{suffix}.png")
        title = f"{site} {str_name} (n={s.shape[0]:,}) ST>8" + title_suffix

    # Create figure
    if violin:
        violin_figure(
            x_lbl, y_lbl, y_fit, xq, yq, c1, c2, c3,
            title, filename, showmeans=True, showextrema=False
        )
    else:
        quantiles_figure(
            x_lbl, y_lbl, y_fit, c1, c2, c3, xq, yq, ylabel,
            title, filename, quantiles, labels
        )

    return None


def prep_plot_data_for_quantiles_plot(df, pressure_bins, quantiles, violin=False):
    # PREPARE PLOTTING DATA
    df["pp"] = df.pw_hpa * 100 / P_ATM
    df["quant"] = pd.qcut(df.pp, pressure_bins, labels=False)
    # find linear fit
    c1, c2, c3 = three_c_fit(df)
    print(c1, c2, c3)
    df["de_p"] = c3 * (P_ATM / 100000) * (np.exp(-1 * df.elev / 8500) - 1)
    df["y"] = df.y - df.de_p  # revise y and bring to sea level

    # Find quantile data per bin
    xq = np.zeros(pressure_bins)
    if violin:
        yq = []
    else:
        yq = np.zeros((pressure_bins, len(quantiles)))
    for i, group in df.groupby(df.quant):
        xq[i] = group.pp.median()
        if violin:
            yq.append(group.y.to_numpy())
        else:
            for j in range(len(quantiles)):
                yq[i, j] = group.y.quantile(quantiles[j])
    return c1, c2, c3, xq, yq


def quantiles_figure(x, y, y2, c1, c2, c3, xq, yq, ylabel, title,
                     filename, quantiles, labels, y_sp=None):
    clrs_g = ["#c8d5b9", "#8fc0a9", "#68b0ab", "#4a7c59", "#41624B"]
    clrs_p = ["#C9A6B7", "#995C7A", "#804D67", "663D52", "#4D2E3E"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(alpha=0.3)
    ax.plot(x, y, label="LBL", c="k")
    ax.plot(x, y2, label="fit", ls="--", lw=1, c="g")
    if y_sp is not None:
        ax.plot(x, y_sp, label="Shakespeare", c="0.2", ls=":")
    for i in range(int(len(quantiles) / 2)):
        t = int(-1 * (i + 1))
        ax.fill_between(
            xq, yq[:, i], yq[:, t], alpha=0.3, label=labels[i],
            fc=clrs_g[i], ec="0.9",
        )
        # to make smoother, use default step setting
    text = r"$\varepsilon$ = " + f"{c1} + {c2}" + r"$\sqrt{p_w}$" + \
           f" + {c3}(" + r"$e^{-z/H}$" + " - 1)"
    ax.text(
        0.95, 0.05, s=text, transform=ax.transAxes, ha="right", va="bottom",
        backgroundcolor="1.0", alpha=0.8
    )
    ax.set_xlabel("p$_w$ [-]")
    ax.set_xlim(0, 0.03)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel(ylabel)
    ax.legend(ncols=3, loc="upper left")
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def violin_figure(x, y, y2, xq, yq, c1, c2, c3, title, filename,
                  showmeans=True, showextrema=False):
    clrs_g = ["#c8d5b9", "#8fc0a9", "#68b0ab", "#4a7c59", "#41624B"]

    showmedians = False if showmeans else True
    parts_list = []
    if showmeans:
        parts_list.append("cmeans")
    if showmedians:
        parts_list.append("cmedians")
    if showextrema:
        parts_list.append("cmins")
        parts_list.append("cmaxes")
        parts_list.append("cbars")

    fig, (ax, ax0) = plt.subplots(2, 1, figsize=(8, 5),
                                  sharex=True, height_ratios=[41, 2])
    fig.subplots_adjust(hspace=0.05)
    parts = ax.violinplot(
        yq, xq, showmeans=showmeans,
        showextrema=showextrema,
        showmedians=showmedians,
        widths=np.diff(xq).min(),
    )
    ax.plot(x, y, label="LBL", c="k")
    ax.plot(x, y2, label="fit", ls="--", lw=1, c=clrs_g[-1])
    ax.legend()

    for pc in parts['bodies']:
        pc.set_facecolor(clrs_g[1])
        pc.set_edgecolor(clrs_g[1])
    for p in parts_list:
        vp = parts[p]
        vp.set_edgecolor(clrs_g[2])

    ax0.set_ylim(0, 0.02)
    ax.set_ylim(0.59, 1.0)
    # hide the spines between ax and ax2
    ax.spines.bottom.set_visible(False)
    ax0.xaxis.tick_bottom()
    ax0.spines.top.set_visible(False)
    ax.xaxis.tick_top()
    ax0.tick_params(labeltop=False)  # don't put tick labels at the top
    ax0.set_yticks([0])

    # Draw lines indicating broken axis
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
    ax0.plot([0, 1], [1, 1], transform=ax0.transAxes, **kwargs)
    ax0.set_xlim(0, 0.025)
    ax0.set_xlabel("p$_w$ [-]")
    ax.set_ylabel("effective sky emissivity [-]")

    text = r"$\varepsilon$ = " + f"{c1} + {c2}" + r"$\sqrt{p_w}$" + \
           f" + {c3}(" + r"$e^{-z/H}$" + " - 1)"
    ax.text(
        0.95, 0.0, s=text, transform=ax.transAxes, ha="right", va="bottom",
        backgroundcolor="1.0", alpha=0.8
    )

    ax.grid(True, alpha=0.3)
    ax0.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax0.set_axisbelow(True)
    ax.set_title(title, loc="left")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_fig3_tau():
    # two subplots showing wideband individual and cumulative contributions
    lbl = [
        'H2O', 'CO2', 'O3', 'Aerosols', 'N2O', 'CH4', 'O2', 'N2', 'Overlaps'
    ]
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(len(lbl))]

    df = ijhmt_to_tau("fig3_esky_i.csv")  # tau, first p removed
    x = df.index.to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    i = 0
    y_ref = np.ones(len(x))
    for s in lbl:
        col = s if i == 0 else f"p{s}"
        ax1.plot(x, df[col], label=s, c=cmaplist[i])
        ax2.plot(x, y_ref * df[col], label=s, c=cmaplist[i])
        y_ref = y_ref * df[col]
        i += 1
    ax1.plot(x, y_ref, label="total", c="0.0", ls="--")
    ax2.plot(x, y_ref, label="total", c="0.0", ls="--")
    ax1.set_title("Individual contributions", loc="left")
    ax2.set_title("Cumulative transmissivity", loc="left")
    ax1.set_xlim(x[0], x[-1])
    ax1.set_xlabel("$p_w$ [-]")
    ax2.set_xlabel("$p_w$ [-]")
    ax1.set_ylabel("transmissivity [-]")
    ax2.set_ylabel("transmissivity [-]")
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 0.5)
    ax2.grid(alpha=0.3)
    ax1.grid(alpha=0.3)
    ax2.legend(ncol=2, loc="upper right")
    # plt.show()
    filename = os.path.join("figures", "fig3_tau.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


def plot_fit3_dopt():
    # single plot showing d_opt vs pw
    lbl = [
        'H2O', 'CO2', 'O3', 'Aerosols', 'N2O', 'CH4', 'O2', 'N2', 'Overlaps'
    ]
    labels = [
        'H$_2$O', 'CO$_2$', 'O$_3$', 'aerosols',
        'N$_2$O', 'CH$_4$', 'O$_2$', 'N$_2$', 'overlaps'
    ]
    cmap = mpl.colormaps["Paired"]
    cmaplist = [cmap(i) for i in range(len(lbl))]

    # df = ijhmt_to_tau("fig3_esky_i.csv")
    df = ijhmt_to_tau("fig5_esky_ij_b4.csv")  # tau, first p removed
    # df = ijhmt_to_individual_e("fig3_esky_i.csv")
    x = df.index.to_numpy()

    fig, ax = plt.subplots(figsize=(5, 5), sharex=True)
    i = 0
    y_ref = np.zeros(len(x))
    # y_ref = np.ones(len(x))
    for s in lbl:
        y = -1 * np.log(df[s])
        # y = df[s]
        ax.plot(x, y, label=labels[i], c=cmaplist[i])
        # y_ref = y_ref * y  # transmissivity
        y_ref += y  # dopt or emissivity
        i += 1
    ax.plot(x, y_ref, label="total", c="0.0", ls="--")
    ax.set_title("Individual contributions (band 4)", loc="left")
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("$p_w$ [-]")
    ax.set_ylabel(r"$d_{\rm{opt}}$ [-]")
    # ax.set_ylabel("emissivity [-]")
    ax.set_ylim(0, 0.02)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, loc="upper right")
    plt.tight_layout()
    plt.show()
    filename = os.path.join("figures", "temp.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    return None


if __name__ == "__main__":
    print()