"""Keep functions not in use but helpful for referencing"""

from main import *
import scipy
from sklearn.metrics import mean_squared_error
from corr26b import get_tsky, join_surfrad_asos, shakespeare, \
    shakespeare_comparison, import_cs_compare_csv
from fraction import fe_lt, fi_lt

from constants import *


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
    # Get scale height H
    lon_pts = f["lon"]
    lat_pts = np.flip(f["lat"])  # must be in ascending order for interp
    h = np.flip(f["H"], axis=1)
    xx = np.rot90(h)
    plt.imshow(xx, norm="log")
    plt.colorbar()
    plt.show()
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


if __name__ == "__main__":
    print()