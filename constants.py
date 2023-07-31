SIGMA = 5.6697e-8  # W/m^2 K^4
P_ATM = 101325  # Pa
P_ATM_BAR = P_ATM / 100000

# radiative constants
E_C1 = 3.7418e8  # W um^4 / m^2 ~> 2*pi*h*c^2
E_C2 = 1.4389e4  # um K         ~> hc/k_B

SURFRAD_SITES = [
    'Bondville_IL', 'Boulder_CO', 'Desert_Rock_NV', 'Fort_Peck_MT',
    'Goodwin_Creek_MS', 'Penn_State_PA', 'Sioux_Falls_SD']  # Tbl 1
SURF_SITE_CODES = ['BON', 'BOU', 'DRA', 'FPK', 'GWC', 'PSU', 'SXF']
LATs = [40.05, 40.13, 36.62, 48.31, 34.25, 40.72, 43.73]
LONs = [-88.37, -105.24, -116.06, -105.10, -89.87, -77.93, -96.62]
ALTs = [213, 1689, 1007, 634, 98, 376, 437]  # m

SURFRAD = dict(
    BON=dict(name="Bondville_IL", lat=40.05, lon=-88.37, alt=213, rho=0.247),
    BOU=dict(name="Boulder_CO", lat=40.13, lon=-105.24, alt=1689, rho=0.199),
    DRA=dict(name="Desert_Rock_NV", lat=36.62, lon=-116.06, alt=1007, rho=0.211),
    FPK=dict(name="Fort_Peck_MT", lat=48.31, lon=-105.10, alt=634, rho=0.247),
    GWC=dict(name="Goodwin_Creek_MS", lat=34.25, lon=-89.87, alt=98, rho=0.200),
    PSU=dict(name="Penn_State_PA", lat=40.72, lon=-77.93, alt=376, rho=0.252),
    SXF=dict(name="Sioux_Falls_SD", lat=43.73, lon=-96.62, alt=437, rho=0.238),
)
# pd.DataFrame.from_dict(SURFRAD, orient="index")
# albedo and surface type from Marion, 2021, osti 1763970

ELEV_DICT = {}
LON_DICT = {}
for i in SURFRAD:
    ELEV_DICT[i] = SURFRAD[i]["alt"]
    LON_DICT[i] = SURFRAD[i]["lon"]
ELEVATIONS = sorted(ELEV_DICT.items(), key=lambda x: x[1])  # sorted list

# "#F8E16C" old SXF: "#FFC2B4",
SEVEN_COLORS = ["#156064", "#00C49A", "#EAC50B", "#99201E",
                "#FB8F67", "#437C90", "#7E2E84"]
COLOR7_DICT = {}
for i in range(len(ELEVATIONS)):
    COLOR7_DICT[ELEVATIONS[i][0]] = SEVEN_COLORS[i]

# Generate from integrating shakespeare data (function in ref_func)
SITE_H_DICT = {
    'BON': 2283.261266476327, 'BOU': 2036.589702450849,
    'DRA': 2772.565558794065, 'FPK': 2504.2325580970546,
    'GWC': 2253.5745262708365, 'PSU': 2284.1533602287977,
    'SXF': 2306.1676698118363
}


SURF_ASOS = dict(
    BON=dict(usaf=725315, wban=94870),
    BOU=dict(usaf=720533, wban=160),
    DRA=dict(usaf=723870, wban=3160),
    FPK=dict(usaf=727686, wban=94017),
    GWC=dict(usaf=720541, wban=53806),
    PSU=dict(usaf=725105, wban=14770),
    SXF=dict(usaf=726510, wban=14944),
)

SURF_COLS = [
    'yr', 'jday', 'month', 'day', 'hr', 'minute', 'dt', 'zen', 'dw_solar',
    'qc1', 'uw_solar', 'qc2', 'direct_n', 'qc3', 'diffuse', 'qc4', 'dw_ir',
    'qc5', 'dw_casetemp', 'qc6', 'dw_dometemp', 'qc7', 'uw_ir', 'qc8',
    'uw_castemp', 'qc9', 'uw_dometemp', 'qc10', 'uvb', 'qc11', 'par', 'qc12',
    'netsolar', 'qc13', 'netir', 'qc14', 'totalnet', 'qc15', 'temp', 'qc16',
    'rh', 'qc17', 'windspd', 'qc18', 'winddir', 'qc19', 'pressure', 'qc20'
]  # used in process_site in corr26b.py

# broadband values
LI_TABLE1 = {
    "H2O": [0.2996, 2.2747, 0.3784],
    "CO2": [0.2893, -0.5640, 0.1821],
    "O3": [0.0126, -0.5119, 1.1744],
    "aerosols": [0.0191, -0.1421, 0.6121],
    "N2O": [13.8712, -13.8761, 0.0001],
    "CH4": [0.0245, -0.0313, 0.0790],
    "O2": [0, 0, 0],
    "N2": [0, 0, 0],
    "overlaps": [0.0524, -0.1423, 0.2998],
    "total": [0.6173, 1.6940, 0.5035]
}
LBL_LABELS = {  # used for labeling in plots
    "H2O": "H$_2$O",
    "CO2": "CO$_2$",
    "O3": "O$_3$",
    "aerosols": "aerosols",
    "N2O": "N$_2$O",
    "CH4": "CH$_4$",
    "O2": "O$_2$",
    "N2": "N$_2$",
    "overlaps": "overlaps",
    "total": "total"
}
N_SPECIES = len(LI_TABLE1) - 1  # number of contributing components (from TABLE 1)
N_BANDS = 7  # number of bands

BANDS_V = {  # cm^-1
    "b1": (0.00001, 400),
    "b2": (400, 580),
    "b3": (580, 750),
    "b4": (750, 1400),
    "b5": (1400, 2250),
    "b6": (2250, 2400),
    "b7": (2400, 2500),
}

BANDS_L = {  # micron
    "b1": (25, 100000),
    "b2": (17.2, 25),
    "b3": (13.3, 17.2),
    "b4": (7.1, 13.3),
    "b5": (4.4, 7.1),
    "b6": (4.2, 4.4),
    "b7": (4.0, 4.2),
}
