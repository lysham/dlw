SIGMA = 5.6697e-8  # W/m^2 K^4
P_ATM = 101325  # Pa

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
    BON=dict(name="Bondville_IL", lat=40.05, lon=-88.37, alt=213),
    BOU=dict(name="Boulder_CO", lat=40.13, lon=-105.24, alt=1689),
    DRA=dict(name="Desert_Rock_NV", lat=36.62, lon=-116.06, alt=1007),
    FPK=dict(name="Fort_Peck_MT", lat=48.31, lon=-105.10, alt=634),
    GWC=dict(name="Goodwin_Creek_MS", lat=34.25, lon=-89.87, alt=98),
    PSU=dict(name="Penn_State_PA", lat=40.72, lon=-77.93, alt=376),
    SXF=dict(name="Sioux_Falls_SD", lat=43.73, lon=-96.62, alt=437),
)

ELEV_DICT = {}
LON_DICT = {}
for i in SURFRAD:
    ELEV_DICT[i] = SURFRAD[i]["alt"]
    LON_DICT[i] = SURFRAD[i]["lon"]
ELEVATIONS = sorted(ELEV_DICT.items(), key=lambda x: x[1])  # sorted list

SEVEN_COLORS = ["#156064", "#00C49A", "#F8E16C", "#FFC2B4",
                "#FB8F67", "#437C90", "#7E2E84"]

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
    "overlaps": [0.0524, -0.1423, 0.2998]
}
N_SPECIES = 7  # number of contributing components (from TABLE 1)
N_BANDS = 7  # number of bands