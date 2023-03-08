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

SURF_ASOS = dict(
    BON=dict(usaf=999999, wban=54808),
    BOU=dict(usaf=720533, wban=160),
    DRA=dict(usaf=723870, wban=3160),
    FPK=dict(usaf=727686, wban=94017),
    GWC=dict(usaf=720541, wban=53806),
    PSU=dict(usaf=725105, wban=14770),
    SXF=dict(usaf=711680, wban=99999),
)

SURF_COLS = [
    'yr', 'jday', 'month', 'day', 'hr', 'minute', 'dt', 'zen', 'dw_solar',
    'qc1', 'uw_solar', 'qc2', 'direct_n', 'qc3', 'diffuse', 'qc4', 'dw_ir',
    'qc5', 'dw_casetemp', 'qc6', 'dw_dometemp', 'qc7', 'uw_ir', 'qc8',
    'uw_castemp', 'qc9', 'uw_dometemp', 'qc10', 'uvb', 'qc11', 'par', 'qc12',
    'netsolar', 'qc13', 'netir', 'qc14', 'totalnet', 'qc15', 'temp', 'qc16',
    'rh', 'qc17', 'windspd', 'qc18', 'winddir', 'qc19', 'pressure', 'qc20'
]  # used in process_site in corr26b.py