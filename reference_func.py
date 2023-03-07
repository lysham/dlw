"""Keep functions not in use but helpful for referencing"""

from main import *
from corr26b import get_tsky


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


if __name__ == "__main__":
    print()