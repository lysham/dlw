"""explore linear perturbation of lbl correlations"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


p0 = np.linspace(0.001, 2.5/100, 20)
frac_change_p = np.array([0.005, 0.01, 0.02, 0.03])

b2_h2o = 0.1083 + 0.0748 * np.tanh(270.8944 * p0)
b3_h2o = -0.2308 + 0.6484 * np.power(p0, 0.1280)
b4_h2o = 0.0289 + 6.2436 * np.power(p0, 0.9010)
de_b2_h2o = 20.2629 * np.power(np.cosh(270.8944 * p0), -2)
de_b3_h2o = (0.6484 * 0.1280) * np.power(p0, 0.1280 - 1)
de_b4_h2o = (6.2436 * 0.9010) * np.power(p0, 0.9010 - 1)

b2_co2 = 0.0002
b3_co2 = 0.3038 - 0.5262 * np.power(p0, 0.1497)
b4_co2 = 0.0144 - 0.1740 * np.power(p0, 0.7268)
de_b2_co2 = np.zeros(len(p0))
de_b3_co2 = (-0.5262 * 0.1497) * np.power(p0, 0.1497 - 1)
de_b4_co2 = (-0.1740 * 0.7268) * np.power(p0, 0.7268 - 1)

total2 = (de_b2_co2 / b2_co2) + (de_b2_h2o / b2_h2o) + (de_b3_h2o / b3_h2o) + \
         (de_b4_co2 / b4_co2) + (de_b4_h2o / b4_h2o)
total = (de_b3_co2 / b3_co2) + (de_b3_h2o / b3_h2o) + \
        (de_b4_co2 / b4_co2) + (de_b4_h2o / b4_h2o)

fig, ax = plt.subplots()
ax.grid(alpha=0.3)
for f in frac_change_p:
    frac_change_e = (f * p0) * total
    ax.plot(p0, frac_change_e, label=f)
ax.legend(title="frac change in p$_w$")
ax.set_ylabel("fractional change in emissivity")
ax.set_xlabel("dimensionless water vapor (p$_0$)")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p0, total2, label="B2, B3, B4")
ax.plot(p0, total - (de_b3_co2 / b3_co2), label="B3, B4")
title = "Multiplier term (B3 contains H2O only)"
ax.set_title(title, loc="left")
ax.grid(alpha=0.3)
ax.legend()
ax.set_xlabel("dimensionless water vapor (p$_0$)")
ax.set_ylim(-200, 1400)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(p0, total)
ax.plot(p0, de_b3_co2, label="de/dp")
ax.plot(p0, b3_co2, label="e")
ax.plot(p0, (de_b3_co2 / b3_co2), label="(de/dp) / e")
ax.set_title("CO2 B3", loc="left")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlabel("dimensionless water vapor (p$_0$)")
plt.show()