## Overview
The HDF5 file contains training and validation data that was used to 
generate the results for the manuscript by Matsunobu and Coimbra, 
submitted to JGR: Atmospheres, 2023

Open access SURFRAD data was processed using the techniques outlined in the
associated manuscript. Data is indexed by solar time and provided per site for
2010 through 2015.

'tra' and 'val' columns indicate whether the given sample was included in the
training or validation set, respectively.
Training and validation sets can be reconstructed by concatenating
all 'tra' or 'val' samples across sites. Sample code is given below to 
reconstruct original training and validation sets.

***
## Data labels

Column names and descriptions are as follows:
- dlw_m: measured downwelling longwave [W/m^2]
- ghi_m: measured global horizontal irradiance [W/m^2]
- dni_m: measured direct normal irradiance [W/m^2]
- dhi_m: measured diffuse horizontal irradiance [W/m^2]
- rh_m: measured relative humidity [%]
- pa_m: measured atmospheric pressure [hPa]
- t_m: measured temperature [K]
- sza: solar zenith angle [deg]
- ghi_c: clear sky global horizontal irradiance [W/m^2]
- dni_c: clear sky direct normal irradiance [W/m^2]
- dhi_c: clear sky diffuse horizontal irradiance [W/m^2]
- cs1: clear sky filter 1
- cs2: clear sky filter 2
- site_elev: station elevation [m]
- clr_pct: fraction of samples identified as clear for the given site and day
- clr_num: number of samples identified as clear for the given site and day
- pw_hpa: water vapor partial pressure [hPa]
- alt_correction: altitude correction
- tra: indicate if sample is included in training set
- val: indicate if sample is included in validation set
- sqrt_pw: square root of non-dimensional water vapor partial pressure
- e_sky: effective clear sky emissivity

The last two columns, 'sqrt_pw' and 'e_sky' represent the input and target
for linear regression, i.e. e_sky = c_1 + (c_2 * sqrt_pw).
Altitude corrected sky emissivity, or expected emissivity for a station at
sea-level, is found by e_sky - alt_correction.

***
## Sample code (Python v3.8)

```
import pandas as pd

site = "GWC"  # or other station code
df = pd.read_hdf("data.h5", key=site)  # import single site
# --> modify "data.h5" to "data_sample.h5"
```

Training and validation sets can be reconstructed as below.
Linear regression on `sqrt_pw` to predict `e_sky` - `alt_correction` in the
resultant training set will reproduce results in the associated manuscript.
```
training = []
validation = []
surfrad_sites = ['BON', 'DRA', 'FPK', 'GWC', 'PSU', 'SXF', 'TBL']
# --> only ['GWC'] included in `data_sample.h5`
for site in surfrad_sites:  # loop through sites
    df = pd.read_hdf("data.h5", key=site)
    df["site"] = site  # add site name
    training.append(df.loc[df.tra])  # append samples marked as training
    validation.append(df.loc[df.val])  # append samples marked as validation
# join respective set samples across sites
training = pd.concat(training, ignore_index=False)
validation = pd.concat(validation, ignore_index=False)
```

Reproduce regression results
```
from sklearn.linear_model import LinearRegression

c1 = 0.6  # set intercept (c1 constant)
x = training.sqrt_pw.to_numpy().reshape(-1, 1)
y = training.e_sky - training.alt_correction - c1  # adjust for altitude and c1
y = y.to_numpy().reshape(-1, 1)

model = LinearRegression(fit_intercept=False)
model.fit(x, y)

c2 = model.coef_[0][0]
print(f"c1={c1:.3f}, c2={c2:.3f}")
# output: c1=0.600, c2=1.652
```

Results can be explored by modifying choice of filters and constructing new
training and validation sets.
