
import os
import pandas as pd
from matplotlib import pyplot as plt

trainfeatpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/brats20_pyradfeatures.csv"

trainfeat_raw = pd.read_csv(trainfeatpath)

# drop diagnostic columns
# filter out diagnostics columns
nondiagnostics_cols = ['diagnostics' not in elem for elem in list(trainfeat_raw.columns)]

featinp_filt = trainfeat_raw.loc[:, nondiagnostics_cols]

t1cvolfeat = featinp_filt.loc[(featinp_filt["Sequence"] == "T1c") & (featinp_filt["Labelname"] == "enhancing"), ["Survival_days", "Age", "original_shape_VoxelVolume"]]

t1cvolfeat.loc[t1cvolfeat["Survival_days"] == "ALIVE (361 days later)", "Survival_days"] = 500

plt.scatter(t1cvolfeat["Survival_days"], t1cvolfeat["original_shape_VoxelVolume"])
plt.show()

plt.scatter(t1cvolfeat["Survival_days"], t1cvolfeat["Age"])
plt.show()


