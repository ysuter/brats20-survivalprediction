#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode_eor(inp: str):
    if type(inp) != str:
        if not np.isfinite(inp):
            return np.NaN
    elif inp == "GTR":
        return 1
    else:
        return 0

inpfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/training_allfeat_bins.csv",
                      index_col="ID")
# handle alive patient with setting OS to 500 days
inpfeat.loc[inpfeat["Survival_days"] == 'ALIVE (361 days later)', "Survival_days"] = 500
inpfeat["Survival_days"] = inpfeat["Survival_days"].astype(np.float)

# encode EOR
inpfeat["Extent_of_Resection"] = [encode_eor(elem) for elem in inpfeat["Extent_of_Resection"]]
featprocess_nosurv = inpfeat.drop(columns=["Survival_days"])

#  check mutual correlation of features
corr_matrix = featprocess_nosurv.corr()
# corr_matrix.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_trainingfeat.csv")

f = plt.figure(figsize=(100, 100))
plt.matshow(corr_matrix, fignum=f.number)
plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=14, rotation=45)
plt.yticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix.png")
plt.show()
