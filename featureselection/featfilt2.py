#!/usr/bin/env python3

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm


def encode_eor(inp: str):
    if type(inp) != str:
        if not np.isfinite(inp):
            return np.NaN
    elif inp == "GTR":
        return 1
    else:
        return 0


cph = CoxPHFitter()

# inpfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/training_allfeat_bins.csv",
#                       index_col="ID")
inpfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_nobinduplicates.csv",
                      index_col="ID")
# handle alive patient with setting OS to 500 days
inpfeat.loc[inpfeat["Survival_days"] == 'ALIVE (361 days later)', "Survival_days"] = 500
inpfeat["Survival_days"] = inpfeat["Survival_days"].astype(np.float)

# encode EOR
inpfeat["Extent_of_Resection"] = [encode_eor(elem) for elem in inpfeat["Extent_of_Resection"]]

madvalues = inpfeat.mad(axis=0)

survival = inpfeat["Survival_days"]
featprocess_nosurv = inpfeat.drop(columns=["Survival_days"])

cind_df = pd.DataFrame(np.ones([featprocess_nosurv.shape[1], 3])*np.NaN, columns=["Feature", "ConcordanceIndex",
                                                                                  "Numsamples"])
cind_df["Feature"] = featprocess_nosurv.columns
cind_df.set_index(keys="Feature", inplace=True)

# check for columns with on NaNs
nantest = np.sum(pd.isnull(featprocess_nosurv), axis=0)
featprocess_nosurv.columns[nantest == featprocess_nosurv.shape[0]]
nancolidx = np.where(nantest == featprocess_nosurv.shape[0])

for col in tqdm(featprocess_nosurv.columns.values):
    indexname = col
    currfeat = inpfeat[[col, "Survival_days"]]

    # drop NaNs
    currfeat.dropna(axis=0, inplace=True)
    numsamples = currfeat.shape[0]

    cph.fit(currfeat, duration_col='Survival_days', show_progress=False, step_size=0.1)
    ci = cph.score_
    cind_df.loc[col] = [cph.score_, numsamples]
    # cind_df.to_csv('/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/concordanceidx_training.csv')

    # ri_selected = ri_selected.drop(columns=['values_patient'], axis=1)
cind_df.to_csv('/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/concordanceidx_training_nobinduplicates.csv')
