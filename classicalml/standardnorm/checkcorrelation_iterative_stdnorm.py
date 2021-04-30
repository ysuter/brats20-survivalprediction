#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def encode_eor(inp: str):
    if type(inp) != str:
        if not np.isfinite(inp):
            return np.NaN
    elif inp == "GTR":
        return 1
    else:
        return 0


inpfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_Cg055_stdnorm.csv",
                      index_col="ID")
# load c-index information
cindices = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/concordanceidx_training_nobinduplicates_stdnorm.csv", index_col="Feature")
# handle alive patient with setting OS to 500 days
inpfeat.loc[inpfeat["Survival_days"] == 'ALIVE (361 days later)', "Survival_days"] = 500
inpfeat["Survival_days"] = inpfeat["Survival_days"].astype(np.float)

# encode EOR
# inpfeat["Extent_of_Resection"] = [encode_eor(elem) for elem in inpfeat["Extent_of_Resection"]]
featprocess_nosurv = inpfeat.drop(columns=["Survival_days"])
featprocess_nosurv = inpfeat.drop(columns=["Survival_class"])

#  check mutual correlation of features
print("- calculating correlation matrix")
corr_matrix = featprocess_nosurv.corr().abs()
print("- finished calculating correlation matrix")
# # corr_matrix.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_trainingfeat.csv")

# corr_matrix = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_trainingfeat.csv")
# corr_matrix.set_index("Unnamed: 0", inplace=True)
print("Data loaded.")

# save correlation matrix
corr_matrix.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_c055_stdnorm.csv")
corr_np = corr_matrix.to_numpy()
mask = np.triu(np.ones_like(corr_np, dtype=np.bool))
corr_masked = corr_matrix.mask(mask)

maxcorr = np.nanmax(corr_masked.values.flatten())
curr_corrmat = corr_masked

currfeat = featprocess_nosurv
iterateidx = 0
while maxcorr > 0.95:
    print(iterateidx)
    testidx = corr_masked[corr_masked == maxcorr].stack().index.tolist()

    featdroplist = []
    # for each highly correlated feature pair, only keep the one with the higher c-index
    for featcomb in testidx:
        # look up c-indices of both features, keep the one with the larger
        curr_cindlist = [cindices.loc[elem, "ConcordanceIndex"] for elem in featcomb]
        # add the lower one to the drop list
        featdroplist.append(featcomb[np.argmin(curr_cindlist)])

    featdroplist_unique = np.unique(featdroplist)
    currfeat.drop(columns=featdroplist_unique, inplace=True)

    curr_corrmat = currfeat.corr().abs()
    corr_np = curr_corrmat.to_numpy()
    mask = np.triu(np.ones_like(corr_np, dtype=np.bool))
    corr_masked = curr_corrmat.mask(mask)

    maxcorr = np.nanmax(corr_masked.values.flatten())
    print(maxcorr)
    print(currfeat.shape)
    iterateidx += 1
    print('----------')

print(currfeat.shape)
# put survival column back into the feature matrix
survinfo = inpfeat["Survival_days"]
iterativecorr = currfeat.merge(survinfo, on="ID")
iterativecorr.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_iterativeremoved_stdnorm.csv")

# plot correlation matrix
f = plt.figure(figsize=(200, 200))
plt.matshow(corr_masked)
# only show group ticks
feattypes = [elem.split('_')[0:2] for elem in currfeat.columns]
labels = np.array([[0,15],[16,36],[37,82],[83,111],[112,149]])

# plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=5, rotation=45)
# plt.xticks(range(corr_masked.shape[1]), corr_masked.columns, fontsize=2, rotation=90)
# plt.yticks(range(corr_masked.shape[1]), corr_masked.columns, fontsize=2)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
# plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/reducedcorr_iterative_stdnorm.png", dpi=400)
plt.show()
