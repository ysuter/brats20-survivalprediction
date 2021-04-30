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

# inpfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/training_allfeat_bins.csv",
#                       index_col="ID")
# inpfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_nobinduplicates.csv",
#                       index_col="ID")
inpfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_nobinduplicates_Cg055.csv",
                      index_col="ID")
# load c-index information
cindices = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/concordanceidx_training_nobinduplicates.csv", index_col="Feature")
# handle alive patient with setting OS to 500 days
inpfeat.loc[inpfeat["Survival_days"] == 'ALIVE (361 days later)', "Survival_days"] = 500
inpfeat["Survival_days"] = inpfeat["Survival_days"].astype(np.float)

# encode EOR
# inpfeat["Extent_of_Resection"] = [encode_eor(elem) for elem in inpfeat["Extent_of_Resection"]]
featprocess_nosurv = inpfeat.drop(columns=["Survival_days"])

#  check mutual correlation of features
print("- calculating correlation matrix")
corr_matrix = featprocess_nosurv.corr().abs()
print("- finished calculating correlation matrix")
# # corr_matrix.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_trainingfeat.csv")

# corr_matrix = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_trainingfeat.csv")
# corr_matrix.set_index("Unnamed: 0", inplace=True)
print("Data loaded.")

# save correlation matrix
corr_matrix.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_c055.csv")
corr_np = corr_matrix.to_numpy()
mask = np.triu(np.ones_like(corr_np, dtype=np.bool))
corr_masked = corr_matrix.mask(mask)

f = plt.figure(figsize=(200, 200))
plt.matshow(corr_masked)
# plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=5, rotation=45)
plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=2, rotation=90)
plt.yticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=2)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
# plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_cg055.png", dpi=400)
plt.show()
# plt.savefig("/media/yan
#
# # get highly correlated features
# corr_condition = corr_masked[corr_masked > 0.95]  # filter values, output is not boolean!
# testidx = corr_masked[corr_masked > 0.95].stack().index.tolist()
#
# featdroplist = []
# # for each highly correlated feature pair, only keep the one with the higher c-index
# for featcomb in testidx:
#     # look up c-indices of both features, keep the one with the larger
#     curr_cindlist = [cindices.loc[elem, "ConcordanceIndex"] for elem in featcomb]
#     # add the lower one to the drop list
#     featdroplist.append(featcomb[np.argmin(curr_cindlist)])
#
# print(featdroplist)
# featdroplist_unique = np.unique(featdroplist)
# feat_filtered = inpfeat.drop(columns=featdroplist_unique, inplace=False)
# feat_filtered.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_Cg055_nocorrelated.csv")
#
# flatcorr = corr_matrix.values.flatten()
# # plot correlation distribution
# plt.hist(corr_matrix.values.flatten())
# plt.show()
#
# f = plt.figure(figsize=(15, 15))
# plt.matshow(corr_matrix)
# plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=5, rotation=45)
# plt.yticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=54)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=10)
# plt.title('Correlation Matrix', fontsize=16)
# plt.savefig("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix.png")
# plt.show()
# #
# #
# #
# # f = plt.figure(figsize=(20, 20))
# # plt.matshow(corr_matrix.abs())
# # # plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=14, rotation=45)
# # # plt.yticks(range(corr_matrix.shape[1]), corr_matrix.columns, fontsize=14)
# # cb = plt.colorbar()
# # cb.ax.tick_params(labelsize=10)
# # plt.title('Correlation Matrix', fontsize=16)
# # plt.savefig("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_abs.png", dpi=300)
# # plt.show()
# #
# mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
# # # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.savefig("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/corrmatrix_absns.png", dpi=300)
# plt.show()
