#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="ticks", font="Serif", palette="deep")


def filterbest(inputdf: pd.DataFrame, bestparams):

    # check if param1 is NaN
    if pd.isna(bestparams[2]):
        outdf = inputdf.loc[(inputdf["Feature selector"] == bestparams[0]) & (inputdf["ML method"] == bestparams[1])]

    elif pd.isna(bestparams[3]):
        outdf = inputdf.loc[(inputdf["Feature selector"] == bestparams[0]) & (inputdf["ML method"] == bestparams[1])
                            & (inputdf["Parameter1"] == bestparams[2])]

    else:
        outdf = inputdf.loc[(inputdf["Feature selector"] == bestparams[0])
                                        & (inputdf["ML method"] == bestparams[1])
                                        & (inputdf["Parameter1"] == bestparams[2])
                                        & (inputdf["Parameter2"] == bestparams[3])]

    return outdf


def bestselectsplits(inputdf: pd.DataFrame, bestvals: np.array, modelname: str):
    outdf = filterbest(inputdf, bestvals)
    outdf.drop(columns=["Parameter1", "Parameter2", "ML method", "Feature selector"],
                                     inplace=True)
    outdf.loc[:, "Model"] = [modelname] * outdf.shape[0]
    outdf.set_index("Split")

    return outdf


outpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/cvresults"

# results without priors
# nopriors = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_acc.csv",
#                        index_col=["ML method", "Feature selector", "Parameter1", "Parameter2"])
manual = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_ageshape_regression_manual.csv")

# only use best parameter for each method, aggregate by parameter first
manual_perparam = manual.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
manual_perparam_sorted_acc = manual_perparam.reset_index().sort_values(by="Accuracy")
manual_perparam_sorted_balacc = manual_perparam.reset_index().sort_values(by="Balanced Accuracy")
manual_onlybest_acc = manual_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
manual_onlybest_balacc = manual_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
manual_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
manual_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])

manual_onlybest_acc["Feature selector"] = ["1"]

manual_bestidx_acc = np.where(manual_onlybest_acc["Accuracy"].values == np.max(manual_onlybest_acc["Accuracy"].values))
manual_bestidx_balacc = np.where(manual_onlybest_balacc["Balanced Accuracy"].values == np.max(manual_onlybest_balacc["Balanced Accuracy"].values))
manual_aggr_wide_acc = manual_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
manual_aggr_wide_balacc = manual_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
manual_bestparams_acc = manual_onlybest_acc.iloc[manual_bestidx_acc[0]]
manual_bestparams_balacc = manual_onlybest_balacc.iloc[manual_bestidx_balacc[0]]
print("Manual, accuracy:")
print(manual_bestparams_acc)
print(manual_bestparams_acc.values)
print("Manual, balanced accuracy:")
print(manual_bestparams_balacc)
print(manual_bestparams_balacc.values)
print('------')

# save to csv for inspection
manual_aggr_wide_acc.to_csv(os.path.join(outpath, "manual_acc_regression.csv"))
manual_aggr_wide_balacc.to_csv(os.path.join(outpath, "manual_balacc_regression.csv"))

overallmin = np.min([manual_aggr_wide_acc.values, manual_aggr_wide_balacc.values])
overallmax = np.max([manual_aggr_wide_acc.values, manual_aggr_wide_balacc.values])

fig = plt.figure()
ax = sns.heatmap(manual_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, linewidths=.5, annot_kws={"size": 8})
plt.title('No priors: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_manual_acc_regression.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(manual_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, linewidths=.5, annot_kws={"size": 8})
plt.title('No priors: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_manual_balacc_regression.png"), bbox_inches="tight")
plt.show()
