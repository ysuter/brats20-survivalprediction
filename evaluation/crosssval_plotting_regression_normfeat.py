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
nopriors = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_regression2.csv")

# results with sequence prior
seqprior = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_seqprior_regression2.csv")

# results with robustness prior
robprior = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_robprior_regression2.csv")

# results with both priors
robseqprior = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_robseqprior_regression2.csv")

# results with age, shape and position features
ageshape = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_ageshape_regression.csv")


# only use best parameter for each method, aggregate by parameter first
nopriors_perparam = nopriors.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
nopriors_perparam_sorted_acc = nopriors_perparam.reset_index().sort_values(by="Accuracy")
nopriors_perparam_sorted_balacc = nopriors_perparam.reset_index().sort_values(by="Balanced Accuracy")
nopriors_onlybest_acc = nopriors_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
nopriors_onlybest_balacc = nopriors_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
nopriors_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
nopriors_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
nopriors_bestidx_acc = np.where(nopriors_onlybest_acc["Accuracy"].values == np.max(nopriors_onlybest_acc["Accuracy"].values))
nopriors_bestidx_balacc = np.where(nopriors_onlybest_balacc["Balanced Accuracy"].values == np.max(nopriors_onlybest_balacc["Balanced Accuracy"].values))
nopriors_aggr_wide_acc = nopriors_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
nopriors_aggr_wide_balacc = nopriors_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
nopriors_bestparams_acc = nopriors_onlybest_acc.iloc[nopriors_bestidx_acc[0]]
nopriors_bestparams_balacc = nopriors_onlybest_balacc.iloc[nopriors_bestidx_balacc[0]]
print("No prior, accuracy:")
print(nopriors_bestparams_acc)
print(nopriors_bestparams_acc.values)
print("No prior, balanced accuracy:")
print(nopriors_bestparams_balacc)
print(nopriors_bestparams_balacc.values)
print('------')

seqprior_perparam = seqprior.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
seqprior_perparam_sorted_acc = seqprior_perparam.reset_index().sort_values(by="Accuracy")
seqprior_perparam_sorted_balacc = seqprior_perparam.reset_index().sort_values(by="Balanced Accuracy")
seqprior_onlybest_acc = seqprior_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
seqprior_onlybest_balacc = seqprior_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
seqprior_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
seqprior_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
seqprior_bestidx_acc = np.where(seqprior_onlybest_acc["Accuracy"].values == np.max(seqprior_onlybest_acc["Accuracy"].values))
seqprior_bestidx_balacc = np.where(seqprior_onlybest_balacc["Balanced Accuracy"].values == np.max(seqprior_onlybest_balacc["Balanced Accuracy"].values))
seqprior_aggr_wide_acc = seqprior_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
seqprior_aggr_wide_balacc = seqprior_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
seqprior_bestparams_acc = seqprior_onlybest_acc.iloc[seqprior_bestidx_acc[0]]
seqprior_bestparams_balacc = seqprior_onlybest_balacc.iloc[seqprior_bestidx_balacc[0]]
print("Sequence prior, accuracy:")
print(seqprior_bestparams_acc)
print(seqprior_bestparams_acc.values)
print("Sequence prior, balanced accuracy:")
print(seqprior_bestparams_balacc)
print(seqprior_bestparams_balacc.values)
print('------')

robprior_perparam = robprior.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
robprior_perparam_sorted_acc = robprior_perparam.reset_index().sort_values(by="Accuracy")
robprior_perparam_sorted_balacc = robprior_perparam.reset_index().sort_values(by="Balanced Accuracy")
robprior_onlybest_acc = robprior_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
robprior_onlybest_balacc = robprior_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
robprior_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
robprior_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
robprior_bestidx_acc = np.where(robprior_onlybest_acc["Accuracy"].values == np.max(robprior_onlybest_acc["Accuracy"].values))
robprior_bestidx_balacc = np.where(robprior_onlybest_balacc["Balanced Accuracy"].values == np.max(robprior_onlybest_balacc["Balanced Accuracy"].values))
robprior_aggr_wide_acc = robprior_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
robprior_aggr_wide_balacc = robprior_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
robprior_bestparams_acc = robprior_onlybest_acc.iloc[robprior_bestidx_acc[0]]
robprior_bestparams_balacc = robprior_onlybest_balacc.iloc[robprior_bestidx_balacc[0]]
print("Robustness prior, accuracy:")
print(robprior_bestparams_acc)
print(robprior_bestparams_acc.values)
print("Robustness prior, balanced accuracy:")
print(robprior_bestparams_balacc)
print(robprior_bestparams_balacc.values)
print('------')

robseqprior_perparam = robseqprior.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
robseqprior_perparam_sorted_acc = robseqprior_perparam.reset_index().sort_values(by="Accuracy")
robseqprior_perparam_sorted_balacc = robseqprior_perparam.reset_index().sort_values(by="Balanced Accuracy")
robseqprior_onlybest_acc = robseqprior_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
robseqprior_onlybest_balacc = robseqprior_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
robseqprior_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
robseqprior_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
robseqprior_bestidx_acc = np.where(robseqprior_onlybest_acc["Accuracy"].values == np.max(robseqprior_onlybest_acc["Accuracy"].values))
robseqprior_bestidx_balacc = np.where(robseqprior_onlybest_balacc["Balanced Accuracy"].values == np.max(robseqprior_onlybest_balacc["Balanced Accuracy"].values))
robseqprior_aggr_wide_acc = robseqprior_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
robseqprior_aggr_wide_balacc = robseqprior_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
robseqprior_bestparams_acc = robseqprior_onlybest_acc.iloc[robseqprior_bestidx_acc[0]]
robseqprior_bestparams_balacc = robseqprior_onlybest_balacc.iloc[robseqprior_bestidx_balacc[0]]
print("Both priors, accuracy:")
print(robseqprior_bestparams_acc)
print(robseqprior_bestparams_acc.values)
print("Both priors, balanced accuracy:")
print(robseqprior_bestparams_balacc)
print(robseqprior_bestparams_balacc.values)
print('------')

ageshape_perparam = ageshape.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
ageshape_perparam_sorted_acc = ageshape_perparam.reset_index().sort_values(by="Accuracy")
ageshape_perparam_sorted_balacc = ageshape_perparam.reset_index().sort_values(by="Balanced Accuracy")
ageshape_onlybest_acc = ageshape_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
ageshape_onlybest_balacc = ageshape_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
ageshape_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
ageshape_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
ageshape_bestidx_acc = np.where(ageshape_onlybest_acc["Accuracy"].values == np.max(ageshape_onlybest_acc["Accuracy"].values))
ageshape_bestidx_balacc = np.where(ageshape_onlybest_balacc["Balanced Accuracy"].values == np.max(ageshape_onlybest_balacc["Balanced Accuracy"].values))
ageshape_aggr_wide_acc = ageshape_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
ageshape_aggr_wide_balacc = ageshape_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
ageshape_bestparams_acc = ageshape_onlybest_acc.iloc[ageshape_bestidx_acc[0]]
ageshape_bestparams_balacc = ageshape_onlybest_balacc.iloc[ageshape_bestidx_balacc[0]]
print("Age, shape and position, accuracy:")
print(ageshape_bestparams_acc)
print(ageshape_bestparams_acc.values)
print("Age, shape and position, balanced accuracy:")
print(ageshape_bestparams_balacc)
print(ageshape_bestparams_balacc.values)
print('------')


# save to csv for inspection
nopriors_aggr_wide_acc.to_csv(os.path.join(outpath, "nopriors_acc_regression2.csv"))
nopriors_aggr_wide_balacc.to_csv(os.path.join(outpath, "nopriors_balacc_regression2.csv"))
seqprior_aggr_wide_acc.to_csv(os.path.join(outpath, "seqprior_acc_regression2.csv"))
seqprior_aggr_wide_balacc.to_csv(os.path.join(outpath, "seqprior_balacc_regression2.csv"))
robprior_aggr_wide_acc.to_csv(os.path.join(outpath, "robprior_acc_regression2.csv"))
robprior_aggr_wide_balacc.to_csv(os.path.join(outpath, "robprior_balacc_regression2.csv"))
robseqprior_aggr_wide_acc.to_csv(os.path.join(outpath, "robseqprior_acc_regression2.csv"))
robseqprior_aggr_wide_balacc.to_csv(os.path.join(outpath, "robseqprior_balacc_regression2.csv"))
ageshape_aggr_wide_acc.to_csv(os.path.join(outpath, "ageshape_acc_regression2.csv"))
ageshape_aggr_wide_balacc.to_csv(os.path.join(outpath, "ageshape_balacc_regression2.csv"))

allvalues_acc = np.concatenate((nopriors_aggr_wide_acc.values, seqprior_aggr_wide_acc.values, robprior_aggr_wide_acc.values, robseqprior_aggr_wide_acc.values, ageshape_aggr_wide_acc), axis=None)
allvalues_balacc = np.concatenate((nopriors_aggr_wide_balacc.values, seqprior_aggr_wide_balacc.values, robprior_aggr_wide_balacc.values, robseqprior_aggr_wide_balacc.values,ageshape_aggr_wide_balacc), axis=None)
overallmin = np.min([allvalues_acc, allvalues_balacc])
overallmax = np.max([allvalues_acc, allvalues_balacc])

fig = plt.figure()
ax = sns.heatmap(nopriors_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, linewidths=.5, annot_kws={"size": 8})
plt.title('No priors: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_noprior_acc_regression2.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(nopriors_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, linewidths=.5, annot_kws={"size": 8})
plt.title('No priors: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_noprior_balacc_regression2.png"), bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = sns.heatmap(seqprior_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Sequence prior: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_seq_acc_regression2.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(seqprior_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Sequence prior: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_seq_balacc_regression2.png"), bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = sns.heatmap(robprior_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Robustness prior: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_rob_acc_regression2.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(robprior_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Robustness prior: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_rob_balacc_regression2.png"), bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = sns.heatmap(robseqprior_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Both priors: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_robseq_acc_regression2.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(robseqprior_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Both priors: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_robseq_balacc_regression2.png"), bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = sns.heatmap(ageshape_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Age, shape and position: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_ageshape_acc_regression2.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(ageshape_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Age, shape and position: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_ageshape_balacc_regression2.png"), bbox_inches="tight")
plt.show()

# # load data for classification using only age as a feature
# agedata = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_age_accnew.csv")

# age_perparam = agedata.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
# age_perparam_sorted_acc = age_perparam.reset_index().sort_values(by="Accuracy")
# age_perparam_sorted_balacc = age_perparam.reset_index().sort_values(by="Balanced Accuracy")
# age_onlybest_acc = age_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
# age_onlybest_balacc = age_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
# age_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
# age_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
# age_bestidx_acc = np.where(age_onlybest_acc["Accuracy"].values == np.max(age_onlybest_acc["Accuracy"].values))
# age_bestidx_balacc = np.where(age_onlybest_balacc["Balanced Accuracy"].values == np.max(age_onlybest_balacc["Balanced Accuracy"].values))
# age_aggr_wide_acc = age_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
# age_aggr_wide_balacc = age_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
# age_bestparams_acc = age_onlybest_acc.iloc[age_bestidx_acc[0]]
# age_bestparams_balacc = age_onlybest_balacc.iloc[age_bestidx_balacc[0]]
# print("Age, accuracy:")
# print(age_bestparams_acc)
# print(age_bestparams_acc.values)
# print("Age, balanced accuracy:")
# print(age_bestparams_balacc)
# print(age_bestparams_balacc.values)
# print('------')

# get best ML method (highest mean bal. acc. across splits)
# aggregate best models for each experiment
nopriors_bestvals_acc = np.squeeze(nopriors_bestparams_acc.values)
nopriors_bestvals_balacc = np.squeeze(nopriors_bestparams_balacc.values)
seq_bestvals_balacc = np.squeeze(seqprior_bestparams_balacc.values)
rob_bestvals_balacc = np.squeeze(robprior_bestparams_balacc.values)
robseq_bestvals_balacc = np.squeeze(robseqprior_bestparams_balacc.values)
# age_bestvals_balacc = np.squeeze(age_bestparams_balacc.values)
ageshape_bestvals_balacc = np.squeeze(ageshape_bestparams_balacc.values)

# age_bestcombdata_balacc = bestselectsplits(agedata, age_bestvals_balacc, "Age")
noprior_bestcombdata_balacc = bestselectsplits(nopriors, nopriors_bestvals_balacc, "No prior")
seqrior_bestcombdata_balacc = bestselectsplits(seqprior, seq_bestvals_balacc, "Sequence prior")
robprior_bestcombdata_balacc = bestselectsplits(robprior, rob_bestvals_balacc, "Robustness prior")
robseqprior_bestcombdata_balacc = bestselectsplits(robseqprior, robseq_bestvals_balacc, "Both priors")
ageshape_bestcombdata_balacc = bestselectsplits(ageshape, ageshape_bestvals_balacc, "Age, shape and position")

allbest = pd.concat((noprior_bestcombdata_balacc, seqrior_bestcombdata_balacc,
                     robprior_bestcombdata_balacc, robseqprior_bestcombdata_balacc, ageshape_bestcombdata_balacc),
                    ignore_index=True)

fig = plt.figure(figsize=(5, 5))
ax = sns.boxplot(x="Model", y="Accuracy", data=allbest)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    horizontalalignment='center',
    # fontweight='normal',
    fontsize='medium',
    ha='right'
)
plt.ylim([0, 1])
plt.title("Regression Accuracy, 10-fold cross-validation", fontdict={"fontweight": "bold"})
plt.tight_layout()
plt.savefig(os.path.join(outpath, "boxplot_modelcomparison_acc_regression2.png"))
plt.show()

# aggregate mean and std for all method
meandf = allbest.drop(columns="Split").groupby(by="Model").mean().rename(columns={"Accuracy": "mean acc.", "Balanced Accuracy": "mean balacc."})
stddf = allbest.drop(columns="Split").groupby(by="Model").std().rename(columns={"Accuracy": "acc. std", "Balanced Accuracy": "std balacc."})

overallmeanstd = meandf.merge(stddf, on="Model")
overallmeanstd.to_csv(os.path.join(outpath, "modelcv_meanstd_acc_regression2.csv"))

fig = plt.figure(figsize=(5, 5))
ax = sns.boxplot(x="Model", y="Balanced Accuracy", data=allbest)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    horizontalalignment='center',
    # fontweight='normal',
    fontsize='medium',
    ha='right'
)
plt.ylim([0, 1])
plt.title("Regression Balanced Accuracy, 10-fold cross-validation", fontdict={"fontweight": "bold"})
plt.tight_layout()
plt.savefig(os.path.join(outpath, "boxplot_modelcomparison_balacc_regression2.png"))
plt.show()
