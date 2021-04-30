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
nopriors = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_accnew_stdnorm.csv")

# results with sequence prior
seqprior = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_acc_seqnew_stdnorm.csv")

# results with robustness prior
robprior = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_acc_robnew_stdnorm.csv")

# results with both priors
robseqprior = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_acc_robseq_stdnorm.csv")

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

# robseqprior_bestcombdata.set_index("Split")

# save to csv for inspection
nopriors_aggr_wide_acc.to_csv(os.path.join(outpath, "nopriors_acc_stdnorm.csv"))
nopriors_aggr_wide_balacc.to_csv(os.path.join(outpath, "nopriors_balacc_stdnorm.csv"))
seqprior_aggr_wide_acc.to_csv(os.path.join(outpath, "seqprior_acc_stdnorm.csv"))
seqprior_aggr_wide_balacc.to_csv(os.path.join(outpath, "seqprior_balacc_stdnorm.csv"))
robprior_aggr_wide_acc.to_csv(os.path.join(outpath, "robprior_acc_stdnorm.csv"))
robprior_aggr_wide_balacc.to_csv(os.path.join(outpath, "robprior_balacc_stdnorm.csv"))
robseqprior_aggr_wide_acc.to_csv(os.path.join(outpath, "robseqprior_acc_stdnorm.csv"))
robseqprior_aggr_wide_balacc.to_csv(os.path.join(outpath, "robseqprior_balacc_stdnorm.csv"))

allvalues_acc = np.concatenate((nopriors_aggr_wide_acc.values, seqprior_aggr_wide_acc.values, robprior_aggr_wide_acc.values, robseqprior_aggr_wide_acc.values), axis=None)
allvalues_balacc = np.concatenate((nopriors_aggr_wide_balacc.values, seqprior_aggr_wide_balacc.values, robprior_aggr_wide_balacc.values, robseqprior_aggr_wide_balacc.values), axis=None)
overallmin = np.min([allvalues_acc, allvalues_balacc])
overallmax = np.max([allvalues_acc, allvalues_balacc])

fig = plt.figure()
ax = sns.heatmap(nopriors_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, linewidths=.5, annot_kws={"size": 8})
plt.title('No priors: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_noprior_acc_stdnorm.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(nopriors_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, linewidths=.5, annot_kws={"size": 8})
plt.title('No priors: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_noprior_balacc_stdnorm.png"), bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = sns.heatmap(seqprior_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Sequence prior: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_seq_acc_stdnorm.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(seqprior_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Sequence prior: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_seq_balacc_stdnorm.png"), bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = sns.heatmap(robprior_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Robustness prior: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_rob_acc_stdnorm.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(robprior_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Robustness prior: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_rob_balacc_stdnorm.png"), bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = sns.heatmap(robseqprior_aggr_wide_acc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Both priors: Mean accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_robseq_acc_stdnorm.png"), bbox_inches="tight")
plt.show()
fig = plt.figure()
ax = sns.heatmap(robseqprior_aggr_wide_balacc, annot=True, fmt=".2f", vmin=overallmin, vmax=overallmax, robust=True, linewidths=.5, annot_kws={"size": 8})
plt.title('Both priors: Mean balanced accuracy, 10-fold cross-validation', fontdict={'weight': "bold"})
plt.savefig(os.path.join(outpath, "heatmap_robseq_balacc_stdnorm.png"), bbox_inches="tight")
plt.show()

# load data for classification using only age as a feature
agedata = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_age_accnew.csv")
# agedata_aggr = agedata.drop(columns=["Split", "Parameter1", "Parameter2", "Feature selector"]).groupby(by=["ML method"]).mean()
# bestageidx = np.argmax(agedata_aggr.values)
# print("Age classification")
# print("Best model: " + agedata_aggr.index[bestageidx])
# print(agedata_aggr.values[bestageidx])
# age_bestcombdata = agedata.loc[agedata["ML method"] == agedata_aggr.index[bestageidx]]
# age_bestcombdata.drop(columns=["Parameter1", "Parameter2", "ML method", "Feature selector"], inplace=True)
# age_bestcombdata.loc[:, "Model"] = ["Age"] * age_bestcombdata.shape[0]
# age_bestcombdata.set_index("Split")

age_perparam = agedata.drop(columns=["Split"]).groupby(by=["Feature selector", "ML method", "Parameter1", "Parameter2"], dropna=False).mean()
age_perparam_sorted_acc = age_perparam.reset_index().sort_values(by="Accuracy")
age_perparam_sorted_balacc = age_perparam.reset_index().sort_values(by="Balanced Accuracy")
age_onlybest_acc = age_perparam_sorted_acc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
age_onlybest_balacc = age_perparam_sorted_balacc.drop_duplicates(keep='last', subset=["ML method", "Feature selector"])
age_onlybest_acc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
age_onlybest_balacc.set_index(["ML method", "Feature selector", "Parameter1", "Parameter2"])
age_bestidx_acc = np.where(age_onlybest_acc["Accuracy"].values == np.max(age_onlybest_acc["Accuracy"].values))
age_bestidx_balacc = np.where(age_onlybest_balacc["Balanced Accuracy"].values == np.max(age_onlybest_balacc["Balanced Accuracy"].values))
age_aggr_wide_acc = age_onlybest_acc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Accuracy", columns="Feature selector", dropna=False)
age_aggr_wide_balacc = age_onlybest_balacc.drop(columns=["Parameter1", "Parameter2"]).pivot_table(index="ML method", values="Balanced Accuracy", columns="Feature selector", dropna=False)
age_bestparams_acc = age_onlybest_acc.iloc[age_bestidx_acc[0]]
age_bestparams_balacc = age_onlybest_balacc.iloc[age_bestidx_balacc[0]]
print("Age, accuracy:")
print(age_bestparams_acc)
print(age_bestparams_acc.values)
print("Age, balanced accuracy:")
print(age_bestparams_balacc)
print(age_bestparams_balacc.values)
print('------')

# get best ML method (highest mean bal. acc. across splits)
# aggregate best models for each experiment
nopriors_bestvals_acc = np.squeeze(nopriors_bestparams_acc.values)
nopriors_bestvals_balacc = np.squeeze(nopriors_bestparams_balacc.values)
seq_bestvals_balacc = np.squeeze(seqprior_bestparams_balacc.values)
rob_bestvals_balacc = np.squeeze(robprior_bestparams_balacc.values)
robseq_bestvals_balacc = np.squeeze(robseqprior_bestparams_balacc.values)
age_bestvals_balacc = np.squeeze(age_bestparams_balacc.values)

age_bestcombdata_balacc = bestselectsplits(agedata, age_bestvals_balacc, "Age")
noprior_bestcombdata_balacc = bestselectsplits(nopriors, nopriors_bestvals_balacc, "No prior")
seqrior_bestcombdata_balacc = bestselectsplits(seqprior, seq_bestvals_balacc, "Sequence prior")
robprior_bestcombdata_balacc = bestselectsplits(robprior, rob_bestvals_balacc, "Robustness prior")
robseqprior_bestcombdata_balacc = bestselectsplits(robseqprior, robseq_bestvals_balacc, "Both priors")

allbest = pd.concat((age_bestcombdata_balacc, noprior_bestcombdata_balacc, seqrior_bestcombdata_balacc, robprior_bestcombdata_balacc, robseqprior_bestcombdata_balacc), ignore_index=True)

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
plt.title("Accuracy, 10-fold cross-validation", fontdict={"fontweight": "bold"})
plt.tight_layout()
plt.savefig(os.path.join(outpath, "boxplot_modelcomparison_acc_stdnorm.png"))
plt.show()

# aggregate mean and std for all method
meandf = allbest.drop(columns="Split").groupby(by="Model").mean().rename(columns={"Accuracy": "mean acc.", "Balanced Accuracy": "mean balacc."})
stddf = allbest.drop(columns="Split").groupby(by="Model").std().rename(columns={"Accuracy": "acc. std", "Balanced Accuracy": "std balacc."})

overallmeanstd = meandf.merge(stddf, on="Model")
overallmeanstd.to_csv(os.path.join(outpath, "modelcv_meanstd_acc_stdnorm.csv"))

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
plt.title("Balanced Accuracy, 10-fold cross-validation", fontdict={"fontweight": "bold"})
plt.tight_layout()
plt.savefig(os.path.join(outpath, "boxplot_modelcomparison_balacc_stdnorm.png"))
plt.show()
#
# # aggregate mean and std for all method
# meandf = allbest.drop(columns="Split").groupby(by="Model").mean().rename(columns={"Balanced Accuracy": "mean"})
# stddf = allbest.drop(columns="Split").groupby(by="Model").std().rename(columns={"Balanced Accuracy": "std"})
# #
# overallmeanstd = meandf.merge(stddf, on="Model")
# overallmeanstd.to_csv(os.path.join(outpath, "modelcv_meanstd_balacc_stdnorm.csv"))
