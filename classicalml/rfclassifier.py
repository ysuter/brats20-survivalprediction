
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from tqdm import tqdm
from matplotlib import pyplot as plt
from skfeature.function.information_theoretical_based import CIFE, JMI, DISR, MIM, CMIM, ICAP, MRMR, MIFS
from sklearn.neural_network import MLPClassifier


# load features
trainfeat = pd.read_csv("/media/yannick/MANAGE/BraTS20/trainfeat_all.csv")

# # eliminate features with a lower C-index than age and MAD zero
# madtest = trainfeat.mad()
# madpassed = madtest[madtest != 0]
#
# madpassed.to_csv('/media/yannick/MANAGE/BraTS20/train_madpassed_names.csv')
#
# madpasseddf = trainfeat[trainfeat.columns.intersection(madpassed.index.values)]
# madpasseddf.to_csv('/media/yannick/MANAGE/BraTS20/train_madpassed.csv')
#
cind_df = pd.DataFrame(np.ones([trainfeat.shape[1], 2])*np.NaN, columns=["Feature", "ConcordanceIndex"])
cind_df["Feature"] = trainfeat.columns
cind_df.set_index(keys="Feature", inplace=True)

cph = CoxPHFitter()

for col in tqdm(trainfeat.drop(columns=["Survival_days", "ID", "Extent_of_Resection", "Survival_class"]).columns.values):
# for col in featprocess_nosurv:
    print(col)
    indexname = col
    currfeat = trainfeat[[col, "Survival_days"]]

    try:
        cph.fit(currfeat, duration_col='Survival_days', show_progress=False, step_size=0.1)
        ci = cph.concordance_index_
        cind_df.loc[col] = ci
    except:
        cind_df.loc[col] = 0

cind_df.to_csv('/media/yannick/MANAGE/BraTS20/trainfeat_cind.csv')

exit()
# load filtered features
# madpasseddf = pd.read_csv('/media/yannick/MANAGE/BraTS20/train_madpassed.csv')

cind_df = pd.read_csv('/media/yannick/MANAGE/BraTS20/trainfeat_cind.csv')
#
# # drop shape features with other sequences than T1c (since they are the same)
# shapefeat = [elem for elem in cind_df["Feature"] if "shape" in elem]
# nont1cshape = [elem for elem in shapefeat if "T1c" not in elem]
#
# cind_df.set_index("Feature", inplace=True)
# cind_df.drop(index=nont1cshape, inplace=True)
#
#
# # keep only features with c-index > threshold
# cind_sel = cind_df.loc[cind_df["CondordanceIndex"] > 0.6]  # Age,0.6490229350265477
#
# cind_sel.to_csv('/media/yannick/MANAGE/BraTS20/trainfeat_cindex_0_6.csv', sep=";")
#
# keepcols_cind = list(cind_sel.index.values) + ["ID", "Survival_days"]
#
# cind_filtered = trainfeat.loc[:, keepcols_cind]
# cind_filtered.to_csv('/media/yannick/MANAGE/BraTS20/trainfeat_cind06_selected.csv', sep=";")

cind_filtered = pd.read_csv('/media/yannick/MANAGE/BraTS20/trainfeat_cind06_selected.csv', sep=";")

X_inp = cind_filtered.drop(columns=["ID", "Survival_days"])
y_inp = cind_filtered["Survival_days"]

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# scoring = ['accuracy', 'balanced_accuracy', 'average_precision', 'recall', 'f1', 'roc_auc']
scoring = ['accuracy', 'balanced_accuracy']

n_cvruns = 50
resultsdat = np.zeros((n_cvruns, 3+1))
resultsdf_train = pd.DataFrame(data=resultsdat, columns=["Run", "STS", "MTS", "LTS"])
resultsdf_train["Run"] = ["run_" + str(elem) for elem in np.arange(n_cvruns)]
resultsdf_train.set_index("Run", inplace=True)

resultsdf_train_acc = resultsdf_train.copy(deep=True)
resultsdf_train_balacc = resultsdf_train.copy(deep=True)
resultsdf_test_acc = resultsdf_train.copy(deep=True)
resultsdf_test_balacc = resultsdf_train.copy(deep=True)

for run in tqdm(np.arange(n_cvruns)):
    scores = cross_validate(clf, X_inp, y_inp, scoring=scoring, cv=3, return_train_score=True)

    resultsdf_test_acc.iloc[run, :] = scores["test_accuracy"]
    resultsdf_test_balacc.iloc[run, :] = scores["test_balanced_accuracy"]

    resultsdf_train_acc.iloc[run, :] = scores["train_accuracy"]
    resultsdf_train_balacc.iloc[run, :] = scores["train_balanced_accuracy"]

# plotting
resultsdf_test_acc.boxplot()
plt.ylim([0, 1])
plt.title("Random Forest, Test accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/cing_test_acc.png")
plt.show()

resultsdf_test_balacc.boxplot()
plt.ylim([0, 1])
plt.title("Random Forest, Test balanced accuracy")
plt.xlabel("Class")
plt.ylabel("Balanced accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/cing_test_balacc.png")
plt.show()


clf = SVC(kernel='rbf')
# scoring = ['accuracy', 'balanced_accuracy', 'average_precision', 'recall', 'f1', 'roc_auc']
scoring = ['accuracy', 'balanced_accuracy']

n_cvruns = 50
resultsdat = np.zeros((n_cvruns, 3+1))
resultsdf_train = pd.DataFrame(data=resultsdat, columns=["Run", "STS", "MTS", "LTS"])
resultsdf_train["Run"] = ["run_" + str(elem) for elem in np.arange(n_cvruns)]
resultsdf_train.set_index("Run", inplace=True)

resultsdf_train_acc = resultsdf_train.copy(deep=True)
resultsdf_train_balacc = resultsdf_train.copy(deep=True)
resultsdf_test_acc = resultsdf_train.copy(deep=True)
resultsdf_test_balacc = resultsdf_train.copy(deep=True)

for run in tqdm(np.arange(n_cvruns)):
    scores = cross_validate(clf, X_inp, y_inp, scoring=scoring, cv=3, return_train_score=True)

    resultsdf_test_acc.iloc[run, :] = scores["test_accuracy"]
    resultsdf_test_balacc.iloc[run, :] = scores["test_balanced_accuracy"]

    resultsdf_train_acc.iloc[run, :] = scores["train_accuracy"]
    resultsdf_train_balacc.iloc[run, :] = scores["train_balanced_accuracy"]

resultsdf_test_acc.boxplot()
plt.ylim([0, 1])
plt.title("SVC, Test accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/svc_cing_test_acc.png")
plt.show()

resultsdf_test_balacc.boxplot()
plt.ylim([0, 1])
plt.title("SVC, Test balanced accuracy")
plt.xlabel("Class")
plt.ylabel("Balanced accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/svc_cing_test_balacc.png")
plt.show()

# CIFE
X_inp = cind_filtered.drop(columns=["ID", "Survival_days"])
y_inp = cind_filtered["Survival_days"]

# normalize
# X_inp_norm =

selidx, selscore, _ = CIFE.cife(X_inp.values, y_inp.values, n_selected_features=5)
selidx_mifs, selscore_mifs, _ = MIFS.mifs(X_inp.values, y_inp.values, n_selected_features=5)

X_cife5 = X_inp.iloc[:, selidx]
X_mifs5 = X_inp.iloc[:, selidx]

clf = SVC(kernel='rbf')
# scoring = ['accuracy', 'balanced_accuracy', 'average_precision', 'recall', 'f1', 'roc_auc']
scoring = ['accuracy', 'balanced_accuracy']

n_cvruns = 50
resultsdat = np.zeros((n_cvruns, 3+1))
resultsdf_train = pd.DataFrame(data=resultsdat, columns=["Run", "STS", "MTS", "LTS"])
resultsdf_train["Run"] = ["run_" + str(elem) for elem in np.arange(n_cvruns)]
resultsdf_train.set_index("Run", inplace=True)

resultsdf_train_acc = resultsdf_train.copy(deep=True)
resultsdf_train_balacc = resultsdf_train.copy(deep=True)
resultsdf_test_acc = resultsdf_train.copy(deep=True)
resultsdf_test_balacc = resultsdf_train.copy(deep=True)

for run in tqdm(np.arange(n_cvruns)):
    scores = cross_validate(clf, X_cife5, y_inp, scoring=scoring, cv=3, return_train_score=True)

    resultsdf_test_acc.iloc[run, :] = scores["test_accuracy"]
    resultsdf_test_balacc.iloc[run, :] = scores["test_balanced_accuracy"]

    resultsdf_train_acc.iloc[run, :] = scores["train_accuracy"]
    resultsdf_train_balacc.iloc[run, :] = scores["train_balanced_accuracy"]

resultsdf_test_acc.boxplot()
plt.ylim([0, 1])
plt.title("CIFE, SVC, Test accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/svc__cife_cing_test_acc.png")
plt.show()

resultsdf_test_balacc.boxplot()
plt.ylim([0, 1])
plt.title("CIFE, SVC, Test balanced accuracy")
plt.xlabel("Class")
plt.ylabel("Balanced accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/svc_cife_cing_test_balacc.png")
plt.show()


clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# scoring = ['accuracy', 'balanced_accuracy', 'average_precision', 'recall', 'f1', 'roc_auc']
scoring = ['accuracy', 'balanced_accuracy']

n_cvruns = 50
resultsdat = np.zeros((n_cvruns, 3+1))
resultsdf_train = pd.DataFrame(data=resultsdat, columns=["Run", "STS", "MTS", "LTS"])
resultsdf_train["Run"] = ["run_" + str(elem) for elem in np.arange(n_cvruns)]
resultsdf_train.set_index("Run", inplace=True)

resultsdf_train_acc = resultsdf_train.copy(deep=True)
resultsdf_train_balacc = resultsdf_train.copy(deep=True)
resultsdf_test_acc = resultsdf_train.copy(deep=True)
resultsdf_test_balacc = resultsdf_train.copy(deep=True)

for run in tqdm(np.arange(n_cvruns)):
    scores = cross_validate(clf, X_cife5, y_inp, scoring=scoring, cv=3, return_train_score=True)

    resultsdf_test_acc.iloc[run, :] = scores["test_accuracy"]
    resultsdf_test_balacc.iloc[run, :] = scores["test_balanced_accuracy"]

    resultsdf_train_acc.iloc[run, :] = scores["train_accuracy"]
    resultsdf_train_balacc.iloc[run, :] = scores["train_balanced_accuracy"]

# plotting
resultsdf_test_acc.boxplot()
plt.ylim([0, 1])
plt.title("CIFE, Random Forest, Test accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/cife_rf_cing_test_acc.png")
plt.show()

resultsdf_test_balacc.boxplot()
plt.ylim([0, 1])
plt.title("CIFE, Random Forest, Test balanced accuracy")
plt.xlabel("Class")
plt.ylabel("Balanced accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/cife_rf_cing_test_balacc.png")
plt.show()

# MIFS with AGE

clf = SVC(kernel='rbf')
# scoring = ['accuracy', 'balanced_accuracy', 'average_precision', 'recall', 'f1', 'roc_auc']
scoring = ['accuracy', 'balanced_accuracy']

n_cvruns = 50
resultsdat = np.zeros((n_cvruns, 3+1))
resultsdf_train = pd.DataFrame(data=resultsdat, columns=["Run", "STS", "MTS", "LTS"])
resultsdf_train["Run"] = ["run_" + str(elem) for elem in np.arange(n_cvruns)]
resultsdf_train.set_index("Run", inplace=True)

resultsdf_train_acc = resultsdf_train.copy(deep=True)
resultsdf_train_balacc = resultsdf_train.copy(deep=True)
resultsdf_test_acc = resultsdf_train.copy(deep=True)
resultsdf_test_balacc = resultsdf_train.copy(deep=True)

for run in tqdm(np.arange(n_cvruns)):
    scores = cross_validate(clf, X_mifs5, y_inp, scoring=scoring, cv=3, return_train_score=True)

    resultsdf_test_acc.iloc[run, :] = scores["test_accuracy"]
    resultsdf_test_balacc.iloc[run, :] = scores["test_balanced_accuracy"]

    resultsdf_train_acc.iloc[run, :] = scores["train_accuracy"]
    resultsdf_train_balacc.iloc[run, :] = scores["train_balanced_accuracy"]

resultsdf_test_acc.boxplot()
plt.ylim([0, 1])
plt.title("MIFS, SVC, Test accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/svc_mifs_cing_test_acc.png")
plt.show()

resultsdf_test_balacc.boxplot()
plt.ylim([0, 1])
plt.title("MIFS, SVC, Test balanced accuracy")
plt.xlabel("Class")
plt.ylabel("Balanced accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/svc_mifs_cing_test_balacc.png")
plt.show()


clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# scoring = ['accuracy', 'balanced_accuracy', 'average_precision', 'recall', 'f1', 'roc_auc']
scoring = ['accuracy', 'balanced_accuracy']

n_cvruns = 50
resultsdat = np.zeros((n_cvruns, 3+1))
resultsdf_train = pd.DataFrame(data=resultsdat, columns=["Run", "STS", "MTS", "LTS"])
resultsdf_train["Run"] = ["run_" + str(elem) for elem in np.arange(n_cvruns)]
resultsdf_train.set_index("Run", inplace=True)

resultsdf_train_acc = resultsdf_train.copy(deep=True)
resultsdf_train_balacc = resultsdf_train.copy(deep=True)
resultsdf_test_acc = resultsdf_train.copy(deep=True)
resultsdf_test_balacc = resultsdf_train.copy(deep=True)

for run in tqdm(np.arange(n_cvruns)):
    scores = cross_validate(clf, X_mifs5, y_inp, scoring=scoring, cv=3, return_train_score=True)

    resultsdf_test_acc.iloc[run, :] = scores["test_accuracy"]
    resultsdf_test_balacc.iloc[run, :] = scores["test_balanced_accuracy"]

    resultsdf_train_acc.iloc[run, :] = scores["train_accuracy"]
    resultsdf_train_balacc.iloc[run, :] = scores["train_balanced_accuracy"]

# plotting
resultsdf_test_acc.boxplot()
plt.ylim([0, 1])
plt.title("MIFS, Random Forest, Test accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/mifs_rf_cing_test_acc.png")
plt.show()

resultsdf_test_balacc.boxplot()
plt.ylim([0, 1])
plt.title("MIFS, Random Forest, Test balanced accuracy")
plt.xlabel("Class")
plt.ylabel("Balanced accuracy")
plt.savefig("/media/yannick/MANAGE/BraTS20/results/mifs_rf_cing_test_balacc.png")
plt.show()