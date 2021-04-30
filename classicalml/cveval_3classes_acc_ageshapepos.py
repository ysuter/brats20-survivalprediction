#!/home/yannick/anaconda3/envs/py36/bin/python

import json
import numpy as np
import os
import pandas as pd
from skfeature.function.information_theoretical_based import CIFE, JMI, DISR, MIM, CMIM, ICAP, MRMR, MIFS
from skfeature.function.similarity_based import reliefF, fisher_score
from skfeature.function.statistical_based import chi_square, gini_index
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from tqdm import tqdm


def survival_classencoding(survarr: np.array, classboundaries: list):
    if len(classboundaries) == 1:
        survival_classes = [0 if elem <= classboundaries[0] else 1 for elem in survarr]

    if len(classboundaries) == 2:
        survival_classes = [int(0) if elem <= classboundaries[0] else int(1) if elem <= classboundaries[1] else int(2) for elem in
                            survarr]

    return np.array(survival_classes)


class_boundary = [304.2, 456.25]

# features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat.csv", index_col="ID")
features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_nobinduplicates.csv", index_col="ID")
splitinfopath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/splitinfo.json"

# handle alive patient with setting OS to 500 days
features.loc[features["Survival_days"] == 'ALIVE (361 days later)', "Survival_days"] = 500
features["Survival_days"] = features["Survival_days"].astype(np.float)
features_nosurv = features.drop(columns="Survival_days", inplace=False)
surv_days = features["Survival_days"]
surv_classes = survival_classencoding(surv_days, class_boundary)

# only use features from the T1c and FLAIR MRIs
colselect = [elem for elem in features_nosurv.columns if (("shape" in elem) or ("Age" in elem) or ("mincet" in elem)
                                                          or ("ventrdist" in elem) or ("maxcet" in elem)
                                                          or ("maxed" in elem)or ("mined" in elem))]
dropcols = list(set(features_nosurv.columns) - set(colselect))
features_nosurv.drop(columns=dropcols, inplace=True)
np.random.seed(42)

# load split infos
with open(splitinfopath) as f:
    kfolds = json.load(f)

# for key, val in kfolds.items():
#     print(key)
#     train = val['train']
#     test = val['test']

numfeat = 9

classifiernames = ["Nearest Neighbors",
                       "Linear SVC",
                       "RBF SVC",
                       "Gaussian Process",
                       "Decision Tree",
                       "Random Forest",
                       "Multilayer Perceptron",
                       "AdaBoost",
                       "Naive Bayes",
                       "QDA",
                       "XGBoost",
                       "Logistic Regression"
                       ]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=100, max_features='auto'),
    MLPClassifier(alpha=1, max_iter=int(1e8)),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier,
    LogisticRegression()]

selectors = [
    reliefF.reliefF,
    fisher_score.fisher_score,
    gini_index.gini_index,
    chi_square.chi_square,
    JMI.jmi,
    CIFE.cife,
    DISR.disr,
    MIM.mim,
    CMIM.cmim,
    ICAP.icap,
    MRMR.mrmr,
    MIFS.mifs]

selectornames_short = ["RELF", "FSCR", "GINI", "CHSQ", "JMI", "CIFE", "DISR", "MIM", "CMIM", "ICAP", "MRMR", "MIFS"]

# class boundary list
class_boundary = [304.2, 456.25]
numsplits = 10


def writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc):
    curr_resultsdict = {"Feature selector": sel_name,
                        "ML method": clf_name,
                        "Split": split,
                        "Parameter1": param1,
                        "Parameter2": param2,
                        "Accuracy": acc,
                        "Balanced Accuracy": balacc,
                        }

    outdf = outdf.append(curr_resultsdict, ignore_index=True)
    print(outdf)
    outdf.to_csv(
        "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_ageshapepos_class.csv",
        index=False)

    return outdf

# Dataframe for highest balanced accuracy for each feature selector / ML combination
outdf = pd.DataFrame(data=[], columns=["Feature selector", "ML method", "Split", "Parameter1", "Parameter2", "Accuracy", "Balanced Accuracy"])

for split in np.arange(numsplits):
    print("Evaluating fold " + str(split))
    train_index = kfolds["fold_" + str(split)]["train"]
    test_index = kfolds["fold_" + str(split)]["test"]

    X_train, X_test = features_nosurv.iloc[train_index], features_nosurv.iloc[test_index]
    y_train, y_test = surv_classes[train_index], surv_classes[test_index]

    # X_train = features_nosurv.iloc[train_index]
    # X_test_df = features_nosurv.iloc[test_index]
    # y_train_df = surv_classes.iloc[train_index]
    # y_test_df = surv_classes.iloc[test_index]

    # for every split, perform feature selection
    for sel_name, sel in zip(selectornames_short, selectors):
        print('#####')
        print(sel_name)
        print('#####')

        if sel_name is "CHSQ":
            # shift X values to be non-negative for chsq feature selection
            X_train_tmp = X_train + np.abs(X_train.min())
            selscore = sel(X_train_tmp, y_train)
            selidx = np.argsort(selscore)[::-1]
            selidx = selidx[0:numfeat]
            selscore = selscore[selidx]
            selscoredf = pd.DataFrame(
                data=np.transpose(np.vstack((X_train.columns[selidx].values, selscore))),
                columns=['Feature', 'Score'])

        elif sel_name == "RELF":
            selscore = sel(X_train.values, y_train, k=numfeat)

            selidx = np.argsort(selscore)[::-1]
            # print(selidx)
            selidx = selidx[0:numfeat]
            selscoredf = pd.DataFrame(
                data=np.transpose(np.vstack((X_train.columns[selidx].values, selscore[selidx]))),
                columns=['Feature', 'Score'])

        elif sel_name == "JMI" or sel_name == "CIFE" or sel_name == "DISR" or sel_name == "MIM" \
                or sel_name == "CMIM" or sel_name == "ICAP" or sel_name == "MRMR" or sel_name == "MIFS":
            selidx, selscore, _ = sel(X_train.values, y_train, n_selected_features=numfeat)
            selscoredf = pd.DataFrame(
                data=np.transpose(np.vstack((X_train.columns[selidx].values, selscore))),
                columns=['Feature', 'Score'])

        else:
            selscore = sel(X_train.values, y_train)

            selidx = np.argsort(selscore)[::-1]
            # print(selidx)
            selidx = selidx[0:numfeat]
            selscoredf = pd.DataFrame(
                data=np.transpose(np.vstack((X_train.columns[selidx].values, selscore[selidx]))),
                columns=['Feature', 'Score'])

        # get subsets for all number of features
        X_train_selected = X_train.iloc[:, selidx[0:numfeat]]
        X_test_selected = X_test.iloc[:, selidx[0:numfeat]]

        ##########################################
        # do classification with all classifiers #
        ##########################################
        best_param1 = np.NaN
        best_param2 = np.NaN
        best_balacc = np.NaN

        for clf_name, clf in zip(classifiernames, classifiers):
            print(clf_name)
            if clf_name is "XGBoost":
                param1 = np.NaN
                param2 = np.NaN
                # best_balacc = np.NaN

                clf = XGBClassifier()
                clf.fit(X_train_selected, y_train)
                # score = clf.score(X_test_selected, y_test)
                if hasattr(clf, "decision_function"):
                    score = clf.decision_function(X_test_selected)
                else:
                    score = clf.predict_proba(X_test_selected)
                y_pred = clf.predict(X_test_selected)
                y_train_pred = clf.predict(X_train_selected)

                predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])
                predscoredf["Score"] = score.tolist()
                curr_balacc = balanced_accuracy_score(y_test, y_pred)
                curr_acc = accuracy_score(y_test, y_pred)

                outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, curr_acc, curr_balacc)

            elif clf_name is "Nearest Neighbors":
                param1 = np.NaN
                param2 = np.NaN
                # best_balacc = np.NaN

                numneighbors = np.arange(3, 22, 3)
                balacclist = np.array([])
                for param1 in tqdm(numneighbors):
                    clf = KNeighborsClassifier(n_neighbors=param1, n_jobs=3)
                    clf.fit(X_train_selected, y_train)
                    if hasattr(clf, "decision_function"):
                        score = clf.decision_function(X_test_selected)
                    else:
                        score = clf.predict_proba(X_test_selected)
                        # print(score)
                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    # auc = roc_auc_score(y_test, y_pred)
                    # print('Number of features: ' + str(numfeat) + ', ' + name + ': ' + str(auc))
                    predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])
                    predscoredf["Score"] = score.tolist()

                    curr_balacc = balanced_accuracy_score(y_test, y_pred)
                    curr_acc = accuracy_score(y_test, y_pred)

                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, curr_acc, curr_balacc)

            elif clf_name is "Linear SVC":
                param1 = np.NaN
                param2 = np.NaN
                # best_balacc = np.NaN

                costparam = [0.25, 0.5, 1, 2, 4]
                balacclist = np.array([])
                for param1 in tqdm(costparam):
                    clf = SVC(kernel="linear", C=param1, verbose=0, max_iter=int(1e8))
                    clf.fit(X_train_selected, y_train)
                    # score = clf.score(X_test_selected, y_test)
                    if hasattr(clf, "decision_function"):
                        score = clf.decision_function(X_test_selected)
                    else:
                        score = clf.predict_proba(X_test_selected)
                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)
                    predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])
                    predscoredf["Score"] = score.tolist()

                    curr_balacc = balanced_accuracy_score(y_test, y_pred)
                    curr_acc = accuracy_score(y_test, y_pred)

                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, curr_acc, curr_balacc)

                # best_balacc_idx = np.argmax(balacclist)
                # best_balacc = balacclist[best_balacc_idx]
                # best_param1 = numneighbors[best_balacc_idx]

            elif clf_name is "RBF SVC":
                param1 = np.NaN
                param2 = np.NaN

                costparam = [0.25, 0.5, 1, 2, 4]
                gamma = ['scale', 'auto', 0.01, 0.1, 1, 10, 100]

                balacclist = np.zeros([len(costparam), len(gamma)])

                for param1_idx, param1 in enumerate(costparam):
                    for gidx, param2 in enumerate(gamma):
                        clf = SVC(gamma=param2, C=param1)
                        clf.fit(X_train_selected, y_train)
                        # score = clf.score(X_test_selected, y_test)
                        if hasattr(clf, "decision_function"):
                            score = clf.decision_function(X_test_selected)
                        else:
                            score = clf.predict_proba(X_test_selected)
                        y_pred = clf.predict(X_test_selected)
                        y_train_pred = clf.predict(X_train_selected)
                        predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])
                        predscoredf["Score"] = score.tolist()

                        curr_balacc = balanced_accuracy_score(y_test, y_pred)
                        curr_acc = accuracy_score(y_test, y_pred)

                        outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, curr_acc, curr_balacc)
                # best_balacc_idx = np.unravel_index(np.argmax(balacclist, axis=None), balacclist.shape)
                # best_balacc = balacclist[best_balacc_idx]
                # best_param1 = costparam[best_balacc_idx[0]]
                # best_param2 = gamma[best_balacc_idx[1]]

            elif clf_name is "Decision Tree":
                # best_param1 = np.NaN
                # best_param2 = np.NaN
                # best_balacc = np.NaN
                param1 = np.NaN
                param2 = np.NaN

                maxdepthlist = [5, 10, 15, 20]
                balacclist = np.array([])
                for param1 in tqdm(maxdepthlist):
                    clf = DecisionTreeClassifier(max_depth=param1)
                    clf.fit(X_train_selected, y_train)
                    # score = clf.score(X_test_selected, y_test)
                    if hasattr(clf, "decision_function"):
                        score = clf.decision_function(X_test_selected)
                    else:
                        score = clf.predict_proba(X_test_selected)
                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])
                    predscoredf["Score"] = score.tolist()

                    # curr_balacc = accuracy_score(y_test, y_pred)
                    # balacclist = np.append(balacclist, curr_balacc)

                    curr_balacc = balanced_accuracy_score(y_test, y_pred)
                    curr_acc = accuracy_score(y_test, y_pred)

                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, curr_acc, curr_balacc)
                # best_balacc_idx = np.argmax(balacclist)
                # best_balacc = balacclist[best_balacc_idx]
                # best_param1 = maxdepthlist[best_balacc_idx]

            elif clf_name is "Multilayer Perceptron":
                param1 = np.NaN
                param2 = np.NaN
                # best_balacc = np.NaN

                alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10]
                balacclist = np.array([])
                for param1 in tqdm(alpha):
                    clf = MLPClassifier(alpha=param1, max_iter=int(1e8))
                    clf.fit(X_train_selected, y_train)
                    # score = clf.score(X_test_selected, y_test)
                    if hasattr(clf, "decision_function"):
                        score = clf.decision_function(X_test_selected)
                    else:
                        score = clf.predict_proba(X_test_selected)
                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])
                    predscoredf["Score"] = score.tolist()

                    # curr_balacc = accuracy_score(y_test, y_pred)
                    curr_balacc = balanced_accuracy_score(y_test, y_pred)
                    curr_acc = accuracy_score(y_test, y_pred)

                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, curr_acc, curr_balacc)
                #     balacclist = np.append(balacclist, curr_balacc)
                # best_balacc_idx = np.argmax(balacclist)
                # best_balacc = balacclist[best_balacc_idx]
                # best_param1 = alpha[best_balacc_idx]

            else:
                param1 = np.NaN
                param2 = np.NaN

                clf.fit(X_train_selected, y_train)
                # score = clf.score(X_test_selected, y_test)
                if hasattr(clf, "decision_function"):
                    score = clf.decision_function(X_test_selected)
                    # print(score)
                else:
                    score = clf.predict_proba(X_test_selected)
                y_pred = clf.predict(X_test_selected)
                y_train_pred = clf.predict(X_train_selected)

                predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])
                predscoredf["Score"] = score.tolist()

                curr_balacc = balanced_accuracy_score(y_test, y_pred)
                curr_acc = accuracy_score(y_test, y_pred)

                outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, curr_acc, curr_balacc)

            # curr_resultsdict = {"Feature selector": sel_name,
            #                      "ML method": clf_name,
            #                      "Split": split,
            #                      "Parameter1": best_param1,
            #                      "Parameter2": best_param2,
            #                      "Accuracy": best_balacc}
            #
            # balacc_df = balacc_df.append(curr_resultsdict, ignore_index=True)
            # print(balacc_df)
            # balacc_df.to_csv(
            #     "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_acc.csv",
            #     index=False)

outdf.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_ageshapepos_class.csv", index=False)
