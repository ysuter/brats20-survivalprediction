#!/home/yannick/anaconda3/envs/py36/bin/python

import json
import numpy as np
import os
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier
import numpy.matlib


# from skfeature.function import similarity_based, information_theoretical_based, statistical_based


def survival_classencoding(survarr: np.array, classboundaries: list):
    if len(classboundaries) == 1:
        survival_classes = [0 if elem <= classboundaries[0] else 1 for elem in survarr]

    if len(classboundaries) == 2:
        survival_classes = [int(0) if elem <= classboundaries[0] else int(1) if elem <= classboundaries[1] else int(2)
                            for elem in
                            survarr]

    return np.array(survival_classes)


class_boundary = [304.2, 456.25]

features = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat.csv",
    index_col="ID")
splitinfopath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/splitinfo.json"

features_nosurv = features["Age"]
surv_days = features["Survival_days"]
surv_classes = survival_classencoding(surv_days, class_boundary)

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

# class boundary list
class_boundary = [304.2, 456.25]
numsplits = 10

# Dataframe for highest balanced accuracy for each feature selector / ML combination
balacc_df = pd.DataFrame(data=[], columns=["Feature selector", "ML method", "Split", "Parameter1", "Parameter2",
                                           "Accuracy"])
for split in np.arange(numsplits):
    print("Evaluating fold " + str(split))
    train_index = kfolds["fold_" + str(split)]["train"]
    test_index = kfolds["fold_" + str(split)]["test"]

    X_train, X_test = features_nosurv.iloc[train_index], features_nosurv.iloc[test_index]
    y_train, y_test = surv_classes[train_index], surv_classes[test_index]

    # get subsets for all number of features
    X_train_selected = X_train.values.reshape(-1, 1)
    X_test_selected = X_test.values.reshape(-1, 1)

    ##########################################
    # do classification with all classifiers #
    ##########################################
    best_param1 = np.NaN
    best_param2 = np.NaN
    best_balacc = np.NaN

    for clf_name, clf in zip(classifiernames, classifiers):
        print(clf_name)
        if clf_name is "XGBoost":
            best_param1 = np.NaN
            best_param2 = np.NaN
            best_balacc = np.NaN

            clf = XGBClassifier()
            clf.fit(X_train_selected, y_train)
            # score = clf.score(X_test_selected, y_test)
            if hasattr(clf, "decision_function"):
                score = clf.decision_function(X_test_selected)
            else:
                score = clf.predict_proba(X_test_selected)
            y_pred = clf.predict(X_test_selected)
            y_train_pred = clf.predict(X_train_selected)

            best_balacc = accuracy_score(y_test, y_pred)

        elif clf_name is "Nearest Neighbors":
            best_param1 = np.NaN
            best_param2 = np.NaN
            best_balacc = np.NaN

            numneighbors = np.arange(3, 22, 3)
            balacclist = np.array([])
            for num_n in tqdm(numneighbors):
                clf = KNeighborsClassifier(n_neighbors=num_n, n_jobs=3)
                # Xtraintmp = np.transpose(np.matlib.repmat(X_train_selected, 2, 1))
                clf.fit(X_train_selected, y_train)
                # if hasattr(clf, "decision_function"):
                #     score = clf.decision_function(X_test_selected)
                # else:
                #     score = clf.predict_proba(np.transpose(np.matlib.repmat(X_test_selected, 2, 1)))
                #     score = clf.predict_proba(np.transpose(np.matlib.repmat(X_test_selected, 2, 1)))
                #     # print(score)
                # y_pred = clf.predict(np.transpose(np.matlib.repmat(X_test_selected, 2, 1)))
                y_pred = clf.predict(X_test_selected)
                # y_train_pred = clf.predict(X_train_selected)

                # auc = roc_auc_score(y_test, y_pred)
                # print('Number of features: ' + str(numfeat) + ', ' + name + ': ' + str(auc))
                predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])

                curr_balacc = accuracy_score(y_test, y_pred)
                balacclist = np.append(balacclist, curr_balacc)
            best_balacc_idx = np.argmax(balacclist)
            best_balacc = balacclist[best_balacc_idx]
            best_param1 = numneighbors[best_balacc_idx]

        elif clf_name is "Linear SVC":
            best_param1 = np.NaN
            best_param2 = np.NaN
            best_balacc = np.NaN

            costparam = [0.25, 0.5, 1, 2, 4]
            balacclist = np.array([])
            for c in tqdm(costparam):
                clf = SVC(kernel="linear", C=c, verbose=False, max_iter=int(1e8))
                clf.fit(X_train_selected, y_train)
                # score = clf.score(X_test_selected, y_test)
                if hasattr(clf, "decision_function"):
                    score = clf.decision_function(X_test_selected)
                else:
                    score = clf.predict_proba(X_test_selected)
                y_pred = clf.predict(X_test_selected)
                y_train_pred = clf.predict(X_train_selected)
                predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])

                curr_balacc = accuracy_score(y_test, y_pred)
                balacclist = np.append(balacclist, curr_balacc)
            best_balacc_idx = np.argmax(balacclist)
            best_balacc = balacclist[best_balacc_idx]
            best_param1 = numneighbors[best_balacc_idx]


        elif clf_name is "RBF SVC":
            best_param1 = np.NaN
            best_param2 = np.NaN
            best_balacc = np.NaN

            costparam = [0.25, 0.5, 1, 2, 4]
            gamma = ['scale', 'auto', 0.01, 0.1, 1, 10, 100]

            balacclist = np.zeros([len(costparam), len(gamma)])

            for cidx, c in enumerate(costparam):
                for gidx, g in enumerate(gamma):
                    clf = SVC(gamma=g, C=c)
                    clf.fit(X_train_selected, y_train)
                    # score = clf.score(X_test_selected, y_test)
                    if hasattr(clf, "decision_function"):
                        score = clf.decision_function(X_test_selected)
                    else:
                        score = clf.predict_proba(X_test_selected)
                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

            best_balacc_idx = np.unravel_index(np.argmax(balacclist, axis=None), balacclist.shape)
            best_balacc = balacclist[best_balacc_idx]
            best_param1 = costparam[best_balacc_idx[0]]
            best_param2 = gamma[best_balacc_idx[1]]

        elif clf_name is "Decision Tree":
            best_param1 = np.NaN
            best_param2 = np.NaN
            best_balacc = np.NaN

            maxdepthlist = [5, 10, 15, 20]
            balacclist = np.array([])
            for d in tqdm(maxdepthlist):
                clf = DecisionTreeClassifier(max_depth=d)
                clf.fit(X_train_selected, y_train)
                # score = clf.score(X_test_selected, y_test)
                if hasattr(clf, "decision_function"):
                    score = clf.decision_function(X_test_selected)
                else:
                    score = clf.predict_proba(X_test_selected)
                y_pred = clf.predict(X_test_selected)
                y_train_pred = clf.predict(X_train_selected)

                curr_balacc = accuracy_score(y_test, y_pred)
                balacclist = np.append(balacclist, curr_balacc)
            best_balacc_idx = np.argmax(balacclist)
            best_balacc = balacclist[best_balacc_idx]
            best_param1 = maxdepthlist[best_balacc_idx]

        elif clf_name is "Multilayer Perceptron":
            best_param1 = np.NaN
            best_param2 = np.NaN
            best_balacc = np.NaN

            alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10]
            balacclist = np.array([])
            for a in tqdm(alpha):
                clf = MLPClassifier(alpha=a, max_iter=int(1e8))
                clf.fit(X_train_selected, y_train)
                # score = clf.score(X_test_selected, y_test)
                if hasattr(clf, "decision_function"):
                    score = clf.decision_function(X_test_selected)
                else:
                    score = clf.predict_proba(X_test_selected)
                y_pred = clf.predict(X_test_selected)
                y_train_pred = clf.predict(X_train_selected)

                predscoredf = pd.DataFrame(data=y_pred, columns=["Survival"])

                curr_balacc = accuracy_score(y_test, y_pred)
                balacclist = np.append(balacclist, curr_balacc)
            best_balacc_idx = np.argmax(balacclist)
            best_balacc = balacclist[best_balacc_idx]
            best_param1 = alpha[best_balacc_idx]


        else:
            best_param1 = np.NaN
            best_param2 = np.NaN
            best_balacc = np.NaN

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

            best_balacc = accuracy_score(y_test, y_pred)

        curr_restultsdict = {"Feature selector": "Age",
                             "ML method": clf_name,
                             "Split": split,
                             "Parameter1": best_param1,
                             "Parameter2": best_param2,
                             "Accuracy": best_balacc}

        balacc_df = balacc_df.append(curr_restultsdict, ignore_index=True)
        print(balacc_df)
        balacc_df.to_csv(
            "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_age_acc.csv",
            index=False)

balacc_df.to_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_age_acc.csv",
    index=False)
