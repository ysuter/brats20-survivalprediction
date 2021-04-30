#!/home/yannick/anaconda3/envs/py36/bin/python


import json
import numpy as np
import os
import pandas as pd
from skfeature.function.information_theoretical_based import CIFE, JMI, DISR, MIM, CMIM, ICAP, MRMR, MIFS
from skfeature.function.similarity_based import reliefF, fisher_score
from skfeature.function.statistical_based import chi_square, gini_index

from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model.bayes import ARDRegression
from sklearn.linear_model.huber import HuberRegressor
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.passive_aggressive import  PassiveAggressiveRegressor
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.linear_model.theil_sen import TheilSenRegressor
from sklearn.linear_model.ransac import RANSACRegressor
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.neighbors.regression import RadiusNeighborsRegressor
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.tree.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm.classes import SVR

from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from scipy.stats import spearmanr   # spearmanr(currttpdata_bratumia["TTP"].values, currttpdata_bratumia["OS"].values, nan_policy='omit')

from tqdm import tqdm


def survival_classencoding(survarr: np.array, classboundaries: list):
    if len(classboundaries) == 1:
        survival_classes = [0 if elem <= classboundaries[0] else 1 for elem in survarr]

    if len(classboundaries) == 2:
        survival_classes = [int(0) if elem <= classboundaries[0] else int(1) if elem <= classboundaries[1] else int(2) for elem in
                            survarr]

    return np.array(survival_classes)


def writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho):
    curr_resultsdict = {"Feature selector": sel_name,
                        "ML method": clf_name,
                        "Split": split,
                        "Parameter1": param1,
                        "Parameter2": param2,
                        "Accuracy": acc,
                        "Balanced Accuracy": balacc,
                        "MSE": mse,
                        "r2": r2,
                        "spearmanr": rho
                        }

    outdf = outdf.append(curr_resultsdict, ignore_index=True)
    print(outdf)
    outdf.to_csv(
        "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_ageshape_regression2.csv",
        index=False)

    return outdf


def gradeoutput(y_test, y_pred, class_boundary):

    y_test_classes = survival_classencoding(y_test, class_boundary)
    y_pred_classes = survival_classencoding(y_pred, class_boundary)
    acc = accuracy_score(y_test_classes, y_pred_classes)
    balacc = balanced_accuracy_score(y_test_classes, y_pred_classes)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rho, _ = spearmanr(y_test, y_pred, nan_policy='omit')

    return [balacc, acc, mse, r2, rho]


class_boundary = [304.2, 456.25]

# features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat.csv", index_col="ID")
features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat2.csv", index_col="ID")
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

numfeat = 9

randomstate = 42
classifiernames = ["Random Forest",
                   "Extra Trees",
                   # "Hist. Gradient Boosting",
                   "AdaBoost",
                   "Gaussian Process",
                   "ARD",
                   # "Huber",
                   "Linear",
                   "Passive Aggressive",
                   "SGD",
                   "Theil-Sen",
                   "RANSAC",
                   "K-Neighbors",
                   "Radius Neighbors",
                   "MLP",
                   "Decision Tree",
                   "Extra Tree",
                   "SVR"
                   ]

classifiers = [
    RandomForestRegressor(n_estimators=200, n_jobs=5, random_state=randomstate),
    ExtraTreesRegressor(n_estimators=200, n_jobs=5, random_state=randomstate),
    # GradientBoostingRegressor(random_state=randomstate),    # learning_rate is a hyper-parameter in the range (0.0, 1.0]
    # HistGradientBoostingClassifier(random_state=randomstate),    # learning_rate is a hyper-parameter in the range (0.0, 1.0]
    AdaBoostRegressor(n_estimators=200, random_state=randomstate),
    GaussianProcessRegressor(normalize_y=True),
    ARDRegression(),
    # HuberRegressor(),   # epsilon:  greater than 1.0, default 1.35
    LinearRegression(n_jobs=5),
    PassiveAggressiveRegressor(random_state=randomstate), # C: 0.25, 0.5, 1, 5, 10
    SGDRegressor(random_state=randomstate),
    TheilSenRegressor(n_jobs=5, random_state=randomstate),
    RANSACRegressor(random_state=randomstate),
    KNeighborsRegressor(weights='distance'),  # n_neighbors: 3, 6, 9, 12, 15, 20
    RadiusNeighborsRegressor(weights='distance'),   # radius: 1, 2, 5, 10, 15
    MLPRegressor(max_iter=10000000, random_state=randomstate),
    DecisionTreeRegressor(random_state=randomstate),    # max_depth = 2, 3, 4, 6, 8
    ExtraTreeRegressor(random_state=randomstate),  # max_depth = 2, 3, 4, 6, 8
    SVR()   # C: 0.25, 0.5, 1, 5, 10
]

selectors = [
    reliefF.reliefF,
    fisher_score.fisher_score,
    chi_square.chi_square,
    JMI.jmi,
    CIFE.cife,
    DISR.disr,
    MIM.mim,
    CMIM.cmim,
    ICAP.icap,
    MRMR.mrmr,
    MIFS.mifs]

selectornames_short = ["RELF", "FSCR", "CHSQ", "JMI", "CIFE", "DISR", "MIM", "CMIM", "ICAP", "MRMR", "MIFS"]

# class boundary list
class_boundary = [304.2, 456.25]
numsplits = 10

# Dataframe for highest balanced accuracy for each feature selector / ML combination
outdf = pd.DataFrame(data=[], columns=["Feature selector", "ML method", "Split", "Parameter1", "Parameter2", "Accuracy", "Balanced Accuracy", "MSE", "r2", "spearmanr"])

for split in np.arange(numsplits):
    print("Evaluating fold " + str(split))
    train_index = kfolds["fold_" + str(split)]["train"]
    test_index = kfolds["fold_" + str(split)]["test"]

    X_train, X_test = features_nosurv.iloc[train_index], features_nosurv.iloc[test_index]
    y_train, y_test = surv_days[train_index], surv_days[test_index]
    # y_train, y_test = surv_classes[train_index], surv_classes[test_index]

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

            if clf_name is "Passive Aggressive":
                param1 = np.NaN
                param2 = np.NaN
                C = [0.25, 0.5, 1, 5, 10]
                for param1 in tqdm(C):
                    clf = PassiveAggressiveRegressor(C=param1, random_state=randomstate)

                    clf.fit(X_train_selected, y_train)

                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            elif clf_name is "SVR":
                param1 = np.NaN
                param2 = np.NaN
                C = [0.25, 0.5, 1, 5, 10]
                for param1 in tqdm(C):
                    clf = SVR(C=param1)

                    clf.fit(X_train_selected, y_train)

                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            elif clf_name is "Decision Tree":
                param1 = np.NaN
                param2 = np.NaN
                max_depthlist = [2, 3, 4, 6, 8]
                for param1 in tqdm(max_depthlist):
                    clf = DecisionTreeRegressor(max_depth=param1, random_state=randomstate)

                    clf.fit(X_train_selected, y_train)

                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            elif clf_name is "Extra Tree":
                param1 = np.NaN
                param2 = np.NaN
                max_depthlist = [2, 3, 4, 6, 8]
                for param1 in tqdm(max_depthlist):
                    clf = ExtraTreeRegressor(max_depth=param1, random_state=randomstate)

                    clf.fit(X_train_selected, y_train)

                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            # elif clf_name is "Hist. Gradient Boosting":
            #     param1 = np.NaN
            #     param2 = np.NaN
            #     lr_list = [0.1, 0.3, 0.6, 0.9]
            #     for param1 in tqdm(lr_list):
            #         clf = HistGradientBoostingClassifier(learning_rate=param1, random_state=randomstate)
            #
            #         clf.fit(X_train_selected, y_train)
            #
            #         y_pred = clf.predict(X_test_selected)
            #         y_train_pred = clf.predict(X_train_selected)
            #
            #         balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
            #         outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            elif clf_name is "Huber":
                param1 = np.NaN
                param2 = np.NaN
                eps_list = [1.1, 1.2, 1.35, 1.5, 2] # epsilon:  greater than 1.0, default 1.35
                for param1 in tqdm(eps_list):
                    clf = HistGradientBoostingClassifier(learning_rate=param1, random_state=randomstate)

                    clf.fit(X_train_selected, y_train)

                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            elif clf_name is "K-Neighbors":
                param1 = np.NaN
                param2 = np.NaN

                neighbors_list = [3, 6, 9, 12, 15, 20] # epsilon:  greater than 1.0, default 1.35
                for param1 in tqdm(neighbors_list):
                    clf = KNeighborsRegressor(n_neighbors=param1, weights='distance')

                    clf.fit(X_train_selected, y_train)

                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            elif clf_name is "Radius Neighbors":
                param1 = np.NaN
                param2 = np.NaN

                radius_list = [1, 2, 5, 10, 15]  # epsilon:  greater than 1.0, default 1.35
                for param1 in tqdm(radius_list):
                    clf = KNeighborsRegressor(radius=param1, weights='distance')

                    clf.fit(X_train_selected, y_train)

                    y_pred = clf.predict(X_test_selected)
                    y_train_pred = clf.predict(X_train_selected)

                    balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                    outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2,
                                         rho)

            else:
                param1 = np.NaN
                param2 = np.NaN

                clf.fit(X_train_selected, y_train)

                y_pred = clf.predict(X_test_selected)
                y_train_pred = clf.predict(X_train_selected)

                balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary)
                outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2,
                                         rho)


outdf.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_ageshape_regression2.csv", index=False)
