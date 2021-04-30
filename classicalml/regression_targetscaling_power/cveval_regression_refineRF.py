#!/home/yannick/anaconda3/envs/py36/bin/python


import json
import numpy as np
import pandas as pd

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from scipy.stats import spearmanr   # spearmanr(currttpdata_bratumia["TTP"].values, currttpdata_bratumia["OS"].values, nan_policy='omit')
from sklearn.preprocessing import PowerTransformer

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
        "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_regression2power_refined.csv",
        index=False)

    return outdf


def gradeoutput(y_test, y_pred, class_boundary, tfm):

    y_test = np.squeeze(tfm.inverse_transform(y_test.reshape(-1, 1)))
    y_pred = np.squeeze(tfm.inverse_transform(y_pred.reshape(-1, 1)))

    y_test_classes = survival_classencoding(y_test, class_boundary)
    y_pred_classes = survival_classencoding(y_pred, class_boundary)
    acc = accuracy_score(y_test_classes, y_pred_classes)
    balacc = balanced_accuracy_score(y_test_classes, y_pred_classes)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rho, _ = spearmanr(y_test, y_pred, nan_policy='omit')

    return [balacc, acc, mse, r2, rho]


class_boundary = [304.2, 456.25]

features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat2wsurv.csv", index_col="ID")
splitinfopath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/splitinfo.json"

features_nosurv = features.drop(columns="Survival_days", inplace=False)
surv_days = features["Survival_days"]
surv_classes = survival_classencoding(surv_days, class_boundary)

np.random.seed(42)
numclasses = 3

# load split infos
with open(splitinfopath) as f:
    kfolds = json.load(f)

numfeat = 9

randomstate = 42
classifiernames = ["Random Forest"]

classifiers = [
    RandomForestRegressor(n_estimators=200, n_jobs=10, random_state=randomstate)]

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

    # scale target with a quantile transform
    qtfm = PowerTransformer(method='yeo-johnson')
    y_train = np.squeeze(qtfm.fit_transform(y_train.values.reshape(-1, 1)))
    y_test = np.squeeze(qtfm.transform(y_test.values.reshape(-1, 1)))
    # y_train, y_test = surv_classes[train_index], surv_classes[test_index]

    # for every split, perform feature selection
    for sel_name, sel in zip([0], [0]):

        X_train_selected = X_train
        X_test_selected = X_test

        ##########################################
        # do classification with all classifiers #
        ##########################################
        best_param1 = np.NaN
        best_param2 = np.NaN
        best_balacc = np.NaN

        for clf_name, clf in zip(classifiernames, classifiers):
            print(clf_name)

            if clf_name is "Random Forest":
                param1 = np.NaN
                param2 = np.NaN
                # minleafsamples = np.arange(1, 11)
                minleafsamples = [1]
                maxfeatures = ['auto']
                # maxfeatures = ['auto', 'sqrt', 'log2']

                for param1 in tqdm(minleafsamples):
                    for param2 in maxfeatures:
                        # clf = RandomForestRegressor(n_estimators=200, min_samples_leaf=param1, max_features=param2, random_state=randomstate)
                        clf = RandomForestRegressor(n_estimators=200, n_jobs=5, random_state=randomstate)
                        # clf = RandomForestRegressor(n_estimators=200, min_samples_leaf=param1, max_features=param2, random_state=randomstate)

                        clf.fit(X_train_selected, y_train)

                        y_pred = clf.predict(X_test_selected)
                        y_train_pred = clf.predict(X_train_selected)

                        balacc, acc, mse, r2, rho = gradeoutput(y_test, y_pred, class_boundary, qtfm)
                        outdf = writeresults(outdf, sel_name, clf_name, split, param1, param2, acc, balacc, mse, r2, rho)

            else:
                pass

outdf.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/cvresults_regression2power_RF_refined.csv", index=False)
