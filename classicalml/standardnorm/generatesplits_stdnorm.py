#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def survival_classencoding(survarr: np.array, classboundaries: list):
    if len(classboundaries) == 1:
        survival_classes = [0 if elem <= classboundaries[0] else 1 for elem in survarr]

    if len(classboundaries) == 2:
        survival_classes = [0 if elem <= classboundaries[0] else 1 if elem <= classboundaries[1] else 2 for elem in
                            survarr]

    return np.array(survival_classes)


class_boundary = [304.2, 456.25]

# read feature matrix
features = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_iterativeremoved.csv",
    index_col="ID")
features_nosurv = features.drop(columns="Survival_days", inplace=False)
survival = features["Survival_days"]
print("Loaded " + str(features.shape[1]) + " features for " + str(features.shape[0]) + " patients.")

# normalize (zero mean, variance one)
scaler = StandardScaler()
scaler.fit(features_nosurv)
features_scaled = scaler.transform(features_nosurv)
stdarr = scaler.scale_
meanarr = scaler.mean_

# save normalzation info
keycols = features_nosurv.columns.values
scaler_stddict = dict(zip(keycols, stdarr))
scaler_meandict = dict(zip(keycols, meanarr))

mean_df = pd.DataFrame.from_dict(scaler_meandict, orient="index", columns=["mean"])
std_df = pd.DataFrame.from_dict(scaler_stddict, orient="index", columns=["std"])
mean_df.index.rename("Feature", inplace=True)
std_df.index.rename("Feature", inplace=True)
scalerdf = mean_df.merge(std_df, on="Feature")

scalerdf.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/scalinginfo.csv")

features_scaleddf = features_nosurv.copy(deep=True)
features_scaleddf.loc[:, :] = features_scaled

# put survival info back
featscaled_wsurv = features_scaleddf.merge(survival, on="ID")
featscaled_wsurv.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat2.csv")

X = features_scaled
y = survival_classencoding(survival, class_boundary)

nsplits = 10

kfold = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=42)
folds = {}
count = 0

for train_ix, test_ix in kfold.split(X, y):
    # select rows
    train_X, test_X = X[train_ix], X[test_ix]
    train_y, test_y = y[train_ix], y[test_ix]
    # summarize train and test composition
    train_0, train_1, train_2 = len(train_y[train_y == 0]), len(train_y[train_y == 1]), len(train_y[train_y == 2])
    test_0, test_1, test_2 = len(test_y[test_y == 0]), len(test_y[test_y == 1]), len(test_y[test_y == 2])
    print('>Train: 0=%d, 1=%d, 2=%d, Test: 0=%d, 1=%d, 2=%d' % (train_0, train_1, train_2, test_0, test_1, test_2))
    # save current split

    folds['fold_{}'.format(count)] = {}
    folds['fold_{}'.format(count)]['train'] = train_ix.tolist()
    folds['fold_{}'.format(count)]['test'] = test_ix.tolist()
    count += 1

print(len(folds) == nsplits)  # assert we have the same number of splits
# dump folds to json

splitinfopath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/splitinfo.json"

with open(splitinfopath, 'w') as fp:
    json.dump(folds, fp)

# loading
with open(splitinfopath) as f:
    kfolds = json.load(f)

for key, val in kfolds.items():
    print(key)
    train = val['train']
    test = val['test']
