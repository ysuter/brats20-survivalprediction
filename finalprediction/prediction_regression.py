#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from skfeature.function.information_theoretical_based import MIFS
from sklearn.ensemble.forest import ExtraTreesRegressor
from joblib import dump, load


trainingfeatures = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat2wsurv.csv", index_col="ID")
clfpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/classifiers/extratree_healthynorm.joblib"
outpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/predictions"

X_train = trainingfeatures.drop(columns="Survival_days", inplace=False)
y_train = trainingfeatures["Survival_days"]
# surv_classes_train = survival_classencoding(surv_days, class_boundary)

np.random.seed(42)

randomstate = 42

print("Train classifier...")
clf = ExtraTreesRegressor(n_estimators=200, n_jobs=5, random_state=randomstate)
clf.fit(X_train, y_train)
# save classifier for further use
dump(clf, clfpath)
print("Training complete...")
# clf = load(clfpath)

# VALIDATION SET
# load validation data
validationfeatures = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/validationfeat_normalized.csv", index_col="ID")

y_pred_validation = clf.predict(validationfeatures)
pred_validation_df = pd.DataFrame(data=zip(validationfeatures.index.values, y_pred_validation), columns=["ID", "Prediction"])
pred_validation_df.to_csv(os.path.join(outpath, "validationprediction.csv"), header=False, index=False)

# TESTING SET
# load test data
testfeatures = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/testingfeat_normalized_NEW.csv", index_col="BraTS20ID")

y_pred_test = clf.predict(testfeatures)
pred_test_df = pd.DataFrame(data=zip(testfeatures.index.values, y_pred_test), columns=["ID", "Prediction"])
pred_test_df.to_csv(os.path.join(outpath, "testprediction.csv"), header=False, index=False)




