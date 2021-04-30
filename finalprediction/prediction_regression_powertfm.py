#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.ensemble.forest import ExtraTreesRegressor, RandomForestRegressor
from joblib import dump, load
from sklearn.preprocessing import PowerTransformer

trainingfeatures = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat2wsurv.csv", index_col="ID")
clfpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/classifiers/extratree_healthynorm_powertfm.joblib"
outpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/predictions"

X_train = trainingfeatures.drop(columns="Survival_days", inplace=False)
y_train = trainingfeatures["Survival_days"]
# transform target
ptfm = PowerTransformer(method='yeo-johnson')
y_train = np.squeeze(ptfm.fit_transform(y_train.values.reshape(-1, 1)))

np.random.seed(42)

randomstate = 42

print("Train classifier...")
clf = RandomForestRegressor(n_estimators=200, n_jobs=5, random_state=randomstate)
clf.fit(X_train, y_train)
# save classifier for further use
dump(clf, clfpath)
print("Training complete...")
# clf = load(clfpath)

# VALIDATION SET
# load validation data
validationfeatures = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/validationfeat_normalized.csv", index_col="ID")

y_pred_validation_tmp = clf.predict(validationfeatures)
y_pred_validation = np.squeeze(ptfm.inverse_transform(y_pred_validation_tmp.reshape(-1, 1)))
pred_validation_df = pd.DataFrame(data=zip(validationfeatures.index.values, y_pred_validation), columns=["ID", "Prediction"])
pred_validation_df.to_csv(os.path.join(outpath, "validationprediction_powertfm_FINAL.csv"), header=False, index=False)

# TESTING SET
# load test data
testfeatures = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/testingfeat_normalized_NEW.csv", index_col="BraTS20ID")

y_pred_test_tmp = clf.predict(testfeatures)
y_pred_test = np.squeeze(ptfm.inverse_transform(y_pred_test_tmp.reshape(-1, 1)))
pred_test_df = pd.DataFrame(data=zip(testfeatures.index.values, y_pred_test), columns=["ID", "Prediction"])
pred_test_df.to_csv(os.path.join(outpath, "testprediction_powertfm_FINAL.csv"), header=False, index=False)




