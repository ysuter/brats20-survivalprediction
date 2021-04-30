#!/usr/bin/env python3

import pandas as pd

# add survival info to training data
featuresforsurv = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_iterativeremoved.csv",
    index_col="ID")
survinfo = featuresforsurv["Survival_days"]
trainfeat_nosurv = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeat_normalized_NEW2.csv", index_col="ID")
trainfeat = trainfeat_nosurv.merge(survinfo, on="ID")

trainfeat.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/training_scaledfeat2wsurv.csv")


