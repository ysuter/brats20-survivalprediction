#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

figoutpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/plots_featurereduction"

# load c-index information
cindices = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/concordanceidx_training_nobinduplicates.csv")

features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_nobinduplicates.csv", index_col="ID")

# get features with a concordance index higher than 0.55
keepfeat = cindices[cindices["ConcordanceIndex"] > 0.55]["Feature"]
dropfeat = list(set(cindices["Feature"]) - set(keepfeat))
# numremainingdf = pd.DataFrame(data=zip(cind_threslist, numremaining), columns=["Threshold", "RemainingFeatures"])

feat_filtered = features.drop(columns=dropfeat, inplace=False)

feat_filtered.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_nobinduplicates_Cg055.csv")

