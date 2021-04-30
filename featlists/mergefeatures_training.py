#!/usr/bin/env python3

import os
import pandas as pd

binwidthlist = [40, 70, 100, 130]
sequencelist = ["t1c", "t1", "t2", "flair"]

basepath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/"

for sidx, sequence in enumerate(sequencelist):
    for binidx, binwidth in enumerate(binwidthlist):
        # load data
        currpath = os.path.join(basepath, "brats20_pyradout_training_" + str(sequence) + "_" + str(binwidth) + "bins.csv")
        currfeat = pd.read_csv(currpath)
        currfeat["numbins"] = [str(binwidth)] * currfeat.shape[0]
        # drop diagnostic columns and image/mask paths
        diagnosticscols = [elem for elem in currfeat.columns.values if 'diagnostic' in elem]
        currfeat.drop(columns=diagnosticscols + ["Age", "Survival_days", "Extent_of_Resection", "Image", "Mask",
                                                 "Label", "Patient_Timepoint", "Reader"], inplace=True)

        currfeat_reshaped = currfeat.pivot_table(index=["ID"], columns=["Labelname", "Sequence", "numbins"])
        # flatten multiindex
        currfeat_reshaped.columns = [' '.join(col).strip() for col in currfeat_reshaped.columns.values]

        if (sidx == 0) and (binidx == 0):
            overalldf = currfeat_reshaped
        else:
            overalldf = pd.merge(overalldf, currfeat_reshaped, left_index=True, right_index=True)

print(overalldf.shape)

# load arbitrary feature csv to get demographics etc
arbfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/brats20_pyradout_training_t1c_40bins.csv", index_col="ID")
arbfeat = arbfeat.loc[arbfeat["Labelname"] == "enhancing"]

keepcols = ['Age', 'Survival_days', 'Extent_of_Resection']
arbfeat = arbfeat.loc[:, keepcols]

# merge
overalldf_full = pd.merge(overalldf, arbfeat, on="ID")

# remove non-t1c shape columns
shapecols = [elem for elem in overalldf_full.columns if 'shape' in elem]
shape_featnames = [elem.split(' ')[0] for elem in shapecols]
shapediscard = [elem for elem in shapecols if 'T1c' not in elem or '40' not in elem]
overalldf_full.drop(columns=shapediscard, inplace=True)

# add position features
posfeatures = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/posfeat_training.csv")
posfeatures.rename(columns={"Subject": "ID"}, inplace=True)
overalldf_full = pd.merge(overalldf_full, posfeatures, on="ID")

overalldf_full.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/training_allfeat_bins.csv")
