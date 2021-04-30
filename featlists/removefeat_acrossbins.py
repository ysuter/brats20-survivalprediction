#!/usr/bin/env python3

import numpy as np
import pandas as pd

# load c-index data
cindices = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/"
                       "concordanceidx_training_all.csv")
features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/training_allfeat_bins.csv")
features.drop(columns=["Unnamed: 0"], inplace=True)

# check for features that are extracted for multiple number of histogram bins
featnames = cindices["Feature"].values
cindices.set_index("Feature", inplace=True)
bininfo = [[' '.join(elem.split(' ')[0:3]), elem.split(' ')[3]] for elem in featnames if len(elem.split(' ')) > 2]

cind_df = pd.DataFrame(data=bininfo, columns=["Feature", "numBins"])

cind_df["C-index"] = [cindices.loc[(" ").join(elem.values[0:2]), "ConcordanceIndex"] for index, elem in cind_df.iterrows()]
potduplicatefeatures = [(" ").join(elem.values[0:2].astype(str)) for index, elem in cind_df.iterrows()]

cind_df["numBins"] = [int(elem) for elem in cind_df["numBins"]]
test_sorted = cind_df.sort_values(by=["Feature", "C-index"], ascending=[True, False])

# keep first duplicate, since it's sorted (descending)
keepdf = test_sorted.drop_duplicates(subset=["Feature"], keep='first')
keeplist = [(" ").join(elem.values[0:2].astype(str)) for index, elem in keepdf.iterrows()]
droplist = list(set(potduplicatefeatures) - set(keeplist))

# drop duplicate on input features
features_filtered = features.drop(columns=droplist, inplace=False)
features_filtered.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainingfeatures_nobinduplicates.csv", index=False)
