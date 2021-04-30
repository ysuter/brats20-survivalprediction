
import os
import pandas as pd

def getsuvclass(survdays):
    survdays = float(survdays)
    if survdays < 10*365/12:
        survclass = 0

    elif survdays > 15*365/12:
        survclass = 2

    else:
        survclass = 1

    return survclass


# merge and reshape training features
pyrad_trainfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/brats20_pyradfeatures.csv")

pyrad_nonimaging = pyrad_trainfeat.loc[:, ['ID', 'Age', 'Survival_days', 'Extent_of_Resection']]
pyrad_nonimaging.drop_duplicates(inplace=True)
# drop diagnostics columns
diagnosticscol = [elem for elem in pyrad_trainfeat.columns if "diagnostics" in elem]

pyrad_trainfeat.drop(columns=diagnosticscol, inplace=True)
pyrad_trainfeat.drop(columns=['Image', 'Mask', 'Age', 'Label', 'Extent_of_Resection',
       'Patient_Timepoint', 'Reader', 'Survival_days'], inplace=True)

featcols = set(pyrad_trainfeat.columns) - set(['ID', 'Age', 'Survival_days', 'Extent_of_Resection'])

feat_wide = pyrad_trainfeat.pivot_table(index=['ID'], columns=['Labelname', 'Sequence'], values=featcols)
feat_wide.columns = feat_wide.columns.to_flat_index()
# pyrad_trainfeat.loc[pyrad_trainfeat["Survival_days"] == "ALIVE (361 days later)", "Survival_days"] = 500

feat_wide_m1 = feat_wide.merge(pyrad_nonimaging, how='left', on='ID')

# reinsert age, EOR and survival info


# read position feautures:
posfeat = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/posfeat_training.csv")

# merge
trainfeat_all = feat_wide_m1.merge(posfeat, how="left", left_on='ID', right_on="Subject")
trainfeat_all.drop(columns="Subject", inplace=True)

trainfeat_all.loc[trainfeat_all["Survival_days"] == "ALIVE (361 days later)", "Survival_days"] = 500

# encode
trainfeat_all["Survival_class"] = [getsuvclass(elem) for elem in trainfeat_all["Survival_days"]]

trainfeat_all.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/trainfeat_all_stdnorm.csv", index=None)