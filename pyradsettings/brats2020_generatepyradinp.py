import csv
import os
import glob
import numpy as np
import pandas as pd


def getsubjfilenames(subjid: str, sequence: str):
    if sequence == "T1c":
        seqfilename = subjid + "_t1ce.nii.gz"
    elif sequence == "T1":
        seqfilename = subjid + "_t1.nii.gz"
    elif sequence == "T2":
        seqfilename = subjid + "_t2.nii.gz"
    elif sequence == "FLAIR":
        seqfilename = subjid + "_flair.nii.gz"

    maskfilename = subjid + "_seghdglio.nii.gz"

    return seqfilename, maskfilename




labelintlist = [1, 2]
labelnames = ['edema', 'enhancing']

sequencelist = ["T1c", "T1", "T2", "FLAIR"]

# inpsurvdata = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/brats20_surv.csv'
inpsurvdata = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/survival_evaluation.csv'
imgbasepath = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_ValidationData'

inpsurv = pd.read_csv(inpsurvdata)
inpsurv.set_index("ID", drop=False, inplace=True)

pyradinp = pd.DataFrame(data=[], columns=list(inpsurv.columns) + ['Image', 'Mask', 'Label', 'Sequence'])

for subj in inpsurv["ID"]:
    for sequence in sequencelist:
        for labelidx, label in enumerate(labelintlist):
            seqfile, maskfile = getsubjfilenames(subj, sequence)
            currsubjdict = {"ID": subj,
                            "Age": inpsurv.at[subj, "Age"],
                            # "Survival_days": inpsurv.at[subj, "Survival_days"],
                            "ResectionStatus": inpsurv.at[subj, "ResectionStatus"],
                            "Image": os.path.join(imgbasepath, subj, seqfile),
                            "Mask": os.path.join(imgbasepath, subj, maskfile),
                            "Label": label,
                            "Labelname": labelnames[labelidx],
                            "Sequence": sequence
                            }

            pyradinp = pyradinp.append(currsubjdict, ignore_index=True)

# rename EOR column to match name in training data
pyradinp.rename(columns={'ResectionStatus':'Extent_of_Resection'}, inplace=True)
print(pyradinp)
print(pyradinp.shape)
pyradinp.to_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/brats20_validation_pyradinp.csv", index=None)
