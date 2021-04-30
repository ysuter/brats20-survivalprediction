
import os
from glob import glob
from shutil import copyfile
import pandas as pd

basepath = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_ValidationData_onlysurvival'
hdgliooutpath = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/validation_heathysegout'

# osinfo = pd.read_csv(
#     "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"
#     "/survival_info.csv")
#
# subjlist = list(osinfo["Brats20ID"])

subjlist = [elem for elem in os.listdir(hdgliooutpath) if '.nii.gz' in elem]

for subj in subjlist:
    currsubjpath = os.path.join(hdgliooutpath, subj)
    subjid = subj.split('.nii.gz')[0]
    newsegfilepath = os.path.join(basepath, subjid, subjid + '_healthyseg.nii.gz')

    copyfile(currsubjpath, newsegfilepath)

print("Done :-)")
