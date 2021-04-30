
import os
from glob import glob
from shutil import copyfile
import pandas as pd

basepath = '/home/yannick/Documents/BraTS2020/MICCAI_BraTS2020_TestingData'
healthyoutpath = '/home/yannick/remotenerve/brats2020/testdata_healthysegout'

subjlist = [elem for elem in os.listdir(healthyoutpath) if '.nii.gz' in elem]

for subj in subjlist:
    currsubjpath = os.path.join(healthyoutpath, subj)
    subjid = subj.split('.nii.gz')[0]
    newsegfilepath = os.path.join(basepath, subjid, subjid + '_healthyseg.nii.gz')

    copyfile(currsubjpath, newsegfilepath)

print("Done :-)")
