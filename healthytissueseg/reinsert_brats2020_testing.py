
import os
from glob import glob
from shutil import copyfile

basepath = '/home/ysuter/brats2020/MICCAI_BraTS2020_TestingData'
hdgliooutpath = '/home/ysuter/brats2020/testdata_hdgliooout'

subjlist = [elem for elem in os.listdir(hdgliooutpath) if '.nii.gz' in elem]

for subj in subjlist:
    currsubjpath = os.path.join(hdgliooutpath, subj)
    subjid = subj.split('.nii.gz')[0]
    newsegfilepath = os.path.join(basepath, subjid, subjid + '_seghdglio.nii.gz')

    copyfile(currsubjpath, newsegfilepath)

print("Done :-)")
