
import os
import shutil
from glob import glob


def generatefilename(origname: str):
    # BraTS20_Training_002_flair.nii.gz
    idparts = origname.split('_')[0:3]
    subjid = '_'.join(idparts)

    newfilename = None
    passbool = False

    if 't1.nii.gz' in origname:
        newfilename = subjid + '_0000.nii.gz'

    elif 't1ce' in origname:
        newfilename = subjid + '_0001.nii.gz'

    elif 't2' in origname:
        newfilename = subjid + '_0002.nii.gz'

    elif 'flair' in origname:
        newfilename = subjid + '_0003.nii.gz'

    else:
        passbool = True

    return passbool, newfilename


# basepath = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData_processing'
basepath = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_ValidationData_processing'
outfolder = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_ValidationData_processing_hdglio'

subjdirs = [elem for elem in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, elem))]

for subj in subjdirs:
    currsubjpath = os.path.join(basepath, subj)
    # get all nifti files
    currniftilist = glob(os.path.join(currsubjpath, "*.nii.gz"))

    for niftifile in currniftilist:
        origpathelems = os.path.split(niftifile)
        passbool, newfilename = generatefilename(origpathelems[1])
        if not passbool:
            os.rename(niftifile, os.path.join(origpathelems[0], newfilename))
            shutil.copy(os.path.join(origpathelems[0], newfilename), os.path.join(outfolder, newfilename))

print("Done :-)")
