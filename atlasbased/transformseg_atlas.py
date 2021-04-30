#!/usr/bin/env python3

import os
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# use the transforms used by SynReg to transform the HD-GLIO output to the atlas space (T1 space -> atlas space)
basepath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"

osinfo = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"
    "/survival_info.csv")

subjlist = list(osinfo["Brats20ID"])

for subj in tqdm(subjlist):
    subjdir = os.path.join(basepath, subj)

    # load HD-GLIO segmentation:
    segimg = sitk.ReadImage(os.path.join(subjdir, subj + "_seghdglio.nii.gz"))

    # load SyN transform
    tfm = sitk.ReadTransform(os.path.join(subjdir, "Atlas_to_t1.mat"))

    # apply transform and save
    segimg_atlasspace = sitk.Resample(segimg, tfm, sitk.sitkNearestNeighbor)

    sitk.WriteImage(segimg_atlasspace, os.path.join(subjdir, subj + "_seghdglio_syn_atlasspace.nii.gz"))
