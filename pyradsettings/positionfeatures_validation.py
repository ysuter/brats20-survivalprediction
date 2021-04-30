#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def geteuclideandist(inpdist: float):
    if inpdist < 0:
        return -np.sqrt(np.abs(inpdist))
    else:
        return np.sqrt(np.abs(inpdist))


rootdir = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_ValidationData_onlysurvival"
# rootdir = "/media/yannick/MANAGE/BraTS20/MICCAI_BraTS2020_ValidationData"
# ventricledistmap = sitk.ReadImage("/media/yannick/MANAGE/BraTS20/ventricle_distancemap.nii.gz")

# subjlist = [elem for elem in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, elem))]

featdf = pd.DataFrame(data=[], columns=["Subject", "x_mincet", "x_maxcet", "y_mincet", "y_maxcet", "z_mincet",
                               "z_maxcet", "x_mined", "x_maxed", "y_mined", "y_maxed", "z_mined",
                               "z_maxed", "cet_ventrdist", "ed_ventdist"])

osinfo = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/survival_evaluation.csv")

subjlist = list(osinfo["ID"])

# valinp = pd.read_csv("/media/yannick/MANAGE/BraTS20/survival_evaluation.csv")
# subjlist = valinp["ID"].values

for subj in tqdm(subjlist):

    # load MNI-registered label map
    segmni = sitk.ReadImage(os.path.join(rootdir, subj, subj + "_segmni.nii.gz"), sitk.sitkUInt8)
    t1mni = sitk.ReadImage(os.path.join(rootdir, subj, subj + "_t1mni.nii.gz"), sitk.sitkFloat32)
    labelmapfilt = sitk.LabelStatisticsImageFilter()
    labelmapfilt.Execute(t1mni, segmni)

    ventricledistmap = sitk.ReadImage(os.path.join(rootdir, subj, "ventricledistmap_t1space.nii.gz"))
    segt1space = sitk.ReadImage(os.path.join(rootdir, subj, subj + "_seghdglio.nii.gz"))

    # get distance to ventricles
    ventr_labelmapfilt = sitk.LabelStatisticsImageFilter()
    ventr_labelmapfilt.Execute(ventricledistmap, segt1space)

    if labelmapfilt.GetCount(2) != 0:
        x_mincet, x_maxcet, y_mincet, y_maxcet, z_mincet, z_maxcet = labelmapfilt.GetBoundingBox(2)
        cet_mindist = ventr_labelmapfilt.GetMinimum(2)
    else:
        x_mincet, x_maxcet, y_mincet, y_maxcet, z_mincet, z_maxcet = [0] * 6
        cet_mindist = 4000
    if labelmapfilt.GetCount(1) != 0:
        x_mined, x_maxed, y_mined, y_maxed, z_mined, z_maxed = labelmapfilt.GetBoundingBox(1)
        ed_mindist = ventr_labelmapfilt.GetMinimum(1)
    else:
        x_mined, x_maxed, y_mined, y_maxed, z_mined, z_maxed = [0] * 6
        ed_mindist = 4000

    currfeat = {"Subject": subj, "x_mincet": x_mincet, "x_maxcet": x_maxcet, "y_mincet": y_mincet,
                "y_maxcet": y_maxcet, "z_mincet": z_mincet, "z_maxcet": z_maxcet, "x_mined": x_mined,
                "x_maxed": x_maxed, "y_mined": y_mined, "y_maxed": y_maxed, "z_mined": z_mined,
                "z_maxed": z_maxed, "cet_ventrdist": cet_mindist, "ed_ventdist": ed_mindist}
    featdf = featdf.append(pd.DataFrame(currfeat, index=[0]))

featdf["cet_ventrdist"] = [geteuclideandist(elem) for elem in featdf["cet_ventrdist"]]
featdf["ed_ventdist"] = [geteuclideandist(elem) for elem in featdf["ed_ventdist"]]
featdf.to_csv(os.path.join(rootdir, "..", "posfeat_validation.csv"), index=False)
# featdf.to_csv(os.path.join(rootdir, "posfeat_validation.csv"))





