#!/usr/bin/env python3

from glob import glob

import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import json


def transform_intensity(inpimg: sitk.Image, outpath: str, orig_i1: float, orig_i2: float, target_i1: float,
                        target_i2: float, target_max: float) -> sitk.Image:

    orig_max = np.max(sitk.GetArrayFromImage(inpimg))

    inpimg_scaled = sitk.RescaleIntensity(inpimg, 0, float(orig_max))

    x = sitk.GetArrayFromImage(inpimg_scaled)

    normimg_np = np.piecewise(x, [x <= orig_i1, (x > orig_i1) & (x <= orig_i2), x > orig_i2],
                    [lambda x: x * target_i1 / orig_i1,
                     lambda x: x * (target_i2 - target_i1) / (orig_i2 - orig_i1) + target_i1 - orig_i1 * (target_i1 - target_i2) / (orig_i1 - orig_i2),
                     lambda x: x * (target_max - target_i2) / (orig_max - orig_i2) + target_i2 - orig_i2 * (target_i2 - target_max) / (orig_i2 - orig_max)])

    normimg = sitk.GetImageFromArray(normimg_np)
    normimg.CopyInformation(inpimg)
    sitk.WriteImage(normimg, outpath)

    return normimg


def normalizeimages(imgpath, seqprefix, intarray_target, sequencemax):
    skippedlist = []

    try:
        # load image and label

        if seqprefix == "CT1":
            splitstring = "_t1ce-biascorr.nii.gz"
        elif seqprefix == "T1":
            splitstring = "_t1-biascorr.nii.gz"
        elif seqprefix == "T2":
            splitstring = "_t2-biascorr.nii.gz"
        elif seqprefix == "FLAIR":
            splitstring = "_flair-biascorr.nii.gz"
        else:
            skippedlist.append(imgpath)
            return skippedlist

        currimg = sitk.ReadImage(imgpath)

        currsubj = os.path.split(imgpath)[-1].split(splitstring)[0]

        labelpath = os.path.join(os.path.split(imgpath)[0], currsubj + "_healthyseg.nii.gz")

        labelimg = sitk.ReadImage(labelpath)

        labelstatfilt = sitk.LabelStatisticsImageFilter()
        labelstatfilt.Execute(currimg, labelimg)

        gmorig = labelstatfilt.GetMean(2)
        wmorig = labelstatfilt.GetMean(3)

        # sort mean intensity values for given sequence in ascending order
        intarray_orig = [gmorig, wmorig]
        indx_sort = np.argsort(intarray_target)
        intarray_target_sorted = [intarray_target[i] for i in indx_sort]
        intarray_orig_sorted = [intarray_orig[i] for i in indx_sort]

        outpath = os.path.join(os.path.split(imgpath)[0], seqprefix + '_pwlin-biascorr_gmwm_match.nii.gz')
        # print('outpath: ' + str(outpath))
        transform_intensity(currimg, outpath, intarray_orig_sorted[0], intarray_orig_sorted[1],
                            intarray_target_sorted[0], intarray_target_sorted[1],
                            sequencemax)

    except:
        skippedlist.append(imgpath)

    return skippedlist


num_cores = 10

# use mean intensities from the training set to normalize the test set accordingly
intarray_target_t1c = [531.0903373888692, 634.6218705214288]
maxmean_t1c = 2547.556
intarray_target_t1 = 484.14687212619907, 597.9634286423741
maxmean_t1 = 1210.4825
intarray_target_t2 = 616.2608032981316, 445.0025944559956
maxmean_t2 = 1929.505
intarray_target_flair = 295.45856159953627, 284.72177265406197
maxmean_flair = 974.4955

t1filename = '*t1-biascorr.nii.gz'
t1cfilename = '*t1ce-biascorr.nii.gz'
t2filename = '*t2-biascorr.nii.gz'
flairfilename = '*flair-biascorr.nii.gz'

# Normalize test set
testsetpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TestingData"

flairlist_test = glob(os.path.join(testsetpath, '*', flairfilename))
t2list_test = glob(os.path.join(testsetpath, '*', t2filename))
t1list_test = glob(os.path.join(testsetpath, '*', t1filename))
t1clist_test = glob(os.path.join(testsetpath, '*', t1cfilename))

seqprefix = "CT1"
t1c_skipped_test = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t1cpath, seqprefix, intarray_target_t1c, maxmean_t1c) for t1cpath in tqdm(t1clist_test))
print('### Skipped T1c, test set:')
print(t1c_skipped_test)

seqprefix = "T1"
t1_skipped_test = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t1path, seqprefix, intarray_target_t1, maxmean_t1) for t1path in tqdm(t1list_test))
print('### Skipped T1, test set:')
print(t1_skipped_test)

seqprefix = "T2"
t2_skipped_test = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t2path, seqprefix, intarray_target_t2, maxmean_t2) for t2path in tqdm(t2list_test))
print('### Skipped T2, test set:')
print(t2_skipped_test)

seqprefix = "FLAIR"
flair_skipped_test = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(flairpath, seqprefix, intarray_target_flair, maxmean_flair) for flairpath in tqdm(flairlist_test))
print('### Skipped FLAIR, test set:')
print(flair_skipped_test)
