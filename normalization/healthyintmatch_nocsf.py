#!/usr/bin/env python3

from glob import glob

import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd


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


def normalizeimages(imgpath, seqprefix, intarray_target, squencemax):
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
        # print('working 3 ...')
        labelstatfilt = sitk.LabelStatisticsImageFilter()
        labelstatfilt.Execute(currimg, labelimg)
        # print('working 4 ...')

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
                            squencemax)

    except:
        skippedlist.append(imgpath)

    return skippedlist


t1filename = '*t1-biascorr.nii.gz'
t1cfilename = '*t1ce-biascorr.nii.gz'
t2filename = '*t2-biascorr.nii.gz'
flairfilename = '*flair-biascorr.nii.gz'

basedir = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData'

flairlist = glob(os.path.join(basedir, '*', flairfilename))
t2list = glob(os.path.join(basedir, '*', t2filename))
t1list = glob(os.path.join(basedir, '*', t1filename))
t1clist = glob(os.path.join(basedir, '*', t1cfilename))

osinfo = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"
    "/survival_info.csv")

subjlist = list(osinfo["Brats20ID"])

t1clistkeep = [elem for elem in t1clist if os.path.split(elem)[-1].split('_t1ce-biascorr.nii.gz')[0] in subjlist]
t1listkeep = [elem for elem in t1list if os.path.split(elem)[-1].split('_t1-biascorr.nii.gz')[0] in subjlist]
t2listkeep = [elem for elem in t2list if os.path.split(elem)[-1].split('_t2-biascorr.nii.gz')[0] in subjlist]
flairlistkeep = [elem for elem in flairlist if os.path.split(elem)[-1].split('_flair-biascorr.nii.gz')[0] in subjlist]

# only keep elements in the survival task training set

# get distribution for each sequence
print('Getting the distribution of the FLAIR images...')
inp_maxflair = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in tqdm(flairlistkeep)]
print('Getting the distribution of the T2 images...')
inp_maxt2 = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in tqdm(t2listkeep)]
print('Getting the distribution of the T1 images...')
inp_maxt1 = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in tqdm(t1listkeep)]
print('Getting the distribution of the T1c images...')
inp_maxt1c = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in tqdm(t1clistkeep)]

maxmean_flair = np.mean(inp_maxflair)
maxmean_t2 = np.mean(inp_maxt2)
maxmean_t1 = np.mean(inp_maxt1)
maxmean_t1c = np.mean(inp_maxt1c)

# get mean intensity values for CSF, WM and GM labels for each sequence
t1_gm_mean = []
t1_wm_mean = []
t1_gm_std = []
t1_wm_std = []

t1c_gm_mean = []
t1c_wm_mean = []
t1c_gm_std = []
t1c_wm_std = []

t2_gm_mean = []
t2_wm_mean = []
t2_gm_std = []
t2_wm_std = []

flair_gm_mean = []
flair_wm_mean = []
flair_gm_std = []
flair_wm_std = []

print('Calculate T1 statistics...')
for img in tqdm(t1listkeep):
    currsubj = os.path.split(img)[-1].split('_t1-biascorr.nii.gz')[0]
    # print(img)
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], currsubj + "_healthyseg.nii.gz")
    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    t1_gm_mean.append(labelstatfilt.GetMean(2))
    t1_wm_mean.append(labelstatfilt.GetMean(3))

    t1_gm_std.append(labelstatfilt.GetSigma(2))
    t1_wm_std.append(labelstatfilt.GetSigma(3))

t1_gm_overallmean = np.mean(t1_gm_mean)
t1_wm_overallmean = np.mean(t1_wm_mean)

print('Calculate T1c statistics...')
for img in tqdm(t1clistkeep):
    currsubj = os.path.split(img)[-1].split('_t1ce-biascorr.nii.gz')[0]
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], currsubj + "_healthyseg.nii.gz")

    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    t1_gm_mean.append(labelstatfilt.GetMean(2))
    t1_wm_mean.append(labelstatfilt.GetMean(3))

    t1_gm_std.append(labelstatfilt.GetSigma(2))
    t1_wm_std.append(labelstatfilt.GetSigma(3))

t1c_gm_overallmean = np.mean(t1_gm_mean)
t1c_wm_overallmean = np.mean(t1_wm_mean)

print('Calculate T2 statistics...')
for img in tqdm(t2listkeep):
    currsubj = os.path.split(img)[-1].split('_t2-biascorr.nii.gz')[0]
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], currsubj + "_healthyseg.nii.gz")

    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    t2_gm_mean.append(labelstatfilt.GetMean(2))
    t2_wm_mean.append(labelstatfilt.GetMean(3))

    t2_gm_std.append(labelstatfilt.GetSigma(2))
    t2_wm_std.append(labelstatfilt.GetSigma(3))

t2_gm_overallmean = np.mean(t2_gm_mean)
t2_wm_overallmean = np.mean(t2_wm_mean)

print('Calculate FLAIR statistics...')
for img in tqdm(flairlistkeep):
    currsubj = os.path.split(img)[-1].split('_flair-biascorr.nii.gz')[0]
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], currsubj + "_healthyseg.nii.gz")

    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    flair_gm_mean.append(labelstatfilt.GetMean(2))
    flair_wm_mean.append(labelstatfilt.GetMean(3))

    flair_gm_std.append(labelstatfilt.GetSigma(2))
    flair_wm_std.append(labelstatfilt.GetSigma(3))

flair_gm_overallmean = np.mean(flair_gm_mean)
flair_wm_overallmean = np.mean(flair_wm_mean)

print('T1c, GM: ' + str(t1c_gm_overallmean))
print('T1c, WM: ' + str(t1c_wm_overallmean))
print('--')
print('T1, GM: ' + str(t1_gm_overallmean))
print('T1, WM: ' + str(t1_wm_overallmean))
print('--')
print('T2, GM: ' + str(t2_gm_overallmean))
print('T2, WM: ' + str(t2_wm_overallmean))
print('--')
print('FLAIR, GM: ' + str(flair_gm_overallmean))
print('FLAIR, WM: ' + str(flair_wm_overallmean))
print('--')

# normalize each sequence
num_cores = 3

print('# Normalizing T1c sequences')
intarray_target = [t1c_gm_overallmean, t1c_wm_overallmean]
seqprefix = "CT1"
t1c_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t1cpath, seqprefix, intarray_target, maxmean_t1c) for t1cpath in tqdm(t1clistkeep))

print('### Skipped T1c:')
print(t1c_skipped)

print('# Normalizing T1 sequences')
intarray_target = [t1_gm_overallmean, t1_wm_overallmean]
seqprefix = "T1"
t1_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t1path, seqprefix, intarray_target, maxmean_t1) for t1path in tqdm(t1listkeep))

print('### Skipped T1:')
print(t1_skipped)

print('# Normalizing T2 sequences')
intarray_target = [t2_gm_overallmean, t2_wm_overallmean]
seqprefix = "T2"
t2_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t2path, seqprefix, intarray_target, maxmean_t2) for t2path in tqdm(t2listkeep))

print('### Skipped T2:')
print(t2_skipped)

print('# Normalizing FLAIR sequences')
intarray_target = [flair_gm_overallmean, flair_wm_overallmean]
seqprefix = "FLAIR"
flair_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(flairpath, seqprefix, intarray_target, maxmean_flair) for flairpath in tqdm(flairlistkeep))

print('### Skipped FLAIR:')
print(flair_skipped)

print("Done.")
