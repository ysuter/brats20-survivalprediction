from glob import glob

import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


def transform_intensity(inpimg: sitk.Image, outpath: str, orig_i1: float, orig_i2: float, orig_i3: float,
                        target_i1: float,
                        target_i2: float, target_i3: float, target_max: float) -> sitk.Image:
    # rescale intensity to [0, orig_max]
    orig_max = np.max(sitk.GetArrayFromImage(inpimg))

    inpimg_scaled = sitk.RescaleIntensity(inpimg, 0, float(orig_max))

    x = sitk.GetArrayFromImage(inpimg_scaled)

    normimg_np = np.piecewise(x, [x <= orig_i1, (x > orig_i1) & (x <= orig_i2), (x > orig_i2) & (x <= orig_i3), x > orig_i3],
                    [lambda x: x * target_i1 / orig_i1,
                     lambda x: x * (target_i2 - target_i1) / (orig_i2 - orig_i1) + target_i1 - orig_i1 * (target_i1 - target_i2) / (orig_i1 - orig_i2),
                     lambda x: x * (target_i3 - target_i2) / (orig_i3 - orig_i2) + target_i2 - orig_i2 * (target_i2 - target_i3) / (orig_i2 - orig_i3),
                     lambda x: x * (target_max - target_i3) / (orig_max - orig_i3) + target_i3 - orig_i3 * (target_i3 - target_max) / (orig_i3 - orig_max)])

    normimg = sitk.GetImageFromArray(normimg_np)

    normimg.CopyInformation(inpimg)

    sitk.WriteImage(normimg, outpath)

    return normimg


def normalizeimages(imgpath, seqprefix, intarray_target, squencemax):
    skippedlist = []

    try:
        # load image and label
        currimg = sitk.ReadImage(imgpath)

        labelpath = os.path.join(os.path.split(imgpath)[0], healthyseg)
        labelimg = sitk.ReadImage(labelpath)

        labelstatfilt = sitk.LabelStatisticsImageFilter()
        labelstatfilt.Execute(currimg, labelimg)

        csforig = labelstatfilt.GetMean(1)
        gmorig = labelstatfilt.GetMean(2)
        wmorig = labelstatfilt.GetMean(3)

        # sort mean intensity values for given sequence in ascending order
        intarray_orig = [csforig, gmorig, wmorig]
        indx_sort = np.argsort(intarray_target)
        intarray_target_sorted = [intarray_target[i] for i in indx_sort]
        intarray_orig_sorted = [intarray_orig[i] for i in indx_sort]

        outpath = os.path.join(os.path.split(imgpath)[0], seqprefix + '_pwlin_match.nii.gz')

        transform_intensity(currimg, outpath, intarray_orig_sorted[0], intarray_orig_sorted[1],
                            intarray_orig_sorted[2],
                            intarray_target_sorted[0], intarray_target_sorted[1], intarray_target_sorted[2],
                            squencemax)

    except:
        skippedlist.append(imgpath)

    return skippedlist


t1filename = 'T1_r2s_bet-biascorr.nii.gz'
t1cfilename = 'CT1_r2s_bet_regT1-biascorr.nii.gz'
t2filename = 'T2_r2s_bet_regT1-biascorr.nii.gz'
flairfilename = 'FLAIR_r2s_bet_regT1-biascorr.nii.gz'
healthyseg = 'healthyseg_pp.nii.gz'


basedir = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/BraTS20'

flairlist = glob(os.path.join(basedir, '*/*', flairfilename), recursive=True)
t2list = glob(os.path.join(basedir, '*/*', t2filename))
t1list = glob(os.path.join(basedir, '*/*', t1filename))
t1clist = glob(os.path.join(basedir, '*/*', t1cfilename))

# get distribution for each sequence
inp_maxflair = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in flairlist]
inp_maxt2 = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in t2list]
inp_maxt1 = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in t1list]
inp_maxt1c = [np.max(sitk.GetArrayFromImage(sitk.ReadImage(img))) for img in t1clist]

maxmean_flair = np.mean(inp_maxflair)
maxmean_t2 = np.mean(inp_maxt2)
maxmean_t1 = np.mean(inp_maxt1)
maxmean_t1c = np.mean(inp_maxt1c)

# get mean intensity values for CSF, WM and GM labels for each sequence
t1_csf_mean = []
t1_gm_mean = []
t1_wm_mean = []
t1_csf_std = []
t1_gm_std = []
t1_wm_std = []

t1c_csf_mean = []
t1c_gm_mean = []
t1c_wm_mean = []
t1c_csf_std = []
t1c_gm_std = []
t1c_wm_std = []

t2_csf_mean = []
t2_gm_mean = []
t2_wm_mean = []
t2_csf_std = []
t2_gm_std = []
t2_wm_std = []

flair_csf_mean = []
flair_gm_mean = []
flair_wm_mean = []
flair_csf_std = []
flair_gm_std = []
flair_wm_std = []

print('Calculate T1 statistics...')
for img in t1list:
    print(img)
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], healthyseg)
    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    t1_csf_mean.append(labelstatfilt.GetMean(1))
    t1_gm_mean.append(labelstatfilt.GetMean(2))
    t1_wm_mean.append(labelstatfilt.GetMean(3))

    t1_csf_std.append(labelstatfilt.GetSigma(1))
    t1_gm_std.append(labelstatfilt.GetSigma(2))
    t1_wm_std.append(labelstatfilt.GetSigma(3))

t1_csf_overallmean = np.mean(t1_csf_mean)
t1_gm_overallmean = np.mean(t1_gm_mean)
t1_wm_overallmean = np.mean(t1_wm_mean)

print('Calculate T1c statistics...')
for img in t1clist:
    print(img)
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], healthyseg)
    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    t1_csf_mean.append(labelstatfilt.GetMean(1))
    t1_gm_mean.append(labelstatfilt.GetMean(2))
    t1_wm_mean.append(labelstatfilt.GetMean(3))

    t1_csf_std.append(labelstatfilt.GetSigma(1))
    t1_gm_std.append(labelstatfilt.GetSigma(2))
    t1_wm_std.append(labelstatfilt.GetSigma(3))

t1c_csf_overallmean = np.mean(t1_csf_mean)
t1c_gm_overallmean = np.mean(t1_gm_mean)
t1c_wm_overallmean = np.mean(t1_wm_mean)

print('Calculate T2 statistics...')
for img in t2list:
    print(img)
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], healthyseg)
    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    t2_csf_mean.append(labelstatfilt.GetMean(1))
    t2_gm_mean.append(labelstatfilt.GetMean(2))
    t2_wm_mean.append(labelstatfilt.GetMean(3))

    t2_csf_std.append(labelstatfilt.GetSigma(1))
    t2_gm_std.append(labelstatfilt.GetSigma(2))
    t2_wm_std.append(labelstatfilt.GetSigma(3))

t2_csf_overallmean = np.mean(t2_csf_mean)
t2_gm_overallmean = np.mean(t2_gm_mean)
t2_wm_overallmean = np.mean(t2_wm_mean)

print('Calculate FLAIR statistics...')
for img in flairlist:
    print(img)
    currimg = sitk.ReadImage(img)
    labelpath = os.path.join(os.path.split(img)[0], healthyseg)
    labelimg = sitk.ReadImage(labelpath)
    labelstatfilt = sitk.LabelStatisticsImageFilter()
    labelstatfilt.Execute(currimg, labelimg)

    flair_csf_mean.append(labelstatfilt.GetMean(1))
    flair_gm_mean.append(labelstatfilt.GetMean(2))
    flair_wm_mean.append(labelstatfilt.GetMean(3))

    flair_csf_std.append(labelstatfilt.GetSigma(1))
    flair_gm_std.append(labelstatfilt.GetSigma(2))
    flair_wm_std.append(labelstatfilt.GetSigma(3))

flair_csf_overallmean = np.mean(flair_csf_mean)
flair_gm_overallmean = np.mean(flair_gm_mean)
flair_wm_overallmean = np.mean(flair_wm_mean)

print(len(flair_csf_mean))

print('T1c, CSF: ' + str(t1c_csf_overallmean))
print('T1c, GM: ' + str(t1c_gm_overallmean))
print('T1c, WM: ' + str(t1c_wm_overallmean))
print('--')
print('T1, CSF: ' + str(t1_csf_overallmean))
print('T1, GM: ' + str(t1_gm_overallmean))
print('T1, WM: ' + str(t1_wm_overallmean))
print('--')
print('T2, CSF: ' + str(t2_csf_overallmean))
print('T2, GM: ' + str(t2_gm_overallmean))
print('T2, WM: ' + str(t2_wm_overallmean))
print('--')
print('FLAIR, CSF: ' + str(flair_csf_overallmean))
print('FLAIR, GM: ' + str(flair_gm_overallmean))
print('FLAIR, WM: ' + str(flair_wm_overallmean))
print('--')

# normalize each sequence
num_cores = multiprocessing.cpu_count() - 1

print('# Normalizing T1c sequences')
intarray_target = [t1c_csf_overallmean, t1c_gm_overallmean, t1c_wm_overallmean]
seqprefix = "CT1"
t1c_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t1cpath, seqprefix, intarray_target, maxmean_t1c) for t1cpath in t1clist)

print('### Skipped T1c:')
print(t1c_skipped)

print('# Normalizing T1 sequences')
intarray_target = [t1_csf_overallmean, t1_gm_overallmean, t1_wm_overallmean]
seqprefix = "T1"
t1_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t1path, seqprefix, intarray_target, maxmean_t1) for t1path in t1list)

print('### Skipped T1:')
print(t1_skipped)

print('# Normalizing T2 sequences')
intarray_target = [t2_csf_overallmean, t2_gm_overallmean, t2_wm_overallmean]
seqprefix = "T2"
t2_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(t2path, seqprefix, intarray_target, maxmean_t2) for t2path in t2list)

print('### Skipped T2:')
print(t2_skipped)

print('# Normalizing FLAIR sequences')
intarray_target = [flair_csf_overallmean, flair_gm_overallmean, flair_wm_overallmean]
seqprefix = "FLAIR"
flair_skipped = Parallel(n_jobs=num_cores)(
    delayed(normalizeimages)(flairpath, seqprefix, intarray_target, maxmean_flair) for flairpath in flairlist)

print('### Skipped FLAIR:')
print(flair_skipped)

print("Done.")
