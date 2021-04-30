#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm

# # load feature matrix
# features = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/brats20_pyradfeatures.csv")
#
# binwidth = 5
#
# # get relevant columns
# features_relevant = features.loc[:, ["Sequence", "Labelname", "original_firstorder_Range"]]
#
# boxplot = features_relevant.boxplot(by=['Sequence', "Labelname"])
# plt.show()
#
# # calculate number of bins for each sequence and label for a given bin width
# features_relevant["Number of bins"] = features_relevant["original_firstorder_Range"] / binwidth
#
# boxplot_binwidth = features_relevant.boxplot(column="Number of bins", by=['Sequence', "Labelname"])
# plt.show()

binwidth = 5
scalefactor = 100

traindir = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"

# loop over each subject, normalize and scale as pyradiomics does, and record ranges
# subjlist = [elem for elem in os.listdir(traindir) if os.path.isdir(os.path.join(traindir, elem))]
osinfo = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"
    "/survival_info.csv")

subjlist = list(osinfo["Brats20ID"])


t1c_range = []
t1_range = []
t2_range = []
flair_range = []

for pat in tqdm(subjlist):
    currpatdir = os.path.join(traindir, pat)

    # t1cimg = sitk.ReadImage(os.path.join(currpatdir, pat + "_t1ce.nii.gz"))
    # t1img = sitk.ReadImage(os.path.join(currpatdir, pat + "_t1.nii.gz"))
    # t2img = sitk.ReadImage(os.path.join(currpatdir, pat + "_t2.nii.gz"))
    # flairimg = sitk.ReadImage(os.path.join(currpatdir, pat + "_flair.nii.gz"))

    t1cimg = sitk.ReadImage(os.path.join(currpatdir, "CT1_pwlin-biascorr_gmwm_match.nii.gz"))
    t1img = sitk.ReadImage(os.path.join(currpatdir, "T1_pwlin-biascorr_gmwm_match.nii.gz"))
    t2img = sitk.ReadImage(os.path.join(currpatdir, "T2_pwlin-biascorr_gmwm_match.nii.gz"))
    flairimg = sitk.ReadImage(os.path.join(currpatdir, "FLAIR_pwlin-biascorr_gmwm_match.nii.gz"))

    # t1cimg = sitk.GetArrayFromImage(sitk.Normalize(t1cimg))
    # t1img = sitk.GetArrayFromImage(sitk.Normalize(t1img))
    # t2img = sitk.GetArrayFromImage(sitk.Normalize(t2img))
    # flairimg = sitk.GetArrayFromImage(sitk.Normalize(flairimg))

    t1cimg = sitk.GetArrayFromImage(t1cimg)
    t1img = sitk.GetArrayFromImage(t1img)
    t2img = sitk.GetArrayFromImage(t2img)
    flairimg = sitk.GetArrayFromImage(flairimg)

    # get ranges
    t1c_range.append(np.ptp(t1cimg))
    t1_range.append(np.ptp(t1img))
    t2_range.append(np.ptp(t2img))
    flair_range.append(np.ptp(flairimg))

# evaluate
scalefactor=1
print("calculation complete.")
t1crange_scaled = np.array(t1c_range) * scalefactor
t1crange_numbins = np.array(t1crange_scaled) / binwidth
plt.hist(t1c_range)
plt.title("T1c - rescaled")
plt.show()

t1range_scaled = np.array(t1_range) * scalefactor
t1range_numbins = np.array(t1range_scaled) / binwidth
plt.hist(t1_range)
plt.title("T1 - rescaled")
plt.show()

t2range_scaled = np.array(t2_range) * scalefactor
t2range_numbins = np.array(t2range_scaled) / binwidth
plt.hist(t2_range)
plt.title("T2 - rescaled")
plt.show()

flairrange_scaled = np.array(flair_range) * scalefactor
flairrange_numbins = np.array(flairrange_scaled) / binwidth
plt.hist(flair_range)
plt.title("FLAIR - rescaled")
plt.show()

numbins = [40, 70, 100, 130]

for nbins in numbins:
    print("################ " + str(nbins) + " ################")
    print("T1c bin width: " + str(np.mean(t1c_range) / nbins))
    print("T1 bin width: " + str(np.mean(t1_range) / nbins))
    print("T2 bin width: " + str(np.mean(t2_range) / nbins))
    print("FLAIR bin width: " + str(np.mean(flair_range) / nbins))
    print("####################################")
