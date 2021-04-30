#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

figoutpath = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/plots_featurereduction"

# load c-index information
cindices = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/"
                       "concordanceidx_training_all.csv")

cind_threslist = np.arange(0.5, 0.65, 0.005)

numremaining = []
numedema = []
numenhancing = []
numt1c = []
numt1 = []
numt2 = []
numflair = []
# number of bins
num40 = []
num70 = []
num100 = []
num130 = []
# origin
numorig = []
numlog2 = []
numlog3 = []
numwavelet = []

for thresh in cind_threslist:
    # filter features above c-index threshold
    featfiltered = cindices.loc[cindices["ConcordanceIndex"] >= thresh]

    # get information about remaining features
    remainingfeatures = featfiltered["Feature"].values

    numremaining.append(len(remainingfeatures))

    numedema.append(len([elem for elem in remainingfeatures if "edema" in elem]))
    numenhancing.append(len([elem for elem in remainingfeatures if "enhancing" in elem]))

    numt1c.append(len([elem for elem in remainingfeatures if "T1c" in elem]))
    numt1.append(len([elem for elem in remainingfeatures if "T1 " in elem]))
    numt2.append(len([elem for elem in remainingfeatures if "T2 " in elem]))
    numflair.append(len([elem for elem in remainingfeatures if "FLAIR " in elem]))

    num40.append(len([elem for elem in remainingfeatures if " 40" in elem]))
    num70.append(len([elem for elem in remainingfeatures if " 70" in elem]))
    num100.append(len([elem for elem in remainingfeatures if " 100" in elem]))
    num130.append(len([elem for elem in remainingfeatures if " 130" in elem]))

    numorig.append(len([elem for elem in remainingfeatures if "original" in elem]))
    numlog2.append(len([elem for elem in remainingfeatures if "log-sigma-2" in elem]))
    numlog3.append(len([elem for elem in remainingfeatures if "log-sigma-3" in elem]))
    numwavelet.append(len([elem for elem in remainingfeatures if "wavelet" in elem]))

# plotting
fig = plt.figure()
plt.plot(cind_threslist, numremaining)
plt.yscale('log')
plt.title("Remaining features for c-index thresholding")
plt.xlabel("C-index threshold")
plt.ylabel("Number of remaining features")
plt.savefig(os.path.join(figoutpath, "overallremaining_cindex.svg"))
plt.savefig(os.path.join(figoutpath, "overallremaining_cindex.png"))
plt.show()

# remaining feature per segmentation label
fig = plt.figure()
plt.plot(cind_threslist, numedema, label="Edema features")
plt.plot(cind_threslist, numenhancing, label="Contrast-enhancement features")
# plt.yscale('log')
plt.title("Remaining features for c-index thresholding")
plt.xlabel("C-index threshold")
plt.ylabel("Number of remaining features")
plt.savefig(os.path.join(figoutpath, "remainingperlabel_cindex.svg"))
plt.savefig(os.path.join(figoutpath, "remainingperlabel_cindex.png"))
plt.legend()
plt.show()

# remaining feature per sequence label
fig = plt.figure()
plt.plot(cind_threslist, numt1c, label="T1c features")
plt.plot(cind_threslist, numt1, label="T1 features")
plt.plot(cind_threslist, numt2, label="T2 features")
plt.plot(cind_threslist, numflair, label="FLAIR features")
# plt.yscale('log')
plt.title("Remaining features for c-index thresholding")
plt.xlabel("C-index threshold")
plt.ylabel("Number of remaining features")
plt.savefig(os.path.join(figoutpath, "remainingpersequence_cindex.svg"))
plt.savefig(os.path.join(figoutpath, "remainingpersequence_cindex.png"))
plt.legend()
plt.show()

# remaining feature per image type
fig = plt.figure()
plt.plot(cind_threslist, numorig, label="Original image")
plt.plot(cind_threslist, numlog2, label="LoG filtered, sigma=2")
plt.plot(cind_threslist, numlog3, label="LoG filtered, sigma=3")
plt.plot(cind_threslist, numwavelet, label="Wavelet image")
# plt.yscale('log')
plt.title("Remaining features for c-index thresholding")
plt.xlabel("C-index threshold")
plt.ylabel("Number of remaining features")
plt.savefig(os.path.join(figoutpath, "remainingperimgtype_cindex.svg"))
plt.savefig(os.path.join(figoutpath, "remainingperimgtype_cindex.png"))
# plt.legend(frameon=False)
plt.show()

# remaining feature per bin count
fig = plt.figure()
plt.plot(cind_threslist, num40, label="40 bins")
plt.plot(cind_threslist, num70, label="70 bins")
plt.plot(cind_threslist, num100, label="100 bins")
plt.plot(cind_threslist, num130, label="130 bins")
# plt.yscale('log')
plt.title("Remaining features for c-index thresholding")
plt.xlabel("C-index threshold")
plt.ylabel("Number of remaining features")
plt.savefig(os.path.join(figoutpath, "remainingperbin_cindex.svg"))
plt.savefig(os.path.join(figoutpath, "remainingperbin_cindex.png"))
plt.legend()
plt.show()

numremainingdf = pd.DataFrame(data=zip(cind_threslist, numremaining), columns=["Threshold", "RemainingFeatures"])



