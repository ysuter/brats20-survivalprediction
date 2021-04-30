#!/usr/bin/env python

import os
import shutil
import SimpleITK as sitk

inputdirhmseg = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/healthyseg/output'
# targetdir = '/home/yannick/remoteubelix/MANAGE/data/postopprogrusable_normalized'
targetdir = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/postopprogrusable_normalized'
segfilename = 'segmentation.nii.gz'
brainmaskname = 'T1_r2s_bet_mask.nii.gz'

# ugly hack
sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(12)
sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(12)

patlist = sorted([elem for elem in os.listdir(targetdir) if os.path.isdir(os.path.join(targetdir, elem))])

missinglist = []

for pat in patlist:
    currpatdir = os.path.join(targetdir, pat)
    tplist = sorted([elem for elem in os.listdir(currpatdir) if os.path.isdir(os.path.join(currpatdir, elem))])

    for tp in tplist:
        currtpdir = os.path.join(currpatdir, tp)
        outfilepath = os.path.join(currtpdir, 'healthyseg_pp.nii.gz')

        if os.path.isfile(outfilepath):
            continue

        sourcefilename = str(pat) + '_' + str(tp) + '.nii.gz'
        sourcefile = os.path.join(inputdirhmseg, sourcefilename)

        print(sourcefilename.split('.')[0])

        # mask with brainmask to exclude
        # try:
        healthysegpath = os.path.join(targetdir, pat, tp, 'healthyseg_raw.nii.gz')
        if not os.path.isfile(healthysegpath):
            shutil.copyfile(sourcefile, healthysegpath)
        healthyinp = sitk.ReadImage(healthysegpath)
        seginp = sitk.ReadImage(os.path.join(currtpdir, segfilename))
        brainmaskinp = sitk.ReadImage(os.path.join(currtpdir, brainmaskname))
        # threshold tumor segmentation to include both labels. Negate mask for later use to mask out tumor areas
        seginp_thresh = sitk.BinaryThreshold(seginp, 1, 2, 0, 1)

        healthy_masked = sitk.Mask(healthyinp, brainmaskinp)
        healthy_notumor = sitk.Mask(healthy_masked, seginp_thresh)

        # except:
        missinglist.append(sourcefilename.split('.')[0])
        # print(missinglist)
        # continue

        # save
        sitk.WriteImage(healthy_notumor, outfilepath)
print(missinglist)
