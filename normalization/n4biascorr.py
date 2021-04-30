#!/usr/bin/env python3

import SimpleITK as sitk
import argparse
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd


def biascorrect_patienfiles(pat_inputpath: str, biascorrsuffix: str):

    patid = os.path.split(pat_inputpath)[-1]

    if os.path.isfile(os.path.join(pat_inputpath, patid + '_flair' + biascorrsuffix + '.nii.gz')):
        return 0

    else:
        # process T1c, T1, T2, and FLAIR images
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetNumberOfThreads(2)

        # BraTS data come as 16-bis signed ints, which is not supported by the N4biascorr filter, therefore the casting...
        # T1c
        t1cimg = sitk.ReadImage(os.path.join(pat_inputpath, patid + '_t1ce.nii.gz'))
        t1c_pixeltype = t1cimg.GetPixelIDValue()
        t1c_cast = sitk.Cast(t1cimg, sitk.sitkFloat64)
        t1c_corr = corrector.Execute(t1c_cast)
        sitk.WriteImage(sitk.Cast(t1c_corr, t1c_pixeltype),
                        os.path.join(pat_inputpath, patid + '_t1ce' + biascorrsuffix + '.nii.gz'))

        # T1
        t1img = sitk.ReadImage(os.path.join(pat_inputpath, patid + '_t1.nii.gz'))
        t1_pixeltype = t1img.GetPixelIDValue()
        t1_cast = sitk.Cast(t1img, sitk.sitkFloat64)
        t1_corr = corrector.Execute(t1_cast)
        sitk.WriteImage(sitk.Cast(t1_corr, t1_pixeltype),
                        os.path.join(pat_inputpath, patid + '_t1' + biascorrsuffix + '.nii.gz'))

        # T2
        t2img = sitk.ReadImage(os.path.join(pat_inputpath, patid + '_t2.nii.gz'))
        t2_pixeltype = t2img.GetPixelIDValue()
        t2_cast = sitk.Cast(t2img, sitk.sitkFloat64)
        t2_corr = corrector.Execute(t2_cast)
        sitk.WriteImage(sitk.Cast(t2_corr, t2_pixeltype),
                        os.path.join(pat_inputpath, patid + '_t2' + biascorrsuffix + '.nii.gz'))

        # FLAIR
        flairimg = sitk.ReadImage(os.path.join(pat_inputpath, patid + '_flair.nii.gz'))
        flair_pixeltype = flairimg.GetPixelIDValue()
        flair_cast = sitk.Cast(flairimg, sitk.sitkFloat64)
        flair_corr = corrector.Execute(flair_cast)
        sitk.WriteImage(sitk.Cast(flair_corr, flair_pixeltype),
                        os.path.join(pat_inputpath, patid + '_flair' + biascorrsuffix + '.nii.gz'))

        return 0


def main(inputpath: str, biascorrsuffix: str, numcpus: int):
    print("Bias field correction using " + str(numcpus) + " CPUs.")

    # parallel processing across patients
    osinfo = pd.read_csv(
        "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"
        "/survival_info.csv")

    patientlist = list(osinfo["Brats20ID"])
    # patientlist = [elem for elem in os.listdir(inputpath) if os.path.isdir(os.path.join(inputpath, elem))]
    patientpaths = [os.path.join(inputpath, elem) for elem in patientlist]

    Parallel(n_jobs=numcpus)(
        delayed(biascorrect_patienfiles)(pat, biascorrsuffix) for pat in tqdm(patientpaths))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for the nnUnet-based healthy tissue segmentation')

    parser.add_argument(
        '--inputpath',
        type=str,
        default="/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData",
        help='Path to the directory with the per-subject subfolders.'
    )

    parser.add_argument(
        '--biascorrsuffix',
        type=str,
        default="_biascorr",
        help='Suffix for bias-field corrected files.'
    )

    parser.add_argument(
        '--numcpus',
        type=int,
        default=5,
        help='Number of CPUs for multiprocessing.'
    )

    args = parser.parse_args()

main(args.inputpath, args.biascorrsuffix, args.numcpus)
