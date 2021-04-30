#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import shutil


def main(inputpath: str, outputpath: str):
    # load table with OS task subjects
    # osinfo = pd.read_csv(
    #     "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData"
    #     "/survival_info.csv")

    osinfo = pd.read_csv("/home/ysuter/brats2020/MICCAI_BraTS2020_TestingData/survival_evaluation.csv")

    patientlist = list(osinfo["BraTS20ID"])

    # check if output path exists, if not, create it
    if not os.path.isdir(outputpath):
        print("Output directory does not exist, creating it...")
        os.makedirs(outputpath, exist_ok=True)
        print("Output directory created: " + outputpath)

    # patientlist = [elem for elem in os.listdir(inputpath) if os.path.isdir(os.path.join(inputpath, elem))]

    filesuffixmapdict = {"_t1ce.nii.gz": "_0000.nii.gz",
                         "_t1.nii.gz": "_0001.nii.gz",
                         "_t2.nii.gz": "_0002.nii.gz",
                         "_flair.nii.gz": "_0003.nii.gz"}

    for patient in patientlist:
        patientpath = os.path.join(inputpath, patient)
        # copy to output directory and rename
        for mriseq in filesuffixmapdict.keys():
            sourefile = os.path.join(patientpath, patient + mriseq)
            destfile = os.path.join(outputpath, patient + filesuffixmapdict[mriseq])
            shutil.copyfile(sourefile, destfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for the nnUnet-based healthy tissue segmentation')

    parser.add_argument(
        '--inputpath',
        type=str,
        # default="/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData",
        default="/home/ysuter/brats2020/MICCAI_BraTS2020_TestingData",
        help='Path to the directory with the per-subject subfolders.'
    )

    parser.add_argument(
        '--outputpath',
        type=str,
        # default="/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/training_healthyseginp",
        default="/home/ysuter/brats2020/testdata_prepped",
        help='Output directory for the renamed files.'
    )

    args = parser.parse_args()

main(args.inputpath, args.outputpath)
