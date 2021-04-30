#!/usr/bin/env python3

import argparse
import os
import shutil
from glob import glob
import ants
from tqdm import tqdm


def main(inputpath: str):

    tmpdir = "/tmp"

    # ventricledistmap = sitk.ReadImage("ventricle_distancemap.nii.gz")
    ventricledistpath = "/home/yannick/Documents/BraTS2020/ventricle_distancemap.nii.gz"
    atlaspath = "/home/yannick/Documents/BraTS2020/MNI152_T1_1mm_brain.nii.gz"

    t1suffix = "_t1.nii.gz"

    # get patient directories
    patdirs = [elem for elem in os.listdir(inputpath) if os.path.isdir(os.path.join(inputpath, elem))]

    ventricledistmap = ants.image_read(ventricledistpath)

    skippedlist = []

    for patdir in tqdm(patdirs):
        currpatdir = os.path.join(inputpath, patdir)
        t1_patfile = os.path.join(currpatdir, patdir + t1suffix)

        try:

            fi = ants.image_read(t1_patfile)
            mi = ants.image_read(atlaspath)

            tx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', verbose=False)
            ventrdist_t1space = ants.apply_transforms(fi, ventricledistmap, tx["fwdtransforms"], interpolator='linear', imagetype=0,
                                                      whichtoinvert=None, compose=None, verbose=False)

            # save transform and warped images to output folder
            shutil.copy(tx["fwdtransforms"][1], os.path.join(currpatdir, "Atlas_to_t1.mat"))
            shutil.copy(tx["invtransforms"][0], os.path.join(currpatdir, "T1_to_atlas.mat"))
            ants.image_write(tx["warpedmovout"], os.path.join(currpatdir, "Atlas_t1space.nii.gz"))
            ants.image_write(tx["warpedfixout"], os.path.join(currpatdir, "T1_atlasspace.nii.gz"))
            ants.image_write(ventrdist_t1space, os.path.join(currpatdir, "ventricledistmap_t1space.nii.gz"))

            # delete temporary *.nii.gz and *.mat files
            try:
                niftitmp = glob(os.path.join(tmpdir, "*.nii.gz"))
                mattmp = glob(os.path.join(tmpdir, "*.mat"))

                [os.remove(elem) for elem in niftitmp]
                [os.remove(elem) for elem in mattmp]
            except:
                pass

        except:
            skippedlist.append(patdir)
            print("Skipped " + str(patdir))
            continue

    print("###### COMPLETED ######")
    print("Skipped cases:")
    print(skippedlist)


if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Registration with ANTs')

    parser.add_argument(
        '--inputpath',
        type=str,
        # default=os.cwd(),
        default="/home/yannick/Documents/BraTS2020/MICCAI_BraTS2020_TestingData",
        help='Path to the current patients or timepoint folder'
    )

    args = parser.parse_args()

main(args.inputpath)
