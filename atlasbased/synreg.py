#!/usr/bin/env python3

import os
import argparse
import shutil
import ants
from tqdm import tqdm
from glob import glob
import SimpleITK as sitk


def main(inputpath: str, outdir: str):

    tmpdir = "/tmp"

    ventricledistmap = sitk.ReadImage("/media/yannick/MANAGE/BraTS20/ventricle_distancemap.nii.gz")

    # create output directory, if it does not already exist
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # get patient directories
    patdirs = [elem for elem in os.listdir(inputpath) if os.path.isdir(os.path.join(inputpath, elem))]

    skippedlist = []

    for patdir in tqdm(patdirs):
        currpatdir = os.path.join(inputpath, patdir)

        caseid = ('_').join([patdir, tp])
            t1_patfile = os.path.join(currpatdir, "T1_r2s_bet-biascorr.nii.gz")
            if not os.path.isfile(t1_patfile) or os.path.isfile(os.path.join(outdir, caseid + "_t1_atlasspace.nii.gz")):
                print("Skipped " + str(caseid))
                continue

            else:
                fi = ants.image_read(t1_patfile)
                mi = ants.image_read(atlaspath)
                try:
                    tx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', verbose=False)


                    # save transform and warped images to output folder
                    shutil.copy(tx["fwdtransforms"][1], os.path.join(outdir, caseid + "_atlas_to_t1.mat"))
                    shutil.copy(tx["invtransforms"][0], os.path.join(outdir, caseid + "_t1_to_atlas.mat"))
                    ants.image_write(tx["warpedmovout"], os.path.join(outdir, caseid + "_atlas_t1space.nii.gz"))
                    ants.image_write(tx["warpedfixout"], os.path.join(outdir, caseid + "_t1_atlasspace.nii.gz"))

                    # delete temporary *.nii.gz and *.mat files
                    try:
                        niftitmp = glob(os.path.join(tmpdir, "*.nii.gz"))
                        mattmp = glob(os.path.join(tmpdir, "*.mat"))

                        [os.remove(elem) for elem in niftitmp]
                        [os.remove(elem) for elem in mattmp]
                    except:
                        pass

                except:
                    skippedlist.append(caseid)
                    print("Skipped " + str(caseid))
                    continue



if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Registration with ANTs')

    parser.add_argument(
        '--inputpath',
        type=str,
        # default=os.cwd(),
        default="/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/progression_hdlgio_availjan20",
        help='Path to the current patients or timepoint folder'
    )

    parser.add_argument(
        '--atlaspath',
        type=str,
        # default=os.cwd(),
        default="/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/tractography/atlas/100HCP-population-mean-T1.nii.gz",
        help='Path to the atlas'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        # default=os.cwd(),
        default="/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/tractography/tfms_atlas_to_t1",
        help='Output directory for the transform files'
    )

    args = parser.parse_args()

main(args.inputpath, args.atlaspath, args.outdir)
