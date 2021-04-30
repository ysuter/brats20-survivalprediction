
import os
import numpy as np
import SimpleITK as sitk


class MergeLabel(Transform):

    def __init__(self, to_combine: dict) -> None:
        super().__init__()
        # to_combine is a dict with keys -> new label and values -> list of labels to merge
        self.to_combine = to_combine

    def __call__(self, img: sitk.Image) -> sitk.Image:
        np_img = sitk.GetArrayFromImage(img)
        merged_img = np.zeros_like(np_img)

        for new_label, labels_to_merge in self.to_combine.items():
            merged_img[np.in1d(np_img.ravel(), labels_to_merge, assume_unique=True).reshape(np_img.shape)] = new_label

        out_img = sitk.GetImageFromArray(merged_img)
        out_img.CopyInfo


# load resampled atlas file
atlas = sitk.ReadImage("/media/yannick/MANAGE/BraTS20/aseg_resampled.nrrd")

# create new labelmaps
atlas_new = sitk.GetImageFromArray(np.zeros(atlas.GetSize()))
atlas_new.CopyInformation(atlas)

ventriclemap = sitk.Threshold(atlas, )