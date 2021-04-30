#!/usr/bin/env python3

import SimpleITK as sitk
import os
import pandas as pd
from radiomics import featureextractor
from tqdm import tqdm

T1c_filename = "CT1_pwlin-biascorr_gmwm_match.nii.gz"
T1_filename = "T1_pwlin-biascorr_gmwm_match.nii.gz"
T2_filename = "T2_pwlin-biascorr_gmwm_match.nii.gz"
mask_suffix = "_seghdglio.nii.gz"

rootdir_onlysurv = "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TestingData"

osinfo = pd.read_csv(
    "/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TestingData/survival_evaluation.csv")

subjlist = list(osinfo["BraTS20ID"])

# valinp = pd.read_csv("/media/yannick/MANAGE/BraTS20/survival_evaluation.csv")
# subjlist = valinp["ID"].values

overallfeatdf = pd.DataFrame(data=[],
                             columns=['log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity edema T1c 40',
                                                   'log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity enhancing T1c 40',
                                                   'original_firstorder_90Percentile edema T1 40',
                                                   'z_mincet',
                                                   'original_firstorder_Median edema T1 40',
                                                   'original_firstorder_10Percentile edema T2 40',
                                                   'wavelet-HHH_glszm_GrayLevelNonUniformity enhancing T1 40',
                                                   'original_shape_Maximum2DDiameterColumn enhancing T1c 40',
                                                   'original_firstorder_Range enhancing T2 40'])

for subj in tqdm(subjlist):
    currsubjdir = os.path.join(rootdir_onlysurv, subj)
    maskpath = os.path.join(currsubjdir, subj + mask_suffix)

    # extract 'log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity edema T1c 40'
    imgpath = os.path.join(currsubjdir, T1c_filename)

    # Initialize feature extractor
    settings = {}
    settings['binWidth'] = 63.676416015625
    settings['resampledPixelSpacing'] = None
    settings['normalize'] = False
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # By default, only original is enabled. Optionally enable some image types:
    extractor.disableAllImageTypes()
    extractor.enableImageTypes(LoG={'sigma': [2.0]})

    # Disable all classes except firstorder
    extractor.disableAllFeatures()

    # Only enable mean and skewness in firstorder
    extractor.enableFeaturesByName(glszm=['GrayLevelNonUniformity'])

    try:
        featureVector = extractor.execute(imgpath, maskpath, label=1)
        # drop diagnostic columns
        diagnosticcols = [elem for elem in featureVector.keys() if "diagnostics" in elem]
        [featureVector.pop(elem, None) for elem in diagnosticcols]
    except:
        featureVector = {"log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity": 0}
    # rename to include sequence, bin, and label
    featureVector['log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity edema T1c 40'] = featureVector.pop("log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity")
    currfeaturedf = pd.DataFrame(data=featureVector, index=[subj])
    # currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector, index=[subj]),
    #                                     left_index=True, right_index=True)

    # 'log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity enhancing T1c 40'
    try:
        featureVector2 = extractor.execute(imgpath, maskpath, label=2)
        diagnosticcols = [elem for elem in featureVector2.keys() if "diagnostics" in elem]
        [featureVector2.pop(elem, None) for elem in diagnosticcols]
    except:
        featureVector2 = {"log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity": 0}
    featureVector2['log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity enhancing T1c 40'] = featureVector2.pop(
        "log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity")
    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector2, index=[subj]),
                                        left_index=True, right_index=True)


    # extract 'original_firstorder_90Percentile edema T1 40'
    imgpath = os.path.join(currsubjdir, T1_filename)
    settings['binWidth'] = 30.251226806640624
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypes(Original={})
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['90Percentile'])
    try:
        featureVector4 = extractor.execute(imgpath, maskpath, label=1)
        diagnosticcols = [elem for elem in featureVector4.keys() if "diagnostics" in elem]
        [featureVector4.pop(elem, None) for elem in diagnosticcols]
    except:
        featureVector4 = {"original_firstorder_90Percentile": 0}
    featureVector4['original_firstorder_90Percentile edema T1 40'] = featureVector4.pop(
        "original_firstorder_90Percentile")
    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector4, index=[subj]),
                                        left_index=True, right_index=True)

    # z_mincet position feature
    # load MNI-registered label map
    segmni = sitk.ReadImage(os.path.join(currsubjdir, subj + "_segmni.nii.gz"), sitk.sitkUInt8)
    t1mni = sitk.ReadImage(os.path.join(currsubjdir, subj + "_t1mni.nii.gz"), sitk.sitkFloat32)
    labelmapfilt = sitk.LabelStatisticsImageFilter()
    labelmapfilt.Execute(t1mni, segmni)

    if labelmapfilt.GetCount(2) != 0:
        x_mincet, x_maxcet, y_mincet, y_maxcet, z_mincet, z_maxcet = labelmapfilt.GetBoundingBox(2)
    else:
        x_mincet, x_maxcet, y_mincet, y_maxcet, z_mincet, z_maxcet = [0] * 6
    if labelmapfilt.GetCount(1) != 0:
        x_mined, x_maxed, y_mined, y_maxed, z_mined, z_maxed = labelmapfilt.GetBoundingBox(1)
    else:
        x_mined, x_maxed, y_mined, y_maxed, z_mined, z_maxed = [0] * 6

    featureVectorPos = {"z_mincet": z_mincet}
    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVectorPos, index=[subj]),
                                        left_index=True, right_index=True)

    # extract 'original_firstorder_Median edema T1 40'
    imgpath = os.path.join(currsubjdir, T1_filename)
    settings['binWidth'] = 30.251226806640624
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Median'])
    try:
        featureVector5 = extractor.execute(imgpath, maskpath, label=1)
        diagnosticcols = [elem for elem in featureVector5.keys() if "diagnostics" in elem]
        [featureVector5.pop(elem, None) for elem in diagnosticcols]
    except:
        featureVector5 = {"original_firstorder_Median": 0}
    featureVector5['original_firstorder_Median edema T1 40'] = featureVector5.pop(
        "original_firstorder_Median")
    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector5, index=[subj]),
                                        left_index=True, right_index=True)

    # extract original_firstorder_10Percentile edema T2 40
    imgpath = os.path.join(currsubjdir, T2_filename)
    settings['binWidth'] = 48.22628173828125
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypes(Original={})
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['10Percentile'])
    try:
        featureVector55 = extractor.execute(imgpath, maskpath, label=1)
        diagnosticcols = [elem for elem in featureVector55.keys() if "diagnostics" in elem]
        [featureVector55.pop(elem, None) for elem in diagnosticcols]
    except:
        featureVector55 = {"original_firstorder_10Percentile": 0}
    featureVector55['original_firstorder_10Percentile edema T2 40'] = featureVector55.pop(
        "original_firstorder_10Percentile")
    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector55, index=[subj]),
                                        left_index=True, right_index=True)

    # extract 'wavelet-HHH_glszm_GrayLevelNonUniformity enhancing T1 40'
    imgpath = os.path.join(currsubjdir, T1_filename)
    settings['binWidth'] = 30.251226806640624
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypes(Wavelet={})
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(glszm=['GrayLevelNonUniformity'])
    try:
        featureVector6 = extractor.execute(imgpath, maskpath, label=2)
        diagnosticcols = [elem for elem in featureVector6.keys() if ("diagnostics" in elem)]
        [featureVector6.pop(elem, None) for elem in diagnosticcols]
        irrelevantcols = [elem for elem in featureVector6.keys() if not ("HHH" in elem)]
        [featureVector6.pop(elem, None) for elem in irrelevantcols]
    except:
        featureVector6 = {"wavelet-HHH_glszm_GrayLevelNonUniformity": 0}

    featureVector6['wavelet-HHH_glszm_GrayLevelNonUniformity enhancing T1 40'] = featureVector6.pop(
        "wavelet-HHH_glszm_GrayLevelNonUniformity")

    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector6, index=[subj]),
                                        left_index=True, right_index=True)

    # extract 'original_shape_Maximum2DDiameterColumn enhancing T1c 40
    imgpath = os.path.join(currsubjdir, T1c_filename)
    settings['binWidth'] = 63.676416015625
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypes(Original={})
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(shape=['Maximum2DDiameterColumn'])
    try:
        featureVector8 = extractor.execute(imgpath, maskpath, label=2)
        diagnosticcols = [elem for elem in featureVector8.keys() if "diagnostics" in elem]
        [featureVector8.pop(elem, None) for elem in diagnosticcols]
    except:
        featureVector8 = {"original_shape_Maximum2DDiameterColumn": 0}
    featureVector8['original_shape_Maximum2DDiameterColumn enhancing T1c 40'] = featureVector8.pop(
        "original_shape_Maximum2DDiameterColumn")
    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector8, index=[subj]),
                                        left_index=True, right_index=True)

    # extract 'original_firstorder_Range enhancing T2 40'
    imgpath = os.path.join(currsubjdir, T2_filename)
    settings['binWidth'] = 48.22628173828125
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypes(Original={})
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Range'])
    try:
        featureVector7 = extractor.execute(imgpath, maskpath, label=2)
        diagnosticcols = [elem for elem in featureVector7.keys() if "diagnostics" in elem]
        [featureVector7.pop(elem, None) for elem in diagnosticcols]
    except:
        featureVector7 = {"original_firstorder_Range": 0}
    featureVector7['original_firstorder_Range enhancing T2 40'] = featureVector7.pop(
        "original_firstorder_Range")
    currfeaturedf = currfeaturedf.merge(pd.DataFrame(data=featureVector7, index=[subj]),
                                        left_index=True, right_index=True)

    # print(currfeaturedf)

    overallfeatdf = overallfeatdf.append(currfeaturedf, sort=False)

overallfeatdf.index.name = 'BraTS20ID'

# save non-normalized features
overallfeatdf.to_csv('/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/testingfeat_NEW.csv')

# normalize features
# load normalization information
scalinginfo = pd.read_csv("/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/featsel_outputs/scalinginfo.csv", index_col="Feature")

featdf_normalized = overallfeatdf.copy(deep=True)
for col in overallfeatdf.columns.values:
    scale_mean = scalinginfo.loc[col, "mean"]
    scale_std = scalinginfo.loc[col, "std"]

    featdf_normalized.loc[:, col] = (featdf_normalized.loc[:, col] - scale_mean) / scale_std

featdf_normalized.to_csv('/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/testingfeat_normalized_NEW.csv')
