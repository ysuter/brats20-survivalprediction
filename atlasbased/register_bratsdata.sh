#!/bin/bash

export FREESURFER_HOME=/usr/local/freesurfer
source /usr/local/freesurfer/SetUpFreeSurfer.sh

cd /media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData
basepath=/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/BRATS/BraTS2020/MICCAI_BraTS2020_TrainingData
atlaspath=/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

for d in */; do

    currdir=${basepath}/${d}

    itktfm=${currdir}/mni_affine.txt

    echo ${itktfm}
 
    if test -f "$itktfm"; then
	  echo "skipping"
    else

        echo ${d}

        mri_robust_register --mov ${currdir}/${d%/}_t1.nii.gz --dst ${atlaspath} --lta ${currdir}/mni_affine.lta --iscale --initorient --affine --satit --maxit 200
        mri_convert --resample_type nearest --apply_transform ${currdir}/mni_affine.lta ${currdir}/${d%/}_seghdglio.nii.gz ${currdir}/${d%/}_segmni.nii.gz
        mri_convert --resample_type interpolate --apply_transform ${currdir}/mni_affine.lta ${currdir}/${d%/}_t1.nii.gz ${currdir}/${d%/}_t1mni.nii.gz
        lta_convert --inlta  ${currdir}/mni_affine.lta --outitk ${currdir}/mni_affine.txt

        echo "processed $d"

    fi

done

echo "done :-)"

