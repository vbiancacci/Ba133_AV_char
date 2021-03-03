#!/usr/bin/env bash
export PATH=~/miniconda3/bin:$PATH 

DIRECTORY=/lfs/l1/legend/users/aalexander/Ba133_AV_char/postproc/
MC_RAW=/lfs/l1/legend/users/aalexander/hdf5_output/detector_IC160A_ba_top_81mmNEW8_01.hdf5
CONF_PATH=${DIRECTORY}detectors/I02160A/constants_I02160A.json
SMEAR=g
fFCCD=0.74
fDLTp_list=(0 0.25 0.5 0.75 1)
 
for fDLTp in 0 0.25 0.5 0.75 1
    do
        echo "$fDLTp is fDLTp"
        python ${DIRECTORY}analysis_DL_top_fast.py $MC_RAW $CONF_PATH $SMEAR $fFCCD $fDLTp
    done