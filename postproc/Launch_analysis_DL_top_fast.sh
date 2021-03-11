#!/usr/bin/env bash
export PATH=~/miniconda3/bin:$PATH 

DIRECTORY=/lfs/l1/legend/users/aalexander/Ba133_AV_char/postproc/
MC_RAW=/lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/raw-IC160A-BA133-uncollimated-top-run0003-81z-newgeometry.hdf5
#MC_RAW=/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/IC-legend/IC160A/Ba133/uncollimated/top/raw-IC160A-BA133-uncollimated-top-run0003-81z-newgeometry-00.hdf5

MC_file_id=IC160A-BA133-uncollimated-top-run0003-81z-newgeometry
#MC_file_id=IC160A-BA133-uncollimated-top-run0003-81z-newgeometry-singlefile

CONF_PATH=${DIRECTORY}detectors/I02160A/constants_I02160A.json
SMEAR=g
fFCCD=0.74
fDLTp_list=(0 0.25 0.5 0.75 1)


for fFCCD in 0.71
    do
        for fDLTp in 0 0.25 0.5 0.75 1
            do
                echo "fFCCD is $fFCCD"
                echo "fDLTp is $fDLTp"
                python ${DIRECTORY}analysis_DL_top_fast.py $MC_RAW $MC_file_id $CONF_PATH $SMEAR $fFCCD $fDLTp
            done
    done