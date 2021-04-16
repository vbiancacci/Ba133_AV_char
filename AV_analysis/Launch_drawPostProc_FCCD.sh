#!/usr/bin/env bash
export PATH=~/miniconda3/bin:$PATH
#python /lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/drawPostProcessed_FCCD.py
#python /lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/drawPostProcessed_FCCD.py --simID sim-V05266A-ba_HS4-top-0r-81z_g --detector V05266A
#python /lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/drawPostProcessed_FCCD.py --simID sim2-V05266A-ba_HS4-top-0r-81z_g --detector V05266A


python /lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/drawPostProcessed_FCCD.py --simID IC160A-BA133-uncollimated-top-run0003-81z-newgeometry_g --detector I02160A --cuts True
