import numpy as np
import pandas as pd
import math
import h5py
import random
import glob
from datetime import datetime
import json
import sys
sys.path.append('../data/')
from Ba133_data_AV_analysis import * 

#Simple script to combine the N MC g4simple hdf5 files into 1 super file

def main():


    if(len(sys.argv) != 3):
        print('Example usage: python combine_simulations.py /lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/IC-legend/IC160A/Ba133/uncollimated/top/ raw-IC160A-BA133-uncollimated-top-run0003-81z-newgeometry')
        #print('Example usage: python combine_simulations.py /lfs/l1/legend/users/bianca/legend-g4simple-simulation/legend/simulations/V05266A/ba_HS4/top_0r_81z/hdf5/ sim-V05266A-ba_HS4-top-0r-81z')
        sys.exit()

    raw_MC_hdf5_path = sys.argv[1] #path/folder containing the N MC files
    MC_file_ID = sys.argv[2] #unique identifier to be saved

    #read in each hdf5 file
    files = os.listdir(raw_MC_hdf5_path)
    files = fnmatch.filter(files, "*.hdf5")
    df_list = []
    for file in files:

        print("file: ", str(file))
        file_no = file[-7]+file[-6]
        print("raw MC file_no: ", file_no)

        g4sfile = h5py.File(raw_MC_hdf5_path+file, 'r')
        # print("g4sfile: ", g4sfile)
        # print(g4sfile.keys())

        g4sntuple = g4sfile['default_ntuples']['g4sntuple']
        g4sdf = pd.DataFrame(np.array(g4sntuple), columns=['event'])

        # # build a pandas DataFrame from the hdf5 datasets we will use
        g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['iRep']['pages']),columns=['iRep']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),columns=['x']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),columns=['y']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),columns=['z']), lsuffix = '_caller', rsuffix = '_other')

        #add new column to each df for the raw MC file no
        g4sdf["raw_MC_fileno"] = file_no
        print(g4sdf)

        df_list.append(g4sdf)

    #concatonate
    df_total = pd.concat(df_list, axis=0, ignore_index=True)
    print(df_total)

    #write output
    output_path = "/lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/"
    df_total.to_hdf(output_path+MC_file_ID+'.hdf5', key='procdf', mode='w')
    

if __name__=="__main__":
    main()
