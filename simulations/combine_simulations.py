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
        sys.exit()

    raw_MC_hdf5_path = sys.argv[1] #path/folder containing the N MC files
    MC_file_ID = sys.argv[2] #unique identifier to be saved

    #read in each hdf5 file
    files = os.listdir(raw_MC_hdf5_path)
    files = fnmatch.filter(files, "*.hdf5")
    df_list = []
    g4sfile_list = []
    for file in files:

        print("file: ", str(file))
        g4sfile = h5py.File(raw_MC_hdf5_path+file, 'r')
        g4sfile_list.append(g4sfile_list)
        print("g4sfile: ", g4sfile)
        print(g4sfile.keys())

        g4sntuple = g4sfile['default_ntuples']['g4sntuple']
        print(g4sntuple)
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

        print(g4sdf)

        df_list.append(g4sdf)

    #concatonate
    df_total = pd.concat(df_list, axis=0, ignore_index=True)
    print(df_total)


    #write output
    output_path = "/lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/"


    with h5py.File(output_path+MC_file_ID,mode='w') as h5fw:
        link_cnt = 0 
        #for h5name in glob.glob(str(raw_MC_hdf5_path)):
        for file_name in files:
            link_cnt += 1
            h5fw['link'+str(link_cnt)] = h5py.ExternalLink(file_name,'/')   


    #df_total.to_hdf(output_path+MC_file_ID+'.hdf5', key='procdf', mode='w')
    

if __name__=="__main__":
    main()
