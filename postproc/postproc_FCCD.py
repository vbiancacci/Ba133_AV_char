import sys, h5py
import pandas as pd
import numpy as np
from datetime import datetime
import argparse


#THIS IS OLD CODE - NOW REPLACED BY analysis_DL_top_fast.py

def main():  

    #print date and time for log:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    parser = argparse.ArgumentParser(description='FCCD postproc for given base simulation')
    parser.add_argument('--simID', action="store_true", default="IC160A_ba_top_81mmNEW8_01")
    parser.add_argument('--detector', action="store_true", default="I02160A")
    args = parser.parse_args()
    MC_file_id, detector = args.simID, args.detector
    print("MC file ID: ", MC_file_id)
    print("detector: ", detector)
    print("")

    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/"

    # have to open the input file with h5py (g4 doesn't write pandas-ready hdf5)
    g4sfile = h5py.File(hdf5_path+'detector_'+MC_file_id+'.hdf5', 'r')
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

    # apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
    # write directly into output dataframe
    detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]
    print(detector_hits)
    keys = detector_hits.keys()
    no_events =  len(detector_hits) #73565535 rows x 8 columns, len = 73565535, size = 73565535x8
    print(no_events)

    #first save original file, i.e. no FCCD
    procdf = pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
    procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})
    # apply energy resolution function
    A,c = 0.039, 287.446 #/1000 #coeficcients from Ba133 resolution graph, keV
    procdf = resolution(procdf, A, c)
    print(procdf)
    procdf.to_hdf(hdf5_path+'processed/processed_detector_'+MC_file_id+'_newresolution.hdf5', key='procdf', mode='w')
    

    #apply FCCD cuts
    FCCD_list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3] #mm
    #FCCD_list = [0.744] #best fit, no cuts
    #FCCD_list = [0.698] #best fit, cuts
    
    for FCCD in FCCD_list:

        detector_hits_FCCD = FCCD_cut(FCCD, detector_hits)
        print('FCCD: ', str(FCCD))
        print(detector_hits_FCCD.size)
        
        procdf = pd.DataFrame(detector_hits_FCCD.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
        procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})
        # apply energy resolution functionA,c = 0.039, 287.446 #/1000 #coeficcients from Ba133 resolution graph, keV
        procdf = resolution(procdf, A, c)
        print(procdf.size)
        #procdf.to_hdf(hdf5_path+'processed/processed_detector_'+MC_file_id+'_FCCD'+str(FCCD)+'mm.hdf5', key='procdf', mode='w')
        procdf.to_hdf(hdf5_path+'processed/processed_detector_'+MC_file_id+'_newresolution_FCCD'+str(FCCD)+'mm.hdf5', key='procdf', mode='w')


    print("done")

def resolution(procdf, A, c):
    "apply resolution function"

    #OLD Resolution function
    ##Modify this value for different energy resolution
    #pctResAt1MeV = 0.15 #0.15 #5#;
    #procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*pctResAt1MeV/100.*np.random.randn(len(procdf['energy'])) #old res
    
    #NEW Resolution function
    procdf['energy'] = procdf['energy'] + (1/1000)*A*np.sqrt(1000*procdf['energy']+c)*np.random.randn(len(procdf['energy']))/2.355
    return procdf


def FCCD_cut(FCCD, detector_hits):

    #define geometry
    position_crystal_from_top = 7.0 #all in mm, taken from gdml files
    crystal_height = 65.4
    bottom_height = 45.3
    cavity_height = 33.7
    groove_height = 2.0
    top_rad = 75.5/2 #top_width/2 
    bottom_rad = 79.8/2
    cavity_rad = 9.3/2
    groove_inner_width = 22.6 #ignore grooves for now
    groove_outer_width = 31.0


    #region 1 geometry
    r = bottom_rad - FCCD
    l = crystal_height - bottom_height - FCCD
    H = position_crystal_from_top + FCCD
    A = top_rad - FCCD
    B = bottom_rad - FCCD
    z0 = H - l*A/(B-A)
    h = l + H - z0

    #r_cylinder = cavity_rad + FCCD
    r_cylinder = cavity_rad + 0.5*FCCD #new 25/11/20 - half DL at bore hole


    #region 2 geometry
    l2 = cavity_height - (l+H)

    #region 3 geometry
    l3 = bottom_height - l2 - FCCD


    #divide detector volume into 3 different regions

    detector_hits_1 = detector_hits.loc[(detector_hits.z<l+H)&(detector_hits.z>H)]#region 1
    print("len region 1: ", len(detector_hits_1))
    detector_hits_FCCD_1 = detector_hits_1.loc[((detector_hits_1.x)**2 + (detector_hits_1.y)**2 < (r**2/h**2)*(detector_hits_1.z-z0)**2)&((detector_hits_1.x)**2 + (detector_hits_1.y)**2 > r_cylinder**2)]
    

    detector_hits_2 = detector_hits.loc[(detector_hits.z<l2+l+H)&(detector_hits.z>l+H)]#region 2
    print("len region 2: ", len(detector_hits_2))
    detector_hits_FCCD_2 = detector_hits_2.loc[((detector_hits_2.x)**2 + (detector_hits_2.y)**2 < B**2)&((detector_hits_2.x)**2 + (detector_hits_2.y)**2 > r_cylinder**2)]

    detector_hits_3 = detector_hits.loc[(detector_hits.z<l+H+l2+l3)&(detector_hits.z>l+H+l2)]#region 3
    print("len region 3: ", len(detector_hits_3))
    detector_hits_FCCD_3 = detector_hits_3.loc[((detector_hits_3.x)**2 + (detector_hits_3.y)**2 < B**2)]

    detector_hits_FCCD_LIST = [detector_hits_FCCD_1, detector_hits_FCCD_2, detector_hits_FCCD_3]

    detector_hits_FCCD = pd.concat(detector_hits_FCCD_LIST, axis=0, ignore_index=True)
    print(detector_hits_FCCD)
    return detector_hits_FCCD


if __name__ == "__main__":
    main()