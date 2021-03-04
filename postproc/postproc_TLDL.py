import sys, h5py
import pandas as pd
import numpy as np
from datetime import datetime

#THIS IS OLD CODE - NOW REPLACED BY analysis_DL_top_fast.py

def main():  

    #print date and time for log:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/"

    if(len(sys.argv) != 2):
        print('Usage: postprochdf5.py [IC160A_ba_top_81mmNEW8_01]')
        sys.exit()

    MC_file_id = sys.argv[1]

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
    print("detector_hits: ", detector_hits)
    keys = detector_hits.keys()
    no_events =  len(detector_hits) #73565535 rows x 8 columns, len = 73565535, size = 73565535x8
    print("no_events: ", no_events)

    #first save original file, i.e. no FCCD
    # procdf = pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
    # procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})
    # # apply energy resolution function
    # procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*pctResAt1MeV/100.*np.random.randn(len(procdf['energy']))
    # print(procdf.size)
    # print(procdf)

    #procdf.to_hdf(hdf5_path+'processed/processed_detector_'+MC_file_id+'.hdf5', key='procdf', mode='w')


    #apply FCCD cuts
    FCCD = 0.75 #mm, fixed
    #DLF_list = np.linspace(0,1,6) #e.g [0,0.1,0.2...1]
    DLF_list = [0,0.2,0.4,0.5,0.6,0.8,1]
    #DLF_list = [0,0.2,0.4,0.5,0.6,0.8]


    print("")
    print("FCCD (fixed): ", str(FCCD))
    print("DLF_list: ", DLF_list)
    

    #for testing
    #DLF_list = [0.5]

    for DLF in DLF_list:

        print("")
        print('DLF: ', str(DLF))
        DL = DLF*FCCD
        print('DL: ', str(DL))
        TL = FCCD-DL
        print('TL: ', str(TL))


        detector_hits_DLTL = DLTL_cut(FCCD,DL,TL, detector_hits)
        #print(detector_hits_DLTL.size)
        #print(detector_hits_DLTL.loc[30132,:])
        
        
        procdf = pd.DataFrame(detector_hits_DLTL.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
        procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})
        print("procdf dltl after grouping")
        print(procdf)
        # apply energy resolution function
        A,c = 0.039, 287.446 #/1000 #coeficcients from Ba133 resolution graph, keV
        procdf['energy'] = procdf['energy'] + (1/1000)*A*np.sqrt(1000*procdf['energy']+c)*np.random.randn(len(procdf['energy']))/2.355
        print("procdf dltl after energy res")
        print(procdf)
        #print(procdf.size)
        procdf.to_hdf(hdf5_path+'processed/processed_detector_'+MC_file_id+'_newresolution_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'.hdf5', key='procdf', mode='w')


    print("done")



def DLTL_cut(FCCD,DL,TL,detector_hits):

    #define geometry - eventually automate this and take as argument of a json file
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
    l1 = crystal_height - bottom_height
    H = position_crystal_from_top
    A = top_rad
    A_FCCD = A - FCCD
    A_TL = A - DL
    B = bottom_rad
    B_FCCD = B - FCCD
    B_TL = B - DL
    #z0 = H - l1*A/(B-A)
    #h = l1 + H - z0
    h = l1*B/(B-A)
    h_FCCD = B_FCCD*(h/B)
    h_TL = B_TL*(h/B)
    z0 = l1 + H - h
    z0_FCCD = z0 + FCCD
    z0_TL = z0 + DL

    r_cavity = cavity_rad 
    r_cavity_FCCD = r_cavity + 0.5*FCCD #new 25/11/20 - half DL at bore hole
    r_cavity_TL = r_cavity + 0.5*DL

    #region 2 geometry
    l2 = cavity_height - (l1+H)

    #region 3 geometry
    l3 = bottom_height - l2

    r = np.sqrt((detector_hits['x'].to_numpy())**2 + (detector_hits['y'].to_numpy())**2) 
    #print("r: ", r)
    detector_hits['r'] = r
    print("detector_hits with r: ", detector_hits)

    #divide detector volume into 3 different z regions

    #_____REGION 1_______

    detector_hits_1 = detector_hits.loc[(detector_hits.z<l1+H)&(detector_hits.z>H)]#region 1 by z
    print("len region 1: ", len(detector_hits_1))

    #First define FAV, with whole FCCD removed
    #detector_hits_FAV_1 = detector_hits_1.loc[(detector_hits_1.z>H+FCCD)&((detector_hits_1.x)**2 + (detector_hits_1.y)**2 < (B_FCCD**2/h_FCCD**2)*(detector_hits_1.z-z0_FCCD)**2)&((detector_hits_1.x)**2 + (detector_hits_1.y)**2 > r_cavity_FCCD**2)]
    detector_hits_FAV_1 = detector_hits_1.loc[(detector_hits_1.z>H+FCCD)&((detector_hits_1.r)**2 < (B_FCCD**2/h_FCCD**2)*(detector_hits_1.z-z0_FCCD)**2)&((detector_hits_1.r)**2 > r_cavity_FCCD**2)]
    print("len detector_hits_FAV_1: ", len(detector_hits_FAV_1))

    #define TL strip
    #detector_hits_TL_1_top = detector_hits_1.loc[(detector_hits_1.z>H+DL)&(detector_hits_1.z<H+FCCD)&(r_cavity_TL<(detector_hits_1.x)**2+(detector_hits_1.y)**2)&((detector_hits_1.x)**2+(detector_hits_1.y)**2<A_TL)]
    detector_hits_TL_1_top = detector_hits_1.loc[(detector_hits_1.z>H+DL)&(detector_hits_1.z<H+FCCD)&(r_cavity_TL<(detector_hits_1.r)**2)&((detector_hits_1.r)**2<A_TL)].copy()
    print("len detector_hits_TL_1_top: ", len(detector_hits_TL_1_top))

    print("detector_hits_TL_1_top: ",detector_hits_TL_1_top)
    TL_depths = (detector_hits_TL_1_top['z'].to_numpy())-H-DL
    print("len TL_depths: ", len(TL_depths))
    #print("TL_depths: ", TL_depths)
    # print("TL_depths max: ", max(TL_depths))
    # print("TL_depths min: ", min(TL_depths))
    CCEs = CCE_weightings(TL_depths, TL)
    print("CCEs: ", CCEs)
    print("CCEs: ", CCEs["CCE"].to_numpy())
    print("CCEs max: ", max(CCEs["CCE"].to_numpy()))
    print("CCEs min: ", min(CCEs["CCE"].to_numpy()))
    print("len CCEs: ",len(CCEs))
    Edep = (detector_hits_TL_1_top["Edep"].to_numpy())*(CCEs["CCE"].to_numpy())
    #print("EDEP: ", Edep)
    print("len edep: ",len(Edep))
    #detector_hits_TL_1_top["Edep"] = Edep
    detector_hits_TL_1_top.loc[:,"Edep"] = Edep
    print("detector_hits_TL_1_top after multiplication: ", detector_hits_TL_1_top)
    print("len detector_hits_TL_1_top after multiplication: ", len(detector_hits_TL_1_top))

    #detector_hits_TL_1_side = detector_hits_1.loc[((detector_hits_1.x)**2 + (detector_hits_1.y)**2 < (B_TL**2/h_TL**2)*(detector_hits_1.z-z0_TL)**2)&((detector_hits_1.x)**2 + (detector_hits_1.y)**2 < (B_FCCD**2/h_FCCD**2)*(detector_hits_1.z-z0_FCCD)**2)]
    detector_hits_TL_1_side = detector_hits_1.loc[((detector_hits_1.r)**2 < (B_TL**2/h_TL**2)*(detector_hits_1.z-z0_TL)**2)&((detector_hits_1.r)**2 > (B_FCCD**2/h_FCCD**2)*(detector_hits_1.z-z0_FCCD)**2)].copy()
    print("len detector_hits_TL_1_side: ", len(detector_hits_TL_1_side))
    TL_depths = (B_TL/h_TL)*(detector_hits_TL_1_side['z'].to_numpy()-z0_TL) - detector_hits_TL_1_side['r'].to_numpy()
    print("len TL_depths: ", len(TL_depths))
    CCEs = CCE_weightings(TL_depths, TL)
    print("len CCEs: ",len(CCEs))
    Edep = (detector_hits_TL_1_side["Edep"].to_numpy())*(CCEs["CCE"].to_numpy())
    print("len edep: ",len(Edep))
    detector_hits_TL_1_side["Edep"] = Edep
    print("len detector_hits_TL_1_side after multiplication: ", len(detector_hits_TL_1_side))

    #detector_hits_TL_1_cavity = detector_hits_1.loc[((detector_hits_1.x)**2 + (detector_hits_1.y)**2 < r_cavity_FCCD)&((detector_hits_1.x)**2 + (detector_hits_1.y)**2 > r_cavity_TL**2)]
    detector_hits_TL_1_cavity = detector_hits_1.loc[((detector_hits_1.r)**2 < r_cavity_FCCD)&((detector_hits_1.r)**2 > r_cavity_TL**2)].copy()
    print("len detector_hits_TL_1_cavity: ", len(detector_hits_TL_1_cavity))
    detector_hits_TL_1_cavity_mod = detector_hits_1.loc[((detector_hits_1.r)**2 < r_cavity_FCCD)]
    print("len detector_hits_TL_1_cavity_mod: ", len(detector_hits_TL_1_cavity))
    TL_depths = (detector_hits_TL_1_cavity['r'].to_numpy())-r_cavity_TL
    print("len TL_depths: ", len(TL_depths))
    CCEs = CCE_weightings(TL_depths, TL)
    print("len CCEs: ",len(CCEs))
    Edep = (detector_hits_TL_1_cavity["Edep"].to_numpy())*(CCEs["CCE"].to_numpy())
    print("len edep: ",len(Edep))
    detector_hits_TL_1_cavity["Edep"] = Edep
    print("len detector_hits_TL_1_cavity after multiplication: ", len(detector_hits_TL_1_cavity))
    

    #_____REGION 2______

    detector_hits_2 = detector_hits.loc[(detector_hits.z<l2+l1+H)&(detector_hits.z>l1+H)]#region 2
    print("len region 2: ", len(detector_hits_2))

    
    #detector_hits_FAV_2 = detector_hits_2.loc[((detector_hits_2.x)**2 + (detector_hits_2.y)**2 < B_FCCD**2)&((detector_hits_2.x)**2 + (detector_hits_2.y)**2 > r_cavity_FCCD**2)]
    detector_hits_FAV_2 = detector_hits_2.loc[((detector_hits_2.r)**2 < B_FCCD**2)&((detector_hits_2.r)**2 > r_cavity_FCCD**2)]
    print("len detector_hits_FAV_2: ", len(detector_hits_FAV_2))

    #detector_hits_TL_2_side = detector_hits_2.loc[((detector_hits_2.x)**2 + (detector_hits_2.y)**2 > B_FCCD**2)&((detector_hits_2.x)**2 + (detector_hits_2.y)**2 < B_TL**2)]
    detector_hits_TL_2_side = detector_hits_2.loc[((detector_hits_2.r)**2 > B_FCCD**2)&((detector_hits_2.r)**2 < B_TL**2)].copy()
    print("len detector_hits_TL_2_side: ", len(detector_hits_TL_2_side))
    TL_depths = B_TL - (detector_hits_TL_2_side['r'].to_numpy())
    print("len TL_depths: ", len(TL_depths))
    CCEs = CCE_weightings(TL_depths, TL)
    print("len CCEs: ",len(CCEs))
    Edep = (detector_hits_TL_2_side["Edep"].to_numpy())*(CCEs["CCE"].to_numpy())
    print("len edep: ",len(Edep))
    detector_hits_TL_2_side["Edep"] = Edep
    print("len detector_hits_TL_2_side after multiplication: ", len(detector_hits_TL_2_side))

    #detector_hits_TL_2_cavity = detector_hits_2.loc[((detector_hits_2.x)**2 + (detector_hits_2.y)**2 < r_cavity_FCCD**2)&((detector_hits_2.x)**2 + (detector_hits_2.y)**2 > r_cavity_TL**2)]
    detector_hits_TL_2_cavity = detector_hits_2.loc[((detector_hits_2.r)**2 < r_cavity_FCCD**2)&((detector_hits_2.r)**2 > r_cavity_TL**2)].copy()
    print("len detector_hits_TL_2_cavity: ", len(detector_hits_TL_2_cavity))
    TL_depths = (detector_hits_TL_2_cavity['r'].to_numpy()) - r_cavity_TL
    print("len TL_depths: ", len(TL_depths))
    CCEs = CCE_weightings(TL_depths, TL)
    print("len CCEs: ",len(CCEs))
    Edep = (detector_hits_TL_2_cavity["Edep"].to_numpy())*(CCEs["CCE"].to_numpy())
    print("len edep: ",len(Edep))
    detector_hits_TL_2_cavity["Edep"] = Edep
    print("len detector_hits_TL_2_cavity after multiplication: ", len(detector_hits_TL_2_cavity))
    
    
    #______REGION 3______
    
    detector_hits_3 = detector_hits.loc[(detector_hits.z<l1+H+l2+l3)&(detector_hits.z>l1+H+l2)]#region 3
    print("len region 3: ", len(detector_hits_3))
    
    #detector_hits_FAV_3 = detector_hits_3.loc[(detector_hits_3.z<H+l1+l2+l3-FCCD)&((detector_hits_3.x)**2 + (detector_hits_3.y)**2 < B_FCCD**2)]
    detector_hits_FAV_3 = detector_hits_3.loc[(detector_hits_3.z<H+l1+l2+l3-FCCD)&((detector_hits_3.r)**2< B_FCCD**2)]
    print("len detector_hits_FAV_3: ", len(detector_hits_FAV_3))

    #detector_hits_TL_3_side = detector_hits_3.loc[((detector_hits_3.x)**2 + (detector_hits_3.y)**2 > B_FCCD**2)&((detector_hits_3.x)**2 + (detector_hits_3.y)**2 < B_TL**2)]
    detector_hits_TL_3_side = detector_hits_3.loc[((detector_hits_3.r)**2 > B_FCCD**2)&((detector_hits_3.r)**2 < B_TL**2)].copy()
    print("len detector_hits_TL_3_side: ", len(detector_hits_TL_3_side))
    TL_depths = B_TL - (detector_hits_TL_3_side['r'].to_numpy())
    print("len TL_depths: ", len(TL_depths))
    CCEs = CCE_weightings(TL_depths, TL)
    print("len CCEs: ",len(CCEs))
    Edep = (detector_hits_TL_3_side["Edep"].to_numpy())*(CCEs["CCE"].to_numpy())
    print("len edep: ",len(Edep))
    detector_hits_TL_3_side["Edep"] = Edep
    print("len detector_hits_TL_3_side after multiplication: ", len(detector_hits_TL_3_side))


    #detector_hits_TL_3_bottom = detector_hits_3.loc[(detector_hits_3.z>l1+l2+l3+H-FCCD)&(detector_hits_3.z<l1+l2+l3+H-DL)&((detector_hits_3.r)**2<B_TL**2)]
    detector_hits_TL_3_bottom = detector_hits_3.loc[(detector_hits_3.z>l1+l2+l3+H-FCCD)&(detector_hits_3.z<l1+l2+l3+H-DL)&((detector_hits_3.r)**2<B_TL**2)].copy()
    print("len detector_hits_TL_3_bottom: ", len(detector_hits_TL_3_bottom))
    TL_depths =  (l1+l2+l3+H-DL) -  (detector_hits_TL_3_bottom['z'].to_numpy())
    print("len TL_depths: ", len(TL_depths))
    CCEs = CCE_weightings(TL_depths, TL)
    print("len CCEs: ",len(CCEs))
    Edep = (detector_hits_TL_3_bottom["Edep"].to_numpy())*(CCEs["CCE"].to_numpy())
    print("len edep: ",len(Edep))
    detector_hits_TL_3_bottom["Edep"] = Edep
    print("len detector_hits_TL_3_bottom after multiplication: ", len(detector_hits_TL_3_bottom))



    #_____Combine all regions_____
    detector_hits_DLTL_LIST = [detector_hits_FAV_1, detector_hits_TL_1_top, detector_hits_TL_1_side, detector_hits_TL_1_cavity, detector_hits_FAV_2, detector_hits_TL_2_side, detector_hits_TL_2_cavity, detector_hits_FAV_3, detector_hits_TL_3_side, detector_hits_TL_3_bottom]

    detector_hits_DLTL = pd.concat(detector_hits_DLTL_LIST, axis=0, ignore_index=True)
    print("len combined detector_hits_DLTL: ", len(detector_hits_DLTL))
    print("detector_hits_DLTL: ", detector_hits_DLTL)

    return detector_hits_DLTL

def CCE_weightings(depths, TL):

    if TL != 0:
        CCEs = (1/TL)*depths #y=mx+c, can be single number or a numpy list/array
    else:
        print("len depths: ",len(depths))
        CCEs = np.ones(len(depths))
        CCEs = np.zeros(len(depths))
        print("TL = 0: CCES = ", CCEs)

    CCEs = pd.DataFrame({"CCE":CCEs})
    return CCEs

if __name__ == "__main__":
    main()