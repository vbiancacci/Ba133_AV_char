import numpy as np
import pandas as pd
import math
import sys 
import h5py
import random
import glob
from datetime import datetime
import json

#FAST version of postproc code for FCCD and DLF implementation
#takes ~ 12mins per fccd/dlf configuration

def main():

    #print date and time for log:
    t0 = datetime.now()
    dt_string = t0.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    if(len(sys.argv) != 7):
        print('Example usage: python analysis_DL_top.py /lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/raw-IC160A-BA133-uncollimated-top-run0003-81z-newgeometry.hdf5 IC160A-BA133-uncollimated-top-run0003-81z-newgeometry detectors/I02160A/I02160A.json g 0.74 0.5')
        sys.exit()

    print("start...")

    MC_raw = sys.argv[1]    #inputfile - e.g. "/lfs/l1/legend/users/aalexander/hdf5_output/detector_IC160A_ba_top_81mmNEW8_01.hdf5"
    MC_file_id = sys.argv[2] #file id for saving output - e.g. IC160A-BA133-uncollimated-top-run0003-81z-newgeometry
    conf_path = sys.argv[3]     #detector geometry - e.g. detectors/I02160A/V02160A.json OR detectors/I02160A/constants_I02160A.json
    smear=str(sys.argv[4])      #energy smearing (g/n) smear(g/g+l/n->gaussian/gaussian+lowenergy/none) - e.g. g
    fFCCD=float(sys.argv[5])    #FCCD thickness - e.g. 0.74
    fDLTp=float(sys.argv[6])    #DL fraction % - e.g. 0.5

    print("MC base file ID: ", MC_file_id)
    print("geometry conf_path: ", conf_path)
    print("resolution smearing: ", smear)
    print("FCCD: ", fFCCD)
    print("DLF: ", fDLTp)
    
    fDLT=fFCCD*fDLTp #dl thickness (mm)

    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/" #general path to save large hdf5 files
                 
    #____Open base MC file - for single file MC___
    # # have to open the input file with h5py (g4 doesn't write pandas-ready hdf5)
    # g4sfile = h5py.File(MC_raw, 'r')
    # print(g4sfile.keys())
    # g4sntuple = g4sfile['default_ntuples']['g4sntuple']
    # g4sdf = pd.DataFrame(np.array(g4sntuple), columns=['event'])
    # # # build a pandas DataFrame from the hdf5 datasets we will use
    # g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
    # g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),lsuffix = '_caller', rsuffix = '_other')
    # g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),lsuffix = '_caller', rsuffix = '_other')
    # g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
    # g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['iRep']['pages']),columns=['iRep']), lsuffix = '_caller', rsuffix = '_other')
    # g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),columns=['x']), lsuffix = '_caller', rsuffix = '_other')
    # g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),columns=['y']), lsuffix = '_caller', rsuffix = '_other')
    # g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),columns=['z']), lsuffix = '_caller', rsuffix = '_other')
    

    #____Open base MC file - for combined MC files____ from using ../simulations/combine_simulations.py
    #if already combined MC pandas df, skip above, and open with pandas directly
    g4sdf = pd.read_hdf(MC_raw, key="procdf")

    # apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
    # write directly into output dataframe
    detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]
    print(detector_hits)
    keys = detector_hits.keys()
    no_hits =  len(detector_hits)
     
    #apply FCCD (DLT) cut
    detector_hits_FCCD = FCCD_cut(detector_hits, fFCCD, fDLT, conf_path)
    print("detector_hits_FCCD")
    print(detector_hits_FCCD)

    #procdf = pd.DataFrame(detector_hits_FCCD.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
    procdf = pd.DataFrame(detector_hits_FCCD.groupby(['event','volID','iRep', 'raw_MC_fileno'], as_index=False)['Edep'].sum())
    procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})
    procdf = procdf[procdf.energy!=0]    

    # apply energy resolution function - explain these?
    if (smear=='g' or smear=='G'):
        procdf['energy']=procdf['energy']*1000+(f_smear(procdf['energy']*1000))/2.355*np.random.randn(len(procdf['energy']))
    elif (smear=='g+l' or smear=='G+L'):
        procdf['energy']=procdf['energy']*1000+f_random(f_smear(procdf['energy']*1000)/2.355)
    else:
        procdf['energy']=procdf['energy']*1000
   
    print(procdf['energy'])

    procdf.to_hdf(hdf5_path+'raw_MC_combined/processed/processed_detector_'+MC_file_id+'_'+smear+'_FCCD'+str(fFCCD)+"mm_DLF"+str(fDLTp)+'.hdf5', key='procdf', mode='w')
    
    print("done")
    print("time elapsed: ")
    print(datetime.now() - t0)




def FCCD_cut(detector_hits,fFCCD,fDLT, conf_path):
    
    #get geometry constants- read config geometry for detector

    with open(conf_path) as json_file:
        json_geometry = json.load(json_file)
        geometry = json_geometry['geometry']

        R_b = radius_in_mm = geometry["radius_in_mm"] #crystal main/bottom radius
        H = height_in_mm = geometry["height_in_mm"] # = cryystal height 
        well_gap_in_mm = geometry["well"]["gap_in_mm"] #radius cavity
        r_c = well_radius_in_mm = geometry["well"]["radius_in_mm"] #radius cavity
        taper_top_outer_angle_in_deg = geometry["taper"]["top"]["outer"]["angle_in_deg"] 
        H_u = taper_top_outer_height_in_mm = geometry["taper"]["top"]["outer"]["height_in_mm"] #height of top conical part
        groove_outer_radius_in_mm =  geometry["groove"]["outer_radius_in_mm"]

    R_u = R_b - H_u*math.tan(taper_top_outer_angle_in_deg*np.pi/180) #radius of top crystal
    h_c = H - well_gap_in_mm #cavity height
    
    #these are the parameters required for code
    height = H
    radius = R_b
    coneRadius  = R_u
    coneHeight = H_u
    boreRadius = r_c
    boreDepth = h_c
    grooveOuterRadius = groove_outer_radius_in_mm
    print(grooveOuterRadius)
    #grooveInnerRadius = ?? - not currently used

    offset = 7. # NB: for V07XXXXX detectors the offset is defined in json files, otherwise it is 7mm

    #create vectors describing detector edges
    if(coneHeight==0):
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([radius,height]))], #bottom
            [TwoDLine(np.array([radius,height]),np.array([radius,0.]))], #side
            [TwoDLine(np.array([radius,0.]),np.array([boreRadius,0.]))], #top
            ])
        fBore=np.array([
            [TwoDLine(np.array([boreRadius,0.]),np.array([boreRadius,boreDepth]))], #top bore hole
            [TwoDLine(np.array([boreRadius,boreDepth]),np.array([0.,boreDepth]))], #top bore hole
            ])
    else:
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([radius,height]))], #bottom
            [TwoDLine(np.array([radius,height]),np.array([radius,coneHeight]))], #side
            [TwoDLine(np.array([radius,coneHeight]),np.array([coneRadius,0.]))], #tapper
            [TwoDLine(np.array([coneRadius,0.]),np.array([boreRadius,0.]))], #top
            ])
        fBore=np.array([
            [TwoDLine(np.array([boreRadius,0.]),np.array([boreRadius,boreDepth]))], #top bore hole
            [TwoDLine(np.array([boreRadius,boreDepth]),np.array([0.,boreDepth]))], #top bore hole
            ])
    
    #add an "r" column to df (r^2=x^2+y^2)
    r = np.sqrt((detector_hits['x'].to_numpy())**2 + (detector_hits['y'].to_numpy())**2) 
    detector_hits['r'] = r
    print("detector_hits with r: ", detector_hits)

    #first remove any errors/accidental deposits outside detector volume
    detector_hits = detector_hits.drop(detector_hits[detector_hits.r>radius].index)
    detector_hits = detector_hits.drop(detector_hits[detector_hits.z-offset>height].index)
    print("detector hits after removing accidental events outside detector volume:")
    print(detector_hits)

    #CCEs
    Edep = detector_hits.Edep
    r = detector_hits.r
    z_minusoffset = detector_hits.z - offset

    CCEs = GetCCEs(fNplus,fBore,r,z_minusoffset,fFCCD,fDLT)
    CCEs = np.array(CCEs)
    Edep_FCCD = CCEs*Edep
    detector_hits['Edep'] = Edep_FCCD
    print("detector_hits with Edep FCCD: ", detector_hits)

    return detector_hits


def f_smear(x):
    a=0.35 #0.27  #new values from ge-proc upgrades  #old value: 0.35
    b=1.99e-3   #2.08e-3                                  #old value: 1.99e-3
    return np.sqrt(a+b*x)


def length_np(v:np.array):
    return sum(v*v)


class TwoDLine():
    def __init__(self, p1:np.array, p2:np.array):
        self.p1=p1
        self.p2=p2
    
    def length(self):
        return sum((self.p1-self.p2)**2)
    
    def distance(self, v:np.array):
        return self.p1+(self.p2-self.p1)* max(0.,min(sum((v-self.p1)*(self.p2-self.p1))/self.length(),1.))
    
    def real_distance(self,v:np.array):
        return math.sqrt(length_np(self.distance(v)-v))
   
    def projections(self, v:np.array):
        #= projection, returns coordinates of projected point
        #v = array of points
        no_points = max(v.shape)
        p1, p2 = self.p1, self.p2
        p1s, p2s = np.array([self.p1]*no_points), np.array([self.p2]*no_points)
        lengths = np.array([self.length()]*no_points)
        zeros, ones = np.zeros(no_points), np.ones(no_points)
        dot_products = np.sum((v-p1s)*(p2s-p1s), axis = 1) #= rirj+zizj for each point
        c = (np.maximum(zeros,np.minimum(dot_products/lengths,ones)))
        projections = p1s + (p2s-p1s)*c[:,None]
        return projections

    def real_distances(self,v:np.array):
        #returns distances from projected 
        #v = array of points
        displacements = self.projections(v)-v
        real_distances = np.linalg.norm(displacements, axis = 1)
        return real_distances


def GetMinimumDistances(chain,points:np.array):
    
    if (len(chain)==0): #what is this condition for?
        return 0

    real_distances = chain[0][0].real_distances(points)
    
    for entry in chain:
        real_distances = np.minimum(real_distances,entry[0].real_distances(points))

    return real_distances
 

def GetDistancesToNPlus(fNPlus,r,z):
    points = np.array(list(zip(r, z)))
    return GetMinimumDistances(fNPlus,points)

def GetDistancesToBore(fBore,r,z):
    points = np.array(list(zip(r, z)))
    return GetMinimumDistances(fBore,points)


def FCCDBore(x,fDLT,fFCCD):

    #x= distances to fBore

    #initialise array of CCEs
    CCEs = np.ones(x.shape[0])

    #Set CCE=0 for events in DL
    CCEs = np.where(x <= fDLT/2, 0, CCEs)

    #Set linear model CCE for events in TL
    if fDLT != fFCCD: #DLF is not 1, i.e. there is a TL
        CCEs = np.where((x>fDLT/2)&(x<fFCCD/2), 2./(fFCCD-fDLT)*x-fDLT/(fFCCD-fDLT), CCEs)

    return CCEs

def FCCDOuter(x,fDLT,fFCCD):

    #x = distances to NPlus

    #initialise array of CCEs
    CCEs = np.ones(x.shape[0])

    #Set CCE=0 for events in DL
    CCEs = np.where(x <= fDLT, 0, CCEs)

    #Set linear model CCE for events in TL
    if fDLT != fFCCD: #DLF is not 1, i.e. there is a TL
        CCEs = np.where((x>fDLT)&(x<fFCCD), 1./(fFCCD-fDLT)*x-fDLT/(fFCCD-fDLT), CCEs)

    return CCEs

def GetCCEs(fNplus,fBore,r,z,fFCCD,fDLT):
    print(fBore)
    print("getting distances to Nplus...")
    distancesToNPlus=GetDistancesToNPlus(fNplus,r,z)
    print("getting distances to fBore...")
    distancesToBore=GetDistancesToBore(fBore,r,z)

    # minDists=np.minimum(distancesToBore,distancesToNPlus)
    # if (minDist < 0):
    #     return 0

    print("getting CCEs...")
    CCEs = np.minimum(FCCDBore(distancesToBore,fDLT,fFCCD),FCCDOuter(distancesToNPlus,fDLT,fFCCD))

    return CCEs

   

if __name__=="__main__":
    main()

