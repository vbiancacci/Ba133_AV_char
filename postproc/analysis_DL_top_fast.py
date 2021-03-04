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
        #print('Example usage: python analysis_DL_top.py lfs/l1/legend/users/aalexander/hdf5_output/detector_IC160A_ba_top_81mmNEW8_01.hdf5 detectors/I02160A/constants_I02160A.json g 0.74 0.5')
        print('Example usage: python analysis_DL_top.py /lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/raw-IC160A-BA133-uncollimated-top-run0003-81z-newgeometry.hdf5 IC160A-BA133-uncollimated-top-run0003-81z-newgeometry detectors/I02160A/constants_I02160A.json g 0.74 0.5')
        sys.exit()

    print("start...")

    MC_raw = sys.argv[1]    #inputfile - e.g. "/lfs/l1/legend/users/aalexander/hdf5_output/detector_IC160A_ba_top_81mmNEW8_01.hdf5"
    MC_file_id = sys.argv[2] #file id for saving output - e.g. IC160A-BA133-uncollimated-top-run0003-81z-newgeometry
    conf_path = sys.argv[3]     #detector geometry - e.g. detectors/I02160A/constants_I02160A.json
    smear=str(sys.argv[4])      #energy smearing (g/n) smear(g/g+l/n->gaussian/gaussian+lowenergy/none) - e.g. g
    fFCCD=float(sys.argv[5])    #FCCD thickness - e.g. 0.74
    fDLTp=float(sys.argv[6])    #DL fraction % - e.g. 0.5

    #MC_file_id = "IC160A_ba_top_81mmNEW8_01" #need to automate this part

    print("MC base file ID: ", MC_file_id)
    print("geometry conf_path: ", conf_path)
    print("resolution smearing: ", smear)
    print("FCCD: ", fFCCD)
    print("DLF: ", fDLTp)
    
    fDLT=fFCCD*fDLTp #dl thickness?

    #read config geometry for detector
    with open(conf_path) as json_file:
        geometry = json.load(json_file)
        r_c = geometry['r_c'] #cavity radius
        R_b = geometry['R_b'] #bottom crystal radius  (79.8/2)
        R_u = geometry['R_u'] #up crystal radius    (75.5/2 )
        h_c = geometry['h_c'] #cavity height
        H = geometry['H'] #crystal height
        H_u = geometry['H_u'] #up crystal height    (65.4-45.3)
        offset = geometry['offset'] #position of crystal from Alcap end

    #added this - needs checking
    radius = R_b
    height = H
    grooveOuterRadius = 31.0/2 #from my code, from xml file
    grooveInnerRadius = 22.6/2 #from my code, from xml file
    grooveDepth  = 2.0 #from my code, from xml file
    coneRadius  = R_u
    coneHeight = H_u
    boreRadius = r_c
    boreDepth = h_c

    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/"
                 
    #Open base MC file

    # # have to open the input file with h5py (g4 doesn't write pandas-ready hdf5)
    # #g4sfile = h5py.File(hdf5_path+'detector_'+MC_file_id+'.hdf5', 'r')
    # g4sfile = h5py.File(MC_raw, 'r')
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
    
    #if already combined MC pandas df, skip above, and open with pandas directly
    g4sdf = pd.read_hdf(MC_raw, key="procdf")

    # apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
    # write directly into output dataframe
    detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]
    keys = detector_hits.keys()
    no_events =  len(detector_hits) #73565535 rows x 8 columns, len = 73565535, size = 73565535x8
     
    #apply FCCD (DLT) cut ]
    detector_hits_FCCD = FCCD_cut(detector_hits, fFCCD, fDLT, conf_path)
    print("detector_hits_FCCD")
    print(detector_hits_FCCD)

    procdf = pd.DataFrame(detector_hits_FCCD.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
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

    #procdf.to_hdf(hdf5_path+'processed/valentina_script/processed_detector_'+MC_file_id+'_'+smear+'_FCCD'+str(fFCCD)+"mm_DLF"+str(fDLTp)+'_fast.hdf5', key='procdf', mode='w')
    procdf.to_hdf(hdf5_path+'raw_MC_combined/processed/processed_detector_'+MC_file_id+'_'+smear+'_FCCD'+str(fFCCD)+"mm_DLF"+str(fDLTp)+'.hdf5', key='procdf', mode='w')
    
    print("done")
    print("time elapsed: ")
    print(datetime.now() - t0)




def FCCD_cut(detector_hits,fFCCD,fDLT, conf_path):
    
    #get geometry constants

    #read config geometry for detector
    with open(conf_path) as json_file:
        geometry = json.load(json_file)
        r_c = geometry['r_c'] #cavity radius
        R_b = geometry['R_b'] #bottom crystal radius  (79.8/2)
        R_u = geometry['R_u'] #up crystal radius    (75.5/2 )
        h_c = geometry['h_c'] #cavity height
        H = geometry['H'] #crystal height
        H_u = geometry['H_u'] #up crystal height    (65.4-45.3)
        offset = geometry['offset'] #position of crystal from Alcap end

    #added this - needs checking
    radius = R_b
    height = H
    grooveOuterRadius = 31.0/2 #from my code, from xml file
    grooveInnerRadius = 22.6/2 #from my code, from xml file
    grooveDepth  = 2.0 #from my code, from xml file
    coneRadius  = R_u
    coneHeight = H_u
    boreRadius = r_c
    boreDepth = h_c
    
    
    #create vectors describing detector edges
    if(coneHeight==0):
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([radius,height]))], #bottom
            [TwoDLine(np.array([radius,height]),np.array([radius,0.]))], #side
            [TwoDLine(np.array([radius,0.]),np.array([boreRadius,0.]))], #top
            ])
        fNbore=np.array([
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

    #CCEs = list(map(GetCCE, [fNplus]*len(r),[fBore]*len(r),r,z_minusoffset, [fFCCD]*len(r), [fDLT]*len(r)))
    
    CCEs = GetCCEs(fNplus,fBore,r,z_minusoffset,fFCCD,fDLT)
    #print("CCEs: ", CCEs)
    CCEs = np.array(CCEs)
    Edep_FCCD = CCEs*Edep
    #print("Edep_FCCD: ", Edep_FCCD)
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
        print("no_points: ", no_points)
        p1, p2 = self.p1, self.p2
        p1s, p2s = np.array([self.p1]*no_points), np.array([self.p2]*no_points)
        lengths = np.array([self.length()]*no_points)
        # print("lengths.shape: ", lengths.shape)
        # print(lengths)
        #print("lengths.type: ", lengths.type)
        zeros, ones = np.zeros(no_points), np.ones(no_points)
        dot_products = np.sum((v-p1s)*(p2s-p1s), axis = 1) #= rirj+zizj for each point
        # print("dot_products.shape: ", dot_products.shape)
        #print("dot_products.type: ", dot_products.type)
        # dot_poducts_over_lengths = dot_products/lengths
        # dot_poducts_over_lengths = np.divide(dot_products,lengths)
        c = (np.maximum(zeros,np.minimum(dot_products/lengths,ones)))
        # print("c.shape: ", c.shape)
        # print("c[:,None].shape: ", (c[:,None]).shape)
        projections = p1s + (p2s-p1s)*c[:,None]
        # print("projections.shape: ", projections.shape)
        return projections

    def real_distances(self,v:np.array):
        #returns distances from projected 
        #v = array of points
        displacements = self.projections(v)-v
        real_distances = np.linalg.norm(displacements, axis = 1)
        # print("real_distances.shape: ", real_distances.shape)
        return real_distances


def GetMinimumDistances(chain,points:np.array):
    
    if (len(chain)==0): #what is this condition for?
        return 0

    # print("chain: ", chain)
    # print("chain[0][0]: ", chain[0][0])
    # print("len(chain): ", len(chain))
    # print("len(points)")
    # print(len(points))
    

    #distance=chain[0][0].real_distance(point)
    #distances = ([chain[0][0]]*len(point)).real_distance(point)
    #distances = [chain[0][0].real_distance(point_i) for point_i in point] #[r for r in relationship_list if r.diff > 1]
    
    real_distances = chain[0][0].real_distances(points)
    
    # print("len(real_distances)")
    # print(len(real_distances))
    # print("real_distances[0]")
    # print(real_distances[0])

    for entry in chain:
        # print("entry: ", entry)
        #distance=min(distance,entry[0].real_distance(point))
        #distances=[min(distance,entry[0].real_distance(point[index])) for index, distance in enumerate(distances)]
        real_distances = np.minimum(real_distances,entry[0].real_distances(points))

    # print("real_distances[0]")
    # print(real_distances[0])

    return real_distances
 

def GetDistancesToNPlus(fNPlus,r,z):
    points = np.array(list(zip(r, z)))
    print("points[0]: ", points[0])
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

    # print("fNplus")
    # print(fNplus)
    # print("fBore")
    # print(fBore)

    print("getting distances to Nplus...")
    distancesToNPlus=GetDistancesToNPlus(fNplus,r,z)
    print("getting distances to fBore...")
    distancesToBore=GetDistancesToBore(fBore,r,z)

    # minDists=np.minimum(distancesToBore,distancesToNPlus)
    # if (minDist < 0):
    #     return 0


    CCEs = np.minimum(FCCDBore(distancesToBore,fDLT,fFCCD),FCCDOuter(distancesToNPlus,fDLT,fFCCD))
   
    # print("CCEs.shape: ", CCEs.shape)

    return CCEs

   

if __name__=="__main__":
    main()

