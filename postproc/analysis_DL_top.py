import numpy as np
import pandas as pd
#import vectormath as vmath
import math
import sys 
#import yaml - cant install
import h5py
import random
import glob
#import uproot - cant install
from pprint import pprint
from datetime import datetime
import json
#from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TMath



def main():

    #print date and time for log:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    if(len(sys.argv) != 6):
        print('Usage: python3 test.py raw-IC160A-Am241-collimated-top-run0003-4z-hdf5-01.hdf5 constants_IC160A.conf smear(g/g+l/n->gaussian/gaussian+lowenergy/none) FCCD DLT(%)')
        sys.exit()

    print("start...")

    MC_file_id = sys.argv[1]    #inputfile - e.g. "IC160A_ba_top_81mmNEW8_01"
    conf_path = sys.argv[2]     #detector geometry - e.g. detectors/I02160A/constants_I02160A.json
    smear=str(sys.argv[3])      #energy smearing (g/n) - e.g. g
    fFCCD=float(sys.argv[4])    #FCCD thickness - e.g. 0.74
    fDLTp=float(sys.argv[5])    #DL fraction - e.g. 0.5

    print("MC base file ID: ", MC_file_id)
    print("geometry conf_path: ", conf_path)
    print("resolution smearing: ", smear)
    print("FCCD: ", fFCCD)
    print("DLF: ", fDLTp)
    
    fDLT=fFCCD*fDLTp #dl thickness?


    #outfile=str(fFCCD).replace(".","")+"fccd_"+str(fDLTp).replace(".","")+"dlt"
    #sys.stdout = open("/lfs/l1/legend/users/bianca/IC_geometry/IC-legend/analysis/ba_top/analysis/log/"+outfile+".out", "w")

    # with open(conf_path) as fid:
    #     config = yaml.load(fid, Loader=yaml.Loader)
    # pprint(config)

    # radius                = config['radius']
    # height                = config['height']
    # grooveOuterRadius     = config['grooveOuterRadius']
    # grooveInnerRadius     = config['grooveInnerRadius']
    # grooveDepth           = config['grooveDepth']
    # coneRadius            = config['coneRadius']
    # coneHeight            = config['coneHeight']
    # boreRadius            = config['boreRadius']
    # boreDepth             = config['boreDepth']


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
                 

    #Open base MC file
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
    keys = detector_hits.keys()
    no_events =  len(detector_hits) #73565535 rows x 8 columns, len = 73565535, size = 73565535x8
     
    #apply FCCD (DLT) cut 
    detector_hits_FCCD = FCCD_cut(detector_hits, fFCCD, fDLT, conf_path)
    print('FCCD: ', str(fFCCD))
    print('DLT: ', str(fDLTp),"%")
    # print(detector_hits_FCCD.size)

    procdf = pd.DataFrame(detector_hits_FCCD.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
    procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})
    procdf = procdf[procdf.energy!=0]    

    # apply energy resolution function - explain these?
    if (smear=='g' or smear=='G'):
        #print(procdf['energy'])
        procdf['energy']=procdf['energy']*1000+(f_smear(procdf['energy']*1000))/2.355*np.random.randn(len(procdf['energy']))
    elif (smear=='g+l' or smear=='G+L'):
        procdf['energy']=procdf['energy']*1000+f_random(f_smear(procdf['energy']*1000)/2.355)
    else:
        procdf['energy']=procdf['energy']*1000
   
    print(procdf['energy'])
    print(procdf['energy'][1764])
    print(procdf['energy'][1765])
    print(procdf['energy'][1766])

    procdf.to_hdf(hdf5_path+'processed/valentina_script/processed_detector_'+MC_file_id+'_'+smear+'_FCCD'+str(fFCCD)+"mm_DLF"+str(fDLTp)+'.hdf5', key='procdf', mode='w')
    #procdf.to_hdf('output/processed_FCCD_'+MC_file_id+'_'+smear+'_'+str(fFCCD).replace(".","")+'fccd_'+str(fDLTp).replace(".","")+'dlt.hdf5', key='procdf', mode='w')
    print("done")




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
    
    r_hits=np.sqrt(detector_hits.x**2+detector_hits.y**2)
    z_hits=detector_hits.z-7.0
    energy_hits=detector_hits.Edep
    
    energy_hits_FCCD=[]
    for r,z,energy,i in zip(r_hits,z_hits,energy_hits,range(len(energy_hits))):
        if (r>radius or z<0 or z>height):
           detector_hits=detector_hits.drop(detector_hits.index[i]) 
           print("error ", r, z)
           # return 0
        else:
            energy_FCCD=GetChargeCollectionEfficiency(fNplus,fBore,r,z,fFCCD,fDLT)*energy
            energy_hits_FCCD.append(energy_FCCD)
        
    detector_hits_FCCD=detector_hits[['event','step','volID','iRep','x','y','z']]
    detector_hits_FCCD['Edep']=np.array(energy_hits_FCCD)
    #with open("output.txt", "w") as text_file_2:
    #    for i in range(len(energy_hits)):
    #        text_file_2.write("%f \t %f \t %f \t %f \t %f \n" % (i,energy_hits.iloc[i],energy_hits_FCCD[i],detector_hits.Edep.iloc[i],detector_hits_FCCD.Edep.iloc[i]))


    return detector_hits_FCCD


def f_smear(x):
    a=0.35 #0.27  #new values from ge-proc upgrades  #old value: 0.35
    b=1.99e-3   #2.08e-3                                  #old value: 1.99e-3
    return np.sqrt(a+b*x)


def FCCDBore(x,fDLT,fFCCD):
    if(x<=fDLT/2):
        return 0.*x
    elif(fDLT!=fFCCD and x>fDLT/2. and x<fFCCD/2.):
        return 2./(fFCCD-fDLT)*x-fDLT/(fFCCD-fDLT)
    else:
        return 1.+0.*x

def FCCDOuter(x,fDLT,fFCCD):
    if(x<=fDLT):
        return 0.*x
    elif(fDLT!=fFCCD and x>fDLT and x<fFCCD):
        return 1./(fFCCD-fDLT)*x-fDLT/(fFCCD-fDLT)
    else:
        return 1.+0.*x


#par_fit=[593929.434376, 1.168730, 2612.961984, 6765.938650, 3.980619, 0.447762] #change

'''
def f_random(sigma_r):
    f_smear_random_tot=list()
    for s in sigma_r:
        function_smear=TFormula("function_smear","([3]* [0]/(2. *  [4]) * exp( (x- [2])/ [4] +  [5]* [5]/(2. * [4]* [4]) ) * TMath::Erfc( (x- [2])/(sqrt(2)*  [5]) +  [5]/(sqrt(2) *  [4]))+  [0] / ( [1] * sqrt(2. * pi)) * exp( -1. * (x -  [2]) * (x -  [2]) / (2. *  [1] *  [1])) )/( [0]*(1.+ [3]))")
        f_smear_random=TF1("smear","function_smear",-100,+100,6)
        A_g, sigma_g, mu_g, B_tail, C, D = par_fit
        A=1
        mu=0
        R=B_tail/A_g
        f_smear_random.SetParameters(A,s,mu,R,C,D)
        smear_v=f_smear_random.GetRandom()
        f_smear_random_tot.append(smear_v)
    return f_smear_random_tot
'''
#f =lambda x: (p[3]* p[0]/(2. *  p[4]) * exp( (x- p[2])/ p[4] +  p[5]* p[5]/(2. * p[4]* p[4]) ) * TMath::Erfc( (x- p[2])/(sqrt(2)*  p[5]) +  p[5]/(sqrt(2) *  p[4]))+  p[0] / ( p[1] * sqrt(2. * pi)) * exp( -1. * (x -  p[2]) * (x -  p[2]) / (2. *  p[1] *  p[1])) )/( p[0]*(1.+ p[3])) 
#    A_g, sigma_g, mu_g, B_tail, C, D = par_fit
#    A=1
#    mu=0
#    R=B_tail/A_g
#    p=[A,sigma_r,mu,R,C,D]
    


def length_np(v:np.array):
    return sum(v*v);


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
   


def GetMinimumDistance(chain,point:np.array):
    if (len(chain)==0):
        return 0
    distance=chain[0][0].real_distance(point)
    for entry in chain:
        distance=min(distance,entry[0].real_distance(point))
    return distance
 

def GetDistanceToNPlus(fNPlus,r,z):
    return GetMinimumDistance(fNPlus,np.array([r,z]))


def GetDistanceToBore(fBore,r,z):
    return GetMinimumDistance(fBore,np.array([r,z]))


def GetChargeCollectionEfficiency(fNplus,fBore,r,z,fFCCD,fDLT):
    distanceToNPlus=GetDistanceToNPlus(fNplus,r,z)
    distanceToBore=GetDistanceToBore(fBore,r,z)
    minDist=min(distanceToBore,distanceToNPlus)
    if (minDist < 0):
        return 0
    return min(FCCDBore(distanceToBore,fDLT,fFCCD),FCCDOuter(distanceToNPlus,fDLT,fFCCD))
    #elif(minDist==distanceToBore):
    #    return FCCDBore(minDist,fDLT,fFCCD) 
    #else:
    #    return FCCDOuter(minDist,fDLT,fFCCD) 



##########################################################################

   

if __name__=="__main__":
    main()

