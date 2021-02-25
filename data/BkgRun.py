import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from scipy import optimize
from scipy import stats
import glob
import pygama
from pygama.analysis import histograms
from pygama.analysis import peak_fitting
import json
from datetime import datetime

#import fitting functions
import sys
sys.path.append('/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/data/')
from Ba133_dlt_analysis import * 

def main():

    #print date and time for log:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    detector = "I02160A"
    t2_folder = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/bkg/pygama/"
   
    keys, data = read_all_t2(t2_folder)
    print("Available keys: " ,keys)

    data_size = data.size #all events
    print("data_size: ", data_size)

    key = "e_ftp"
    key_data = obtain_key_data(data, keys, key, data_size)

    plt.figure()
    plt.hist(key_data, bins=10000)
    plt.ylabel("Counts")
    plt.xlabel("e_ftp")
    plt.yscale("log")
    

    #Linearly calibrated data:
    print("")
    print("Linearly calibrating energy...")

    with open('/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/data/calibration_coef.json') as json_file:
        calibration_coefs = json.load(json_file)
        m = calibration_coefs['m']
        m_err = calibration_coefs['m_err']
        c = calibration_coefs['c']
        c_err = calibration_coefs['c_err']
        a_quad = calibration_coefs['a_quad']
        a_quad_err = calibration_coefs['a_quad_err']
        b_quad = calibration_coefs['b_quad']
        b_quad_err = calibration_coefs['b_quad_err']
        c_quad = calibration_coefs['c_quad']
        c_quad_err = calibration_coefs['c_quad_err']

    print("m: ", m, " , c: ", c)

    calibrated_energy = (key_data-c)/m
    binwidth = 0.15 #0.1 #kev
    bins = np.arange(min(calibrated_energy), max(calibrated_energy) + binwidth, binwidth)

    plt.figure()
    counts, bins_cal, bars = plt.hist(calibrated_energy, bins=bins)
    plt.ylabel("Counts")
    plt.xlabel("Energy (keV)")
    plt.yscale("log")
    plt.xlim(0,450)
    plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/data/plots/bkgrun.png")
    plt.show()


if __name__ =="__main__":
    main()