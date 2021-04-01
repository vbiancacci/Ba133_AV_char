import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from scipy import optimize
from scipy import stats
import glob
import json
from datetime import datetime
from scipy.integrate import quad
import fnmatch
import argparse

import pygama.io.lh5 as lh5
import pygama
from pygama.analysis import histograms
from pygama.analysis import peak_fitting


#code to do gamma line peak counts on calibrated energy data
#can take 2 arguments as an input: the detector (just use I02160A for now) and whether to act on cuts or not
#to do:
# - decided whether to use trapE.lh5 files or eftp .h5 files, currently using eftp .h5
# - finalise choice of peak fitting functions
# - add in directory structure for detectors

def main():


    #print date and time for log:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("") 

    parser = argparse.ArgumentParser(description='Process calibrated Ba133 data for a particular detector, with cuts or not')
    parser.add_argument('--detector', action="store",type=str, default="I02160A")
    parser.add_argument('--cuts', action="store", type=bool, default = False)
    args = parser.parse_args()
    detector, cuts = args.detector, args.cuts
    print("detector: ", detector)
    print("applying cuts: ", str(cuts))
    print("")

    #initialise directories for detectors to save
    if not os.path.exists("detectors/"+detector+"/plots"):
        os.makedirs("detectors/"+detector+"/plots")

    #read tier 2 runs for Ba data
    t2_folder_h5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v00.00/"
    t2_folder_lh5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v01.00/"

    
    if cuts == False:
    
        #.h5 files, e_ftp - dont exist for V05266A
        # df_total_h5= read_all_dsp_h5(t2_folder_h5, cuts)
        # print("df_total_h5: ", df_total_h5)
        # e_ftp_data = df_total_h5['e_ftp']

        #.lh5 files, trapE
        df_total_lh5 = read_all_dsp_lh5(t2_folder_lh5,cuts)
        print("df_total_lh5: ", df_total_lh5)
        trapE_data = df_total_lh5['trapE']

    else:

        #.h5 files, e_ftp - dont exist for V05266A
        passed_cuts = json.load(open('/lfs/l1/legend/users/aalexander/large_files/cuts/'+detector+'_ba_top_passed_cuts_data.json','r')) #passed cuts
        df_total_cuts_h5 = read_all_dsp_h5(t2_folder_h5,cuts, passed_cuts = passed_cuts)
        print("df_total_cuts_h5: ", df_total_cuts_h5)
        e_ftp_data_cuts = df_total_cuts_h5['e_ftp']

        #.lh5 files, trapE
        df_total_cuts_lh5 = read_all_dsp_lh5(t2_folder_lh5, cuts, passed_cuts=passed_cuts)
        print("df_total_cuts_lh5: ", df_total_cuts_lh5)
        trapE_data_cuts = df_total_cuts_lh5['trapE']


    #plt.figure()
    #bins = 10000
    #counts, bins, bars = plt.hist(e_ftp_data, bins=bins, label = "e_ftp no cuts")
    # counts, bins, bars = plt.hist(e_ftp_data_cuts, bins=bins, label = "e_ftp with cuts")
    #counts, bins, bars = plt.hist(trapE_data, bins=bins, label = "trapE no cuts")
    # counts, bins, bars = plt.hist(trapE_data_cuts, bins=bins, label = "trapE with cuts")
    #plt.legend()
    #plt.yscale("log")
    #plt.xlim(0,20000)
    #plt.show()

    # plt.figure()
    # bins = 10000
    # counts, bins, bars = plt.hist(e_ftp_data, bins=bins, label = "e_ftp no cuts")
    # counts, bins, bars = plt.hist(e_ftp_data_cuts, bins=bins, label = "e_ftp with cuts")
    # plt.legend()
    # plt.yscale("log")
    # plt.xlim(0,20000)


    #Linearly calibrated data:
    print("")
    print("Linearly calibrating energy...")

    #with open('detectors/'+detector+'/calibration_coef.json') as json_file:
    with open('detectors/'+detector+'/calibration_coef_trapE.json') as json_file:
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

    binwidth = 0.15 #0.1 #kev

    
    if cuts == False:
        energy_data = (trapE_data-c)/m #change to e_ftp_data if needed
        print("energy data: ", energy_data)
        #energy_data = energy_data[:,1]
        print("energy data type: ", type(energy_data))
        bins = np.arange(min(energy_data), max(energy_data) + binwidth, binwidth)
        counts_energy_data, bins, bars = plt.hist(energy_data, bins=bins, label = "no cuts")
    else:
        energy_data_cuts= (trapE_data_cuts-c)/m #change to e_ftp_data if needed
        print("energy data cuts: ", energy_data_cuts)
        bins = np.arange(min(energy_data_cuts), max(energy_data_cuts) + binwidth, binwidth)
        counts_energy_data_cuts, bins_cuts, bars_cuts = plt.hist(energy_data_cuts, bins=bins, label = "pile up cuts")
    

    #code for plotting both cuts and no cuts together
    # plt.figure()
    # counts_energy_data, bins, bars = plt.hist(energy_data, bins=bins, label = "no cuts")
    # counts_energy_data_cuts, bins_cuts, bars_cuts = plt.hist(energy_data_cuts, bins=bins, label = "pile up cuts")
    # plt.legend()
    # plt.yscale("log")
    # plt.xlim(0,450)
    # plt.xlabel("Energy (keV)")
    # plt.ylabel("Frequency")

    #plt.close("all")  

    #_________Construct dlt observable________
    print("")
    print("Constructing Ba133 dead layer observable...")

    #_______________356keV peak___________
    print("")
    print("356 peak...")


    #fit peak with gaussian and unconstrained cdf
    xmin_356, xmax_356 = 352, 360 #360 #kev 
    if detector == V05266A:
        xmin_356, xmax_356 = 354, 360 #360 #kev
    if cuts == False:
        plt.figure()
        popt, pcov, xfit = fit_peak_356("Energy (keV)", bins, counts_energy_data, xmin_356, xmax_356)
        a,b,c,d,e,f,g = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6] 
        counts, bins, bars = plt.hist(energy_data, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin_356, xmax_356) 
        plt.ylim(0.5*100, 5*10**5) #bin size 0.15
        #plt.ylim(10, 10**5) #0.05
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/356keV_dlt.png")
    else:
        plt.figure()
        popt, pcov, xfit = fit_peak_356("Energy (keV)", bins, counts_energy_data_cuts, xmin_356, xmax_356)
        a,b,c,d,e,f,g = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6] 
        counts, bins, bars = plt.hist(energy_data_cuts, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin_356, xmax_356) 
        plt.ylim(0.5*100, 5*10**5) #bin size 0.15
        #plt.ylim(10, 10**5) #0.05
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/356keV_dlt_cuts.png")

    C_356, C_356_err = gauss_count(a, b, c, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
    print("gauss count: ", C_356, " +/- ", C_356_err )

    # #check with manual integration
    # bina, binb = int(xmin_356/binwidth), int(xmax_356/binwidth)
    # bkg_estimate = g*(xmax_356-xmin_356)
    # integral_356 = binwidth*sum(counts_cuts[bina:binb]) - bkg_estimate
    # print("manual integral check: ", integral_356)
    # print("manual integral check divided by binwidth: ", integral_356/binwidth)

    # #check with 3 sigma integration
    # integral_356_3sigma = quad(gaussian,b-3*c, b+3*c, args=(a,b,c))
    # print("3 sigma integral check: ", integral_356_3sigma)
    # print("3 sigma integral check divided by binwidth: ", integral_356_3sigma[0]/binwidth)


    #try other fits - constrained cdf
    if cuts == False:
        plt.figure()
        popt, pcov, xfit = fit_peak_356_2("Energy (keV)", bins, counts_energy_data, xmin_356, xmax_356)
        a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]
        counts, bins, bars = plt.hist(energy_data, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin_356, xmax_356) 
        plt.ylim(0.5*100, 5*10**5)
        #plt.ylim(10, 10**5) #0.05
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/356keV_dlt_2.png")
    else:
        plt.figure()
        popt, pcov, xfit = fit_peak_356_2("Energy (keV)", bins, counts_energy_data_cuts, xmin_356, xmax_356)
        a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]
        counts, bins, bars = plt.hist(energy_data_cuts, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin_356, xmax_356) 
        plt.ylim(0.5*100, 5*10**5)
        #plt.ylim(10, 10**5) #0.05
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/356keV_dlt_2_cuts.png")

    C_356_2, C_356_2_err = gauss_count(a,b, c, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
    print("gauss count: ", C_356_2, " +/- ", C_356_2_err )

    #propagate error of 2 fits:
    C_356_average = (C_356 + C_356_2)/2
    C_356_average_err = 0.5*np.sqrt(C_356_err**2 + C_356_2_err**2)
    print("gauss count averaged: ", C_356_average, " +/- ", C_356_average_err )

    #extra gaussian fit
    # plt.figure()
    # popt, pcov, xfit = fit_peak_356_4("Energy (keV)", bins, counts_cuts, xmin_356, xmax_356)
    # a,b,c,d,e,f,g,h = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6], popt[7]
    # counts, bins, bars = plt.hist(energy_data_cuts, bins=bins, histtype='step', color='grey')
    # plt.xlim(xmin_356, xmax_356) 
    # plt.ylim(50, 5*10**5)
    # #plt.ylim(10, 10**5) #0.05
    # plt.yscale("log")
    # plt.savefig("plots/356keV_dlt_4_cuts.png")


    #__________79.6/81keV double peak____________
    print("")
    print("79.6/81 keV double peak...")


    #fit peak with double gaussian and double cdf
    xmin_81, xmax_81 = 76.5, 84 #78, 83 #77, 84 #83.5 #kev
    if cuts == False:
        plt.figure()
        popt, pcov, xfit = fit_double_peak_81("Energy (keV)", bins, counts_energy_data, xmin_81, xmax_81)
        a,b,c,d,e,f,g,h = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7] 
        counts, bins, bars = plt.hist(energy_data, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin_81, xmax_81) 
        plt.ylim(100, 5*10**5)
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/81keV_dlt.png")
    else:
        plt.figure()
        popt, pcov, xfit = fit_double_peak_81("Energy (keV)", bins, counts_energy_data_cuts, xmin_81, xmax_81)
        a,b,c,d,e,f,g,h = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7] 
        counts, bins, bars = plt.hist(energy_data_cuts, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin_81, xmax_81) 
        plt.ylim(100, 5*10**5)
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/81keV_dlt_cuts.png")

    R = 2.65/32.9
    C_81, C_81_err = gauss_count(a, b, c, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
    C_79, C_79_err = gauss_count(R*a, d, e, R*np.sqrt(pcov[0][0]), np.sqrt(pcov[3][3]), np.sqrt(pcov[4][4]), binwidth)
    print("gauss count 81: ", C_81, " +/- ", C_81_err )
    print("gauss count 79.6: ", C_79, " +/- ", C_79_err )

    print("")
    print("Using just C_356_2, the constrained cdf")
    O_Ba133 = (C_79 + C_81)/C_356
    O_Ba133_err = O_Ba133*np.sqrt((C_79_err**2 + C_81_err**2)/(C_79+C_81)**2 + (C_356_err/C_356)**2)
    print("O_BA133 = " , O_Ba133, " +/- ", O_Ba133_err)

    print("")
    print("Using the average of C_356_1 and 2")
    O_Ba133_av = (C_79 + C_81)/C_356_average
    O_Ba133_av_err = O_Ba133*np.sqrt((C_79_err**2 + C_81_err**2)/(C_79+C_81)**2 + (C_356_average_err/C_356_average)**2)
    print("O_BA133 = " , O_Ba133_av, " +/- ", O_Ba133_av_err)


    #fit other gaussian gamma peaks
    peak_ranges = [[159,162],[221.5,225],[274,279],[300,306],[381,386]] #Rough by eye
    peaks = [161, 223, 276, 303, 383]
    other_peak_counts = []
    other_peak_counts_err = []
    for index, i in enumerate(peak_ranges):
        if cuts == False:
            plt.figure()
            xmin, xmax = i[0], i[1]
            popt, pcov, xfit = fit_peak_356_2("Energy (keV)", bins, counts_energy_data, xmin, xmax)
            a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]
            counts, bins, bars = plt.hist(energy_data, bins=bins, histtype='step', color='grey')
            plt.xlim(xmin, xmax) 
            plt.yscale("log")
            plt.savefig("detectors/"+detector+"/plots/"+str(peaks[index])+"keV_dlt.png")
        else:
            plt.figure()
            xmin, xmax = i[0], i[1]
            popt, pcov, xfit = fit_peak_356_2("Energy (keV)", bins, counts_energy_data_cuts, xmin, xmax)
            a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]
            counts, bins, bars = plt.hist(energy_data_cuts, bins=bins, histtype='step', color='grey')
            plt.xlim(xmin, xmax) 
            plt.yscale("log")
            plt.savefig("detectors/"+detector+"/plots/"+str(peaks[index])+"keV_dlt_cuts.png")

        C, C_err = gauss_count(a,b, c, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
        print(str(peaks[index]), " keV")
        print("gauss count: ", C, " +/- ", C_err )
        other_peak_counts.append(C)
        other_peak_counts_err.append(C_err)


    #Save count values to json file
    dlt_observables = {
        "C_356": C_356,
        "C_356_err" : C_356_err,
        "C_356_2": C_356_2,
        "C_356_2_err" : C_356_2_err,
        "C_356_average": C_356_average,
        "C_356_average_err" : C_356_average_err,
        "C_81" : C_81,
        "C_81_err" : C_81_err,
        "C_79" : C_79,
        "C_79_err" : C_79_err,
        "O_Ba133" : O_Ba133,
        "O_Ba133_err" : O_Ba133_err,
        "O_Ba133_av" : O_Ba133_av,
        "O_Ba133_av_err" : O_Ba133_av_err,
        "C_161" : other_peak_counts[0],
        "C_161_err" : other_peak_counts_err[0],
        "C_223" : other_peak_counts[1],
        "C_223_err" : other_peak_counts_err[1],
        "C_276" : other_peak_counts[2],
        "C_276_err" : other_peak_counts_err[2],
        "C_303" : other_peak_counts[3],
        "C_303_err" : other_peak_counts_err[3],
        "C_383" : other_peak_counts[4],
        "C_383_err" : other_peak_counts_err[4]
    }

    if cuts == False:
        with open("detectors/"+detector+"/dlt_observables.json", "w+") as outfile: 
            json.dump(dlt_observables, outfile)
    else:
        with open("detectors/"+detector+"/dlt_observables_cuts.json", "w+") as outfile: 
            json.dump(dlt_observables, outfile)

def read_all_dsp_h5(t2_folder, cuts, passed_cuts = None):
    "get data from all tier2/dsp files from same run within a directory. Apply cuts"

    files = os.listdir(t2_folder)
    files = fnmatch.filter(files, "*.h5")

    df_list = []
    df_cuts_list = []
    for file in files:

        #get data, no cuts
        df = pd.read_hdf(t2_folder+file, "data")
        df_list.append(df)

        #apply cuts
        if cuts == True:
            file_mod = file.replace(".h5", "_tier1.lh5")
            file_mod = file_mod.replace("t2_char_data-I02160A","char_data-I02160A")
            idx = passed_cuts[file_mod]
            df_cuts = df.iloc[idx, :]
            #df_cuts = df[df['ievt'].isin(idx)]
            df_cuts_list.append(df_cuts)

    if cuts == False:
        df_total = pd.concat(df_list, axis=0, ignore_index=True)
        return df_total
    else:
        df_total_cuts = pd.concat(df_cuts_list, axis=0, ignore_index=True)
        return df_total_cuts
    
def read_all_dsp_lh5(t2_folder, cuts, passed_cuts = None):

    sto = lh5.Store()
    files = os.listdir(t2_folder)
    files = fnmatch.filter(files, "*lh5")

    df_list = []
    df_cuts_list = []

    for file in files:
    
        #get data, no cuts
        tb = sto.read_object("raw",t2_folder+file)[0]
        df = lh5.Table.get_dataframe(tb)
        df_list.append(df)

        #apply cuts
        if cuts == True:
            file_mod = file.replace("tier2", "tier1")
            idx = passed_cuts[file_mod]
            # tb_cuts = sto.read_object("raw",t2_folder+file, idx=idx)[0] #needs new version of pygama
            # df_cuts = lh5.Table.get_dataframe(tb_cuts)
            df_cuts = df.iloc[idx, :]
            df_cuts_list.append(df_cuts)

    if cuts == False:
        df_total = pd.concat(df_list, axis=0, ignore_index=True)
        return df_total
    else:
        df_total_cuts = pd.concat(df_cuts_list, axis=0, ignore_index=True)
        return df_total_cuts


def linear_fit(x, m, c):
    "linear function"
    f = m*x + c
    return f

def quadratic_fit(x,a,b,c):
    "quadratic function"
    f = a*x**2 + b*x + c
    return f

def sqrt_curve(x,a,c):
    "square root function with offset"
    f = a*np.sqrt(x+c)
    return f

def gaussian(x,a,b,c):
    "gaussian function without offset"
    f = a*np.exp(-((x-b)**2.0/(2.0*c**2.0)))
    #f = a*np.exp(-(pow((x-b),2)/(2.0*pow(c,2))))
    return f

def gaussian_cdf(x,a,b):
    "gaussian cdf function"
    f = stats.norm.cdf(x, a, b) #default e=0=mean/loc, f=1=sigma/scale
    return f

def gaussian_and_bkg(x, a, b, c, d, e, f, g):
    "fit function for 356kev peak"
    f = gaussian(x, a, b, c) - d*gaussian_cdf(x, e, f) + g
    return f

def gaussian_and_bkg_2(x, a, b, c, d, e):
    "fit function for 356kev peak - cdf fixed to same params as gaussian"
    f = gaussian(x, a, b, c) - d*gaussian_cdf(x, b, c) + e
    return f

def gaussian_and_bkg_3(x, a, b, c, d, e, f):
    "fit function for 356kev peak - cdf fixed to the just same mean as gaussian, different sigma"
    f = gaussian(x, a, b, c) - d*gaussian_cdf(x, b, e) + f
    return f


def gaussian_and_bkg_4(x, a, b, c, d, e, f,g,h):
    "fit function for 356kev peak - cdf fixed to the just same mean as gaussian, different sigma, plus extra gaussian"
    f = gaussian(x, a, b, c) - d*gaussian_cdf(x, b, c) + e + gaussian(x,f,g,h)
    return f



def low_energy_tail(x, a ,b):
    """ The "low energy tail" is a heuristic function used to account for any mechanism leading to a systematic underestimation of the energy for a certain class of events. 
    Often the tail is due to:

    1. trapping/recombination of the charged carriers during their drift to the electrodes
    2. events close to the n+ electrode in which part of the charges is created in a weak e-field region (slow pulses)
    3. ballistic deficit -> the integration time of the energy filter is not enough
    4. high rate of the source -> pulses sit on the tail of a previous pulse (pile-ups) and the energy filter does not fully correct for it

    1 and 2 are related to the detector itself and usually cannot be fixed. 
    3 can be fixed by increasing the integration time of the filter but typically this worsen the energy resolution. 
    4 can be reduced by removing from the analysis the events with multiple pulses in the same waveform or with non-flat baselines.
    Slow pulses are probably the main issue for low energy gamma-lines....
    """

    f = a +b 
    return f

def gaussian_and_bkg_5(x, a, b, c, d, e):
    """fit function for 356kev peak consisting of:
    - Gaussian [signal]
    - Gaussian CDF, constained to gaussian centroid [bkg - energy loss of centroid gamma]
    - linear [bkg - ]
    - Low energy tail [bkg - heuristic function used to account for any mechanism leading to a systematic underestimation of the energy for a certain class of events.]
    """
    f = gaussian(x, a, b, c) - d*gaussian_cdf(x, b, c) + linear_fit(x, e, f) + low_energy_tail(x, g, h)
    return f

def double_gaussian_and_bkg(x,a,b,c,d,e,f,g,h):
    "fit function for Ba-133 79/81keV double peak - double gaussian and double cdf step, constrained bkg to mean and sigma of gauss"

    R = 2.65/32.9 #intensity ratio for Ba-133 double peak
    #f = gaussian(x, a, b, c) + R*gaussian(x, a, d, e) - f*gaussian_cdf(x,b,c) - g*gaussian_cdf(x, d, e) + h 
    f = gaussian(x, a, b, c) + R*gaussian(x, a, d, e) + f*gaussian_cdf(x,b,c) + g*gaussian_cdf(x, d, e) + h 
    
    return f

def chi_sq_calc(xdata, ydata, yerr, fit_func, popt):
    "calculate chi sq and p-val of a fit given the data points and fit parameters, e.g. fittype ='linear'"
   
    y_obs = ydata
    y_exp = []

    y_exp = []
    for index, y_i in enumerate(y_obs):
        x_obs = xdata[index]
        y_exp_i = fit_func(x_obs, *popt)
        y_exp.append(y_exp_i)

    #chi_sq, p_value = stats.chisquare(y_obs, y_exp)#this is without errors
    chi_sq = 0.
    residuals = []
    for i in range(len(y_obs)):
        if yerr[i] != 0:
            residual = (y_obs[i]-y_exp[i])/(yerr[i])
        else:
            residual = 0.
        chi_sq += (residual)**2
        residuals.append(residual)

    N = len(y_exp) #number of data points
    dof = N-1
    chi_sq_red = chi_sq/dof

    p_value = 1-stats.chi2.cdf(chi_sq, dof)

    return chi_sq, p_value, residuals, dof

def gauss_count(a,b,c, a_err, b_err,c_err, bin_width):
    "count/integrate gaussian from -inf to plus inf"

    #integral = a*c*np.sqrt(2*np.pi) #old - not normalised to bin width
    integral = a*c*np.sqrt(2*np.pi)/bin_width
    integral_err = integral*np.sqrt((a_err/a)**2 + (c_err/c)**2)

    #3sigma
    integral_356_3sigma_list = quad(gaussian,b-3*c, b+3*c, args=(a,b,c))
    integral = integral_356_3sigma_list[0]/bin_width
    intergral_err = integral_356_3sigma_list[1]/bin_width

    return integral, integral_err

def fit_peak_356(key, bins, counts, xmin, xmax):
    "fit the 356 keV peak with gaussian +cdf bkg"

    no_bins = bins.size 
    binwidth = bins[1]-bins[0]

    xdata = []
    ydata = []
    for bin in bins:
        bin_centre = bin + 0.5*(max(bins)-(min(bins)))/no_bins 
        if bin_centre < xmax and bin_centre > xmin:
            xdata.append(bin_centre)
            bin_index = np.where(bins == bin)[0][0]
            ydata.append(counts[bin_index])

    xdata = np.array(xdata)
    ydata = np.array(ydata)     

    yerr = np.sqrt(ydata) #counting error

    #initial rough guess of params
    aguess = max(ydata) - min(ydata) #gauss amplitude
    aguess_index = np.where(ydata == max(ydata))[0][0]
    bguess = xdata[aguess_index] #gauss mean
    cguess =  1 #gauss sigma
    dguess =  100 #0 #cdf amp
    eguess =  bguess #cdf mean
    fguess = cguess #cdf sigma
    gguess = min(ydata) #offset
    p_guess = [aguess, bguess, cguess, dguess, eguess, fguess, gguess]
    print(p_guess)
    bounds=([0, 0, 0, 0, xmin, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, xmax, np.inf, np.inf])

    popt, pcov = optimize.curve_fit(gaussian_and_bkg, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf", bounds = bounds) #nb, if there is a yerr=0, sigma needs an if statement
    print(popt)
    a,b,c,d,e,f,g = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]

    fig, ax = plt.subplots()
    #ax.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    plt.plot(xfit, gaussian_and_bkg(xfit,*popt), "g", label = "gauss(x,a,b,c) - d*gauss_cdf(x,e,f) + g") 
    plt.plot(xfit, -1*d*gaussian_cdf(xfit,e,f) + g, "r--", label ="-d*gauss_cdf(x,e,f) + g")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.legend(loc="upper right", fontsize=8)

    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, gaussian_and_bkg, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3g \pm %.3g$' % (b, np.sqrt(pcov[1][1])), r'$c=%.3g \pm %.3g$' % (c, np.sqrt(pcov[2][2])), r'$d=%.3g \pm %.3g$' % (d, np.sqrt(pcov[3][3])), r'$e=%.3g \pm %.3g$' % (e, np.sqrt(pcov[4][4])), r'$f=%.3g \pm %.3g$' % (f, np.sqrt(pcov[5][5])),r'$g=%.3g \pm %.3g$' % (g, np.sqrt(pcov[6][6])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'binwidth = $%.2g$ keV'%binwidth))
    plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) #ax.text..ax.tra

    return popt, pcov, xfit


def fit_peak_356_2(key, bins, counts, xmin, xmax):
    "fit the 356 keV peak with gaussian +cdf bkg - cdf mean and sigma constrained to that of gaussian"

    no_bins = bins.size 
    binwidth = bins[1]-bins[0]

    xdata = []
    ydata = []
    for bin in bins:
        bin_centre = bin + 0.5*(max(bins)-(min(bins)))/no_bins 
        if bin_centre < xmax and bin_centre > xmin:
            xdata.append(bin_centre)
            bin_index = np.where(bins == bin)[0][0]
            ydata.append(counts[bin_index])

    xdata = np.array(xdata)
    ydata = np.array(ydata)     

    yerr = np.sqrt(ydata) #counting error

    #initial rough guess of params
    aguess = max(ydata) - min(ydata) #gauss amplitude
    aguess_index = np.where(ydata == max(ydata))[0][0]
    bguess = xdata[aguess_index] #gauss mean
    cguess =  1 #gauss sigma
    dguess =  100 #0 #cdf amp
    eguess = min(ydata) #offset
    p_guess = [aguess,bguess,cguess,dguess,eguess]
    print(p_guess)
    bounds=([0, 0, 0, -np.inf, -np.inf], [np.inf]*5)

    popt, pcov = optimize.curve_fit(gaussian_and_bkg_2, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**8, method ="trf", bounds = bounds)
    print(popt)
    a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]

    fig, ax = plt.subplots()
    #ax.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    plt.plot(xfit, gaussian_and_bkg_2(xfit,*popt), "g", label = "gauss(x,a,b,c) - d*gauss_cdf(x,b,c) + e") 
    plt.plot(xfit, -1*d*gaussian_cdf(xfit,b,c) + e, "r--", label ="-d*gauss_cdf(x,b,c) + e")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.legend(loc="upper right", fontsize=8)

    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, gaussian_and_bkg_2, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3g \pm %.3g$' % (b, np.sqrt(pcov[1][1])), r'$c=%.3g \pm %.3g$' % (c, np.sqrt(pcov[2][2])), r'$d=%.3g \pm %.3g$' % (d, np.sqrt(pcov[3][3])), r'$e=%.3g \pm %.3g$' % (e, np.sqrt(pcov[4][4])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'binwidth = $%.2g$ keV'%binwidth))
    plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) #ax.text..ax.tra

    return popt, pcov, xfit

def fit_peak_356_3(key, bins, counts, xmin, xmax):
    "fit the 356 keV peak with gaussian +cdf bkg - cdf mean constained to that of gaussian"

    no_bins = bins.size 
    binwidth = bins[1]-bins[0]

    xdata = []
    ydata = []
    for bin in bins:
        bin_centre = bin + 0.5*(max(bins)-(min(bins)))/no_bins 
        if bin_centre < xmax and bin_centre > xmin:
            xdata.append(bin_centre)
            bin_index = np.where(bins == bin)[0][0]
            ydata.append(counts[bin_index])

    xdata = np.array(xdata)
    ydata = np.array(ydata)     

    yerr = np.sqrt(ydata) #counting error

    #initial rough guess of params
    aguess = max(ydata) - min(ydata) #gauss amplitude
    aguess_index = np.where(ydata == max(ydata))[0][0]
    bguess = xdata[aguess_index] #gauss mean
    cguess =  1 #gauss sigma
    dguess =  100 #0 #cdf amp
    eguess = 1 #cdf sigma
    fguess = min(ydata) #offset
    p_guess = [aguess,bguess,cguess,dguess,eguess,fguess]
    print(p_guess)
    bounds=([0, 0, 0, 0, 0.1, -1000], [np.inf, np.inf, np.inf, 500, 10, 1000])

    popt, pcov = optimize.curve_fit(gaussian_and_bkg_3, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf", bounds = bounds)
    print(popt)
    a,b,c,d,e,f = popt[0],popt[1],popt[2],popt[3],popt[4], popt[5]

    fig, ax = plt.subplots()
    #ax.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    plt.plot(xfit, gaussian_and_bkg_3(xfit,*popt), "g", label = "gauss(x,a,b,c) - d*gauss_cdf(x,b,e) + f") 
    plt.plot(xfit, -1*d*gaussian_cdf(xfit,b,e) + f, "r--", label ="-d*gauss_cdf(x,b,e) + f")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.legend(loc="upper right", fontsize=8)

    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, gaussian_and_bkg_3, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3g \pm %.3g$' % (b, np.sqrt(pcov[1][1])), r'$c=%.3g \pm %.3g$' % (c, np.sqrt(pcov[2][2])), r'$d=%.3g \pm %.3g$' % (d, np.sqrt(pcov[3][3])), r'$e=%.3g \pm %.3g$' % (e, np.sqrt(pcov[4][4])), r'$f=%.3g \pm %.3g$' % (f, np.sqrt(pcov[5][5])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'binwidth = $%.2g$ keV'%binwidth))
    plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) #ax.text..ax.tra

    return popt, pcov, xfit

def fit_peak_356_4(key, bins, counts, xmin, xmax):
    "fit the 356 keV peak with gaussian +cdf bkg - cdf mean and sigma constrained to that of gaussian, +extra gaussian"

    no_bins = bins.size 
    binwidth = bins[1]-bins[0]

    xdata = []
    ydata = []
    for bin in bins:
        bin_centre = bin + 0.5*(max(bins)-(min(bins)))/no_bins 
        if bin_centre < xmax and bin_centre > xmin:
            xdata.append(bin_centre)
            bin_index = np.where(bins == bin)[0][0]
            ydata.append(counts[bin_index])

    xdata = np.array(xdata)
    ydata = np.array(ydata)     

    yerr = np.sqrt(ydata) #counting error

    #initial rough guess of params
    aguess = max(ydata) - min(ydata) #gauss amplitude
    aguess_index = np.where(ydata == max(ydata))[0][0]
    bguess = xdata[aguess_index] #gauss mean
    cguess =  1 #gauss sigma
    dguess =  100 #0 #cdf amp
    eguess = min(ydata) #offset
    fguess = 100 #coincidence gaussian
    gguess = 357.38
    hguess =  1
    p_guess = [aguess,bguess,cguess,dguess,eguess, fguess,gguess,hguess]
    print(p_guess)
    bounds=([0, 0, 0, 0, -np.inf,0,0,0], [np.inf]*8)

    popt, pcov = optimize.curve_fit(gaussian_and_bkg_4, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf", bounds = bounds)
    print(popt)
    a,b,c,d,e,f,g,h = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]

    fig, ax = plt.subplots()
    #ax.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    plt.plot(xfit, gaussian_and_bkg_4(xfit,*popt), "g", label = "gauss(x,a,b,c) - d*gauss_cdf(x,b,c) + e + gauss(x,f,g,h)") 
    plt.plot(xfit, -1*d*gaussian_cdf(xfit,b,c) + e, "r--", label ="-d*gauss_cdf(x,b,c) + e")
    plt.plot(xfit, gaussian(xfit,f,g,h), "b", label ="gauss(x,f,g,h) - d*gauss_cdf(x,b,c) + e")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.legend(loc="upper right", fontsize=8)

    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, gaussian_and_bkg_4, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3g \pm %.3g$' % (b, np.sqrt(pcov[1][1])), r'$c=%.3g \pm %.3g$' % (c, np.sqrt(pcov[2][2])), r'$d=%.3g \pm %.3g$' % (d, np.sqrt(pcov[3][3])), r'$e=%.3g \pm %.3g$' % (e, np.sqrt(pcov[4][4])), r'$f=%.3g \pm %.3g$' % (f, np.sqrt(pcov[5][5])), r'$g=%.3g \pm %.3g$' % (g, np.sqrt(pcov[6][6])), r'$h=%.3g \pm %.3g$' % (h, np.sqrt(pcov[7][7])),r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'binwidth = $%.2g$ keV'%binwidth))
    plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) #ax.text..ax.tra

    return popt, pcov, xfit

def fit_peak_356_5(key, bins, counts, xmin, xmax):
    "IN DEVELOPMENT fit the 356 keV peak with gaussian +cdf bkg - cdf mean constained to that of gaussian"

    no_bins = bins.size 
    binwidth = bins[1]-bins[0]

    xdata = []
    ydata = []
    for bin in bins:
        bin_centre = bin + 0.5*(max(bins)-(min(bins)))/no_bins 
        if bin_centre < xmax and bin_centre > xmin:
            xdata.append(bin_centre)
            bin_index = np.where(bins == bin)[0][0]
            ydata.append(counts[bin_index])

    xdata = np.array(xdata)
    ydata = np.array(ydata)     

    yerr = np.sqrt(ydata) #counting error

    #initial rough guess of params
    aguess = max(ydata) - min(ydata) #gauss amplitude
    aguess_index = np.where(ydata == max(ydata))[0][0]
    bguess = xdata[aguess_index] #gauss mean
    cguess =  1 #gauss sigma
    dguess =  100 #0 #cdf amp
    eguess = 1 #cdf sigma
    fguess = min(ydata) #offset
    p_guess = [aguess,bguess,cguess,dguess,eguess,fguess]
    print(p_guess)
    bounds=([0, 0, 0, 0, 0.1, -1000], [np.inf, np.inf, np.inf, 500, 10, 1000])

    popt, pcov = optimize.curve_fit(gaussian_and_bkg_3, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf", bounds = bounds)
    print(popt)
    a,b,c,d,e,f = popt[0],popt[1],popt[2],popt[3],popt[4], popt[5]

    fig, ax = plt.subplots()
    #ax.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    plt.plot(xfit, gaussian_and_bkg_3(xfit,*popt), "g", label = "gauss(x,a,b,c) - d*gauss_cdf(x,b,e) + f") 
    plt.plot(xfit, -1*d*gaussian_cdf(xfit,b,e) + f, "r--", label ="-d*gauss_cdf(x,b,e) + f")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.legend(loc="upper right", fontsize=8)

    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, gaussian_and_bkg_3, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3g \pm %.3g$' % (b, np.sqrt(pcov[1][1])), r'$c=%.3g \pm %.3g$' % (c, np.sqrt(pcov[2][2])), r'$d=%.3g \pm %.3g$' % (d, np.sqrt(pcov[3][3])), r'$e=%.3g \pm %.3g$' % (e, np.sqrt(pcov[4][4])), r'$f=%.3g \pm %.3g$' % (f, np.sqrt(pcov[5][5])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'binwidth = $%.2g$ keV'%binwidth))
    plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) #ax.text..ax.tra

    return popt, pcov, xfit


def fit_double_peak_81(key, bins, counts, xmin, xmax):
    "fit the double 79.6 /81 keV peak with gaussian +cdf bkg - cdf mean and sigma constrained to that of gaussian"

    no_bins = bins.size 
    binwidth = bins[1]-bins[0]

    xdata = []
    ydata = []
    for bin in bins:
        bin_centre = bin + 0.5*(max(bins)-(min(bins)))/no_bins 
        if bin_centre < xmax and bin_centre > xmin:
            xdata.append(bin_centre)
            bin_index = np.where(bins == bin)[0][0]
            ydata.append(counts[bin_index])

    xdata = np.array(xdata)
    ydata = np.array(ydata)     

    yerr = np.sqrt(ydata) #counting error

    #initial rough guess of params
    aguess = max(ydata) - min(ydata) #gauss amplitude
    bguess = 80.9979 #80.9979 for sims # 81 for data
    cguess =  0.5 #gauss 81 sigma
    dguess =  79.6142 #79.6142 for sims #79 for data
    eguess = 0.5 #gauss 79.6 sigma 
    #fguess = 100 #cdf 81 amp
    fguess = -100 #cdf 81 amp
    #gguess =  100 #100 #cdf 79.6 amp
    gguess =  -100 #100 #cdf 79.6 amp
    hguess = min(ydata) #offset
    p_guess = [aguess,bguess,cguess,dguess,eguess, fguess, gguess, hguess]
    print(p_guess)
    bounds=([0, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf], [np.inf]*8) #this one for data
    #bounds=([0, 0, 0, 0, 0, 0, 0, -np.inf], [np.inf]*8) #this one for simulations

    popt, pcov = optimize.curve_fit(double_gaussian_and_bkg, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**9, method ="trf") #, bounds = bounds)
    print(popt)
    a,b,c,d,e,f,g,h = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6], popt[7]

    fig, ax = plt.subplots()
    #ax.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='x', ms = 0.75, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    #plt.plot(xfit, double_gaussian_and_bkg(xfit,*popt), "g", label = "gauss(x,a,b,c) + R*gauss(x,a,d,e) - f*gauss_cdf(x,b,c) -g*gauss_cdf(x,d,e) + h") 
    plt.plot(xfit, double_gaussian_and_bkg(xfit,*popt), "g", label = "gauss(x,a,b,c) + R*gauss(x,a,d,e) + f*gauss_cdf(x,b,c) + g*gauss_cdf(x,d,e) + h") 
    #plt.plot(xfit, -1*f*gaussian_cdf(xfit,b,c) -g*gaussian_cdf(xfit,d,e) + h, "r--", label ="-f*gauss_cdf(x,b,c) -g*gauss_cdf(x,d,e) + h")
    #plt.plot(xfit, f*gaussian_cdf(xfit,b,c) +g*gaussian_cdf(xfit,d,e) + h, "r--", label ="-f*gauss_cdf(x,b,c) -g*gauss_cdf(x,d,e) + h")
    plt.plot(xfit, f*gaussian_cdf(xfit,b,c) +g*gaussian_cdf(xfit,d,e) + h, "r--", label ="f*gauss_cdf(x,b,c) +g*gauss_cdf(x,d,e) + h")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.legend(loc="upper right", fontsize=8)

    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, double_gaussian_and_bkg, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3g \pm %.3g$' % (b, np.sqrt(pcov[1][1])), r'$c=%.3g \pm %.3g$' % (c, np.sqrt(pcov[2][2])), r'$d=%.3g \pm %.3g$' % (d, np.sqrt(pcov[3][3])), r'$e=%.3g \pm %.3g$' % (e, np.sqrt(pcov[4][4])), r'$f=%.3g \pm %.3g$' % (f, np.sqrt(pcov[5][5])), r'$g=%.3g \pm %.3g$' % (g, np.sqrt(pcov[6][6])), r'$h=%.3g \pm %.3g$' % (h, np.sqrt(pcov[7][7])), r'$R=2.65/32.9$',r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'binwidth = $%.2g$ keV'%binwidth))
    plt.text(0.02, 0.82, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) #ax.text..ax.tra

    return popt, pcov, xfit


if __name__ =="__main__":
    main()