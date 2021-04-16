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
import sys
import fnmatch

import pygama.io.lh5 as lh5
import pygama
from pygama.analysis import histograms
from pygama.analysis import peak_fitting

# import sys
# sys.path.append('/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/')
# from Ba133_data_AV_analysis import * 

"Script to calibrate Ba spectrum from known peaks and output coefficients into json file -- needs to be replaced by clints pygama energy_cal.py"

def main():

    #print date and time for log:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    if(len(sys.argv) != 2):
        print('Example usage: python Ba133_Calibration.py V05266A')
        sys.exit()

    print("start...")

    detector = sys.argv[1]
    print("detector: ", detector)

    #initialise directories for detectors to save
    if not os.path.exists("detectors/"+detector+"/plots"):
        os.makedirs("detectors/"+detector+"/plots")

    #read tier 2 runs for Ba data

    #.h5 files - do not exist for V05266A or V05268B
    #t2_folder_h5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v00.00/"
    # df_total_h5= read_all_dsp_h5(t2_folder_h5, cuts=False) 
    # print("df_total_h5: ", df_total_h5)
    #e_ftp_data = df_total_h5['e_ftp']

    #.lh5 files
    t2_folder_lh5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v01.00/"
    df_total_lh5 = read_all_dsp_lh5(t2_folder_lh5,cuts=False)
    print("df_total_lh5: ", df_total_lh5)
    print("df_total_lh5 keys: ", df_total_lh5.keys())
    trapE_data = df_total_lh5['trapE']

    key = "trapE" #change this if you want e_ftp instead
    key_data = trapE_data
    no_events = key_data.size #all events = the same as sum(counts)
    print("No. events [key_data.size]: ", no_events)

    no_bins = 25000 #10000 #7722

    #Plot full uncalibrated energy spectrum - e_ftp
    fig, ax = plt.subplots()
    counts, bins, bars = plt.hist(key_data, bins=no_bins)
    binwidth = bins[1] - bins[0]
    print("bin width uncal: ", binwidth, " uncal unit")
    plt.yscale("log")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.xlim(0, 40000)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'# events = $%.0f$' % (no_events), r'binwidth = $%.2f$' % (binwidth)))
    ax.text(0.67, 0.97, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.savefig("detectors/"+detector+"/plots/"+key+".png")

    fig, ax = plt.subplots()
    counts, bins, bars = plt.hist(key_data, bins=no_bins)
    plt.yscale("log")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.xlim(0, 7000) 
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'# events = $%.0f$' % (no_events), r'binwidth = $%.2f$' % (binwidth)))
    ax.text(0.67, 0.97, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.savefig("detectors/"+detector+"/plots/"+key+"_zoom.png")

    #___________Calibration__________
    print("Calibrating...")
 
    print("Calibration peaks:")

    #Fit known calibration peaks from: 
    #https://www.ezag.com/fileadmin/ezag/user-uploads/isotopes/isotopes/Isotrak/isotrak-pdf/Decay_Schema_Data/Ba-133.pdf

    truth_energies = np.array([81.0, 276.40,302.85,356.02,383.85]) #keV
    #rough guess on peak window:
    #peak_lims_guess = [[1277, 1315], [4380, 4450], [4805,4870], [5645,5725], [6090,6170]] #-i02160a, e_ftp
    peak_lims_guess = [[1270, 1315], [4340, 4430], [4760,4840], [5600,5700], [6020,6170]] #V05266A, trapE

    mu_peaks = []
    sigma_peaks = []
    mu_err_peaks = []
    sigma_err_peaks = []

    for index, truth in enumerate(truth_energies):        
        truth_str = str(int(truth))
        print("fitting peak: ", truth_str, " keV")
        xmin, xmax = peak_lims_guess[index][0], peak_lims_guess[index][1]
        print("rough peak boundaries: ", xmin, " , ", xmax)
        plt.figure()
        mu, sigma, mu_err, sigma_err = fit_peak(key, bins, counts, xmin, xmax)
        mu_peaks.append(mu)
        sigma_peaks.append(sigma)
        mu_err_peaks.append(mu_err)
        sigma_err_peaks.append(sigma_err)
        counts, bins, bars = plt.hist(key_data, bins=no_bins, histtype='step', color='grey')
        plt.xlim(xmin, xmax)        
        plt.ylim(10, 0.5*10**6)
        plt.title(truth_str+" keV peak [Uncalibrated]")
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/"+truth_str+"keV_peak_uncal.png")

    #Plot calibration curves:
    print("")
    print("Constructing calibration curve...")
    calibration_data, calibration_data_err = np.array(mu_peaks), np.array(mu_err_peaks)
    plt.figure()
    m, c, m_err, c_err, chi_sq, p_value, residuals, dof = calibration(calibration_data, calibration_data_err, truth_energies, fittype="linear_fit")
    for x, y in zip(truth_energies, calibration_data):
        truth_str = str(int(x))
        plt.annotate(truth_str, (x,y), textcoords="offset points", xytext=(-5,5), ha='center') # horizontal alignment can be left, right or center
    plt.savefig("detectors/"+detector+"/plots/calibration_curve_linear.png")

    #plot linear residuals
    plt.figure()
    residuals_err = [1.0]*len(residuals) #error on residuals is 1 by definition
    plt.errorbar(truth_energies, residuals, yerr = residuals_err, fmt ='bo') #, ms = 1.0, mew = 3.0)
    plt.ylabel("r=(data-fit)/error")
    plt.xlabel("Energy (keV)")
    plt.title("Residual Plot for Linear Calibration Graph")
    for x, y in zip(truth_energies, residuals):
        truth_str = str(int(x))
        plt.annotate(truth_str, (x,y), textcoords="offset points", xytext=(0,5), ha='center') # horizontal alignment can be left, right or center 
    plt.axhline(linewidth=2, color='black', dashes = (5,2,1,2))
    plt.savefig("detectors/"+detector+"/plots/calibration_residuals.png")

    #plot quadratic calibration
    plt.figure()
    a, b, c_quad, a_err, b_err, c_quad_err, chi_sq, p_value, residuals, dof = calibration(calibration_data, calibration_data_err, truth_energies, fittype="quadratic_fit")
    for x, y in zip(truth_energies, calibration_data):
        truth_str = str(int(x))
        plt.annotate(truth_str, (x,y), textcoords="offset points", xytext=(-5,5), ha='center') 
    plt.savefig("detectors/"+detector+"/plots/calibration_curve_quadratic.png")

    #plot quadratic residuals
    plt.figure()
    residuals_err = [1.0]*len(residuals) #error on residuals is 1 by definition
    plt.errorbar(truth_energies, residuals, yerr = residuals_err, fmt ='bo') #, ms = 1.0, mew = 3.0)
    plt.ylabel("r=(data-fit)/error")
    plt.xlabel("Energy (keV)")
    plt.title("Residual Plot for Quadratic Calibration Graph")
    for x, y in zip(truth_energies, residuals):
        truth_str = str(int(x))
        plt.annotate(truth_str, (x,y), textcoords="offset points", xytext=(0,5), ha='center') # horizontal alignment can be left, right or center 
    plt.axhline(linewidth=2, color='black', dashes = (5,2,1,2))
    plt.savefig("detectors/"+detector+"/plots/calibration_residuals_quadratic.png")


    #Save calibration coefficients to json file
    calibration_coef_dict = {
        "m": m,
        "m_err" : m_err,
        "c" : c,
        "c_err" : c_err,
        "a_quad" : a,
        "a_quad_err" : a_err,
        "b_quad" : b,
        "b_quad_err" : b_err,
        "c_quad" : c_quad,
        "c_quad_err" : c_quad_err
    }
    with open("detectors/"+detector+"/calibration_coef_"+key+".json", "w") as outfile: 
        json.dump(calibration_coef_dict, outfile)


    plt.close('all') #Cant keep more than 20 figures open

    #_________Replot calibrated spectra and peaks_________

    #Linearly calibrated data:
    print("")
    print("Linearly calibrating energy...")

    calibrated_energy = (key_data-c)/m
    binwidth = 0.15 #0.1 #keV - rough min resolution
    bins = np.arange(min(calibrated_energy), max(calibrated_energy) + binwidth, binwidth)

    #bin_width = 0.5 #0.5 kev = resolution #plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    #no_bins_ideal = int(max(calibrated_energy/bin_width))
    #print("ideal no bins: ", no_bins_ideal) #=7722

    fig, ax = plt.subplots()
    counts, bins_cal, bars = plt.hist(calibrated_energy, bins=bins)
    no_events = calibrated_energy.size
    print("No. events [cal_energy.size]: ", no_events)
    plt.xlabel("Energy (KeV)")
    plt.ylabel("Counts")
    plt.xlim(0, 2500)
    plt.yscale("log")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'# events = $%.0f$' % (no_events), r'binwidth = $%.2f$ keV' % (binwidth)))
    ax.text(0.67, 0.97, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.savefig("detectors/"+detector+"/plots/calibrated_energy.png") 

    #plot zoomed in
    fig, ax = plt.subplots()
    counts, bins_cal, bars = plt.hist(calibrated_energy, bins=bins)
    plt.xlabel("Energy (KeV)")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.xlim(0,450)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'# events = $%.0f$' % (no_events), r'binwidth = $%.2f$ keV' % (binwidth)))
    ax.text(0.67, 0.97, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.savefig("detectors/"+detector+"/plots/calibrated_energy_zoom.png") 

    print("")
    print("Replotting calibrated spectra and peaks...")

    mu_cal_peaks = []
    sigma_cal_peaks = []
    mu_err_cal_peaks = []
    sigma_err_cal_peaks = []

    for index, truth in enumerate(truth_energies):        
        truth_str = str(int(truth))
        print("fitting calibrated peak: ", truth_str, " keV")
        xmin, xmax = (peak_lims_guess[index][0]-c)/m, (peak_lims_guess[index][1]-c)/m
        print("rough peak boundaries: ", xmin, " , ", xmax)
        plt.figure()
        mu, sigma, mu_err, sigma_err = fit_peak("Energy (keV)", bins_cal, counts, xmin, xmax)
        mu_cal_peaks.append(mu)
        sigma_cal_peaks.append(sigma)
        mu_err_cal_peaks.append(mu_err)
        sigma_err_cal_peaks.append(sigma_err)
        counts, bins, bars = plt.hist(calibrated_energy, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin, xmax)
        plt.ylim(10, 0.5*10**6)
        plt.title(truth_str+" keV peak [Calibrated]")
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/"+truth_str+"keV_peak.png")


    #______Correct resolution plot_______
    print("")
    print("Constructing calibrated resolution plot...")

    energies, energies_err = mu_cal_peaks, mu_err_cal_peaks
    FWHM, FWHM_err = 2*np.sqrt(2*np.log(2))*(np.array(sigma_cal_peaks)), 2*np.sqrt(2*np.log(2))*(np.array(sigma_err_cal_peaks))
    A, offset, A_err, offset_err, chi_sq, p_value, residuals, dof = resolution_plot(energies, energies_err, FWHM, FWHM_err)

    for x, y in zip(energies, FWHM):
        index = np.where(energies == x)[0][0]
        truth_str = str(int(truth_energies[index]))
        plt.annotate(truth_str, (x,y), textcoords="offset points", xytext=(-10,5), ha='center') # horizontal alignment can be left, right or center
    plt.savefig("detectors/"+detector+"/plots/resolution_plot.png")

    #calculate estimated resolution at qbb=2039keV
    resolution_qbb_est = A*np.sqrt(2039 + offset)
    print("Estimated resolution at qbb=2039 keV: ", resolution_qbb_est, " keV") #=1.87 keV, legend target = 2-3kev

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


def gaussian(x,a,b,c,d):
    "gaussian function with offset d"
    f = a*np.exp(-((x-b)**2.0/(2.0*c**2.0))) +d
    return f

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
    f = a*np.sqrt(x + c)
    return f

def chi_sq_calc(xdata, ydata, yerr, fit_func, popt):
    "calculate chi sq and p-val of a fit given the data points and fit parameters, e.g. fittype ='linear'"
   
    y_obs = ydata
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


def fit_peak(key, bins, counts, xmin, xmax): #p_guess):
    "fit a gaussian to a peak and return fit parameters"

    no_bins = bins.size 
    binwidth = bins[1]-bins[0]

    xdata = []
    ydata = []
    for bin in bins:
        bin_centre = bin + 0.5*(max(bins)-(min(bins)))/no_bins #this leads to incorrect indexing so leave out for now
        if bin_centre < xmax and bin_centre > xmin:
            xdata.append(bin_centre)
            bin_index = np.where(bins == bin)[0][0]
            ydata.append(counts[bin_index])

    xdata = np.array(xdata)
    ydata = np.array(ydata)     

    yerr = np.sqrt(ydata) #counting error

    #initial rough guess of gaussian params
    aguess = max(ydata) - min(ydata)
    aguess_index = np.where(ydata == max(ydata))[0][0]
    bguess = xdata[aguess_index]
    cguess = (xmax-xmin)/2
    dguess = min(ydata)
    p_guess = [aguess, bguess, cguess, dguess]
    bounds=(0, [np.inf, np.inf, np.inf, np.inf])
    sigma = []
    for index, i in enumerate(yerr):    
        if i != 0:
            sigma.append(yerr[index])
        else:
            sigma.append(1) #just to prevent errors...
    sigma = np.array(sigma)
    popt, pcov = optimize.curve_fit(gaussian, xdata, ydata, p0=p_guess, sigma = sigma, maxfev = 1000000, method ="trf", bounds = bounds)
    mu, sigma = np.abs(popt[1]), np.abs(popt[2]) #must be positive
    mu_err, sigma_err = np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    
    fig, ax = plt.subplots()
    ax.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "Data", elinewidth = 1, fmt='bo', ms = 0.75, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    plt.plot(xfit, gaussian(xfit,*popt), "g", label = "Gaussian fit")
    plt.xlabel(key)
    plt.ylabel("Counts")
    plt.legend()

    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, gaussian, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if key == "Energy (keV)":
        info_str = '\n'.join((r'$\mu=%.2f \pm %.2f$' % (mu, mu_err, ), r'$\sigma=%.2f \pm %.2f$' % (np.abs(sigma), sigma_err,), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof),r'binwidth = $%.2g$ keV'%binwidth)) #, r'$p=%.3g$'%p_value))
    else: 
        info_str = '\n'.join((r'$\mu=%.2f \pm %.2f$' % (mu, mu_err, ), r'$\sigma=%.2f \pm %.2f$' % (np.abs(sigma), sigma_err,), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof),r'binwidth = $%.2g$'%binwidth)) #, r'$p=%.3g$'%p_value))
    ax.text(0.01, 0.99, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)

    FWHM = 2*np.sqrt(2*np.log(2))*sigma #gaussian FWHM relationship
    peak_range = [mu-FWHM, mu+FWHM]

    return mu, sigma, mu_err, sigma_err


def calibration(data, data_err, truth, fittype):
    "calibrate energy data for given peaks, fittype = linear_fit or quadratic_fit"
    xdata = truth
    ydata = data
    yerr = data_err

    if fittype == "linear_fit":
        popt, pcov = optimize.curve_fit(linear_fit, xdata, ydata, sigma = yerr, maxfev = 1000000, method ="trf")
        m, c = popt[0], popt[1]
        m_err, c_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
        fig, ax = plt.subplots()
        xfit = np.linspace(min(xdata), max(xdata), 1000)
        yfit = linear_fit(xfit,*popt)
        plt.plot(xfit, yfit, "g", label = "linear fit")
        ax.errorbar(truth, data, xerr=0, yerr =yerr, elinewidth = 1, fmt='x', ms = 1.5, mew = 4.0, label = "callibration peaks")
        plt.xlabel("truth (keV)")
        plt.ylabel("calibration peaks")
        plt.legend(loc = 4)

        chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, linear_fit, popt)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        info_str = '\n'.join((r'$m = %.3f \pm %.3f$' % (m, m_err, ), r'$c = %.3f \pm %.3f$' % (c, c_err,), r'$\chi^2/dof=%.0f/%.0f$'%(chi_sq, dof)))# r'$\chi^2 = %.3f$'%(chi_sq), r'$p = %.3g$'%(p_value)))
        ax.text(0.05, 0.95, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)

        return m, c, m_err, c_err , chi_sq, p_value, residuals, dof

    if fittype == "quadratic_fit":
        popt, pcov = optimize.curve_fit(quadratic_fit, xdata, ydata, sigma = yerr, maxfev = 1000000, method ="trf")
        a,b,c = popt[0], popt[1], popt[2]
        a_err, b_err, c_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
        fig, ax = plt.subplots()
        xfit = np.linspace(min(xdata), max(xdata), 1000)
        yfit = quadratic_fit(xfit,*popt)
        plt.plot(xfit, yfit, "g", label = "quadratic fit")
        ax.errorbar(truth, data, xerr=0, yerr =yerr, elinewidth = 1, fmt='x', ms = 1.5, mew = 4.0, label = "callibration peaks")
        plt.xlabel("truth (keV)")
        plt.ylabel("calibration peaks")
        plt.legend(loc = 4)

        chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, quadratic_fit, popt)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        info_str = '\n'.join((r'$a = %.3g \pm %.3g$' % (a, a_err, ), r'$b = %.3f \pm %.3f$' % (b, b_err,), r'$c = %.3f \pm %.3f$' % (c, c_err,),  r'$\chi^2/dof=%.0f/%.0f$'%(chi_sq, dof))) #, r'$p = %.3g$'%(p_value)))
        ax.text(0.05, 0.95, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)

        return a, b, c, a_err, b_err, c_err, chi_sq, p_value, residuals, dof


def resolution_plot(energies, energies_err, FWHM, FWHM_err):
    "Construct and fit resolution plot, x=energies, y =FWHM"
    
    Aguess = max(FWHM) - min(FWHM)
    offset_guess = 0
    p_guess = [Aguess, offset_guess]
    bounds=(0, [np.inf, np.inf])
    sigma = []
    for index, i in enumerate(FWHM_err):    
        if i != 0:
            sigma.append(FWHM_err[index])
        else:
            sigma.append(1) #just to prevent errors...
    sigma = np.array(sigma)
    popt, pcov = optimize.curve_fit(sqrt_curve, energies, FWHM, p0=p_guess, sigma = sigma, maxfev = 1000000, method ="trf", bounds = bounds)
    A, offset = popt[0], popt[1] #must be positive
    A_err, offset_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])

    fig, ax = plt.subplots()
    ax.errorbar(energies, FWHM, xerr=energies_err, yerr = FWHM_err, label = "Data", fmt='o') #, ms=1.5) #, elinewidth = 1, fmt='x', ms = 1.5, mew = 4.0)
    xfit = np.linspace(min(energies), max(energies), 1000)
    plt.plot(xfit, sqrt_curve(xfit,*popt), "g", label = "$A*\sqrt{x +c}$ fit")
    plt.xlabel("Energy (keV)")
    plt.ylabel("FWHM (keV)")
    plt.legend()
    chi_sq, p_value, residuals, dof = chi_sq_calc(energies, FWHM, FWHM_err, sqrt_curve, popt)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$A=%.3f \pm %.3f$' % (A, A_err, ), r'$c=%.3f \pm %.3f$' % (offset, offset_err,), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof))) #, r'$p=%.3g$'%p_value))
    ax.text(0.65, 0.20, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)

    return A, offset, A_err, offset_err, chi_sq, p_value, residuals, dof


if __name__ =="__main__":
    main()