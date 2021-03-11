import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("mplstyle.txt")
from datetime import datetime
import argparse
 
#import fitting functions
import sys
sys.path.append('../data/')
from Ba133_data_AV_analysis import * 

#OLD SCRIPT DO NOT USE

def main(): 

    #print date and time for log:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")


    parser = argparse.ArgumentParser(description='Fit and count MC gamma line for Ba for a particular detector, with cuts or not')
    #parser.add_argument('--simID', action="store_true", default="IC160A_ba_top_81mmNEW8_01_newresolution")
    parser.add_argument('--simID', action="store_true", default="IC160A-BA133-uncollimated-top-run0003-81z-newgeometry_g")
    parser.add_argument('--detector', action="store_true", default="I02160A")
    parser.add_argument('--FCCD', action="store_true", default = 0.71)
    parser.add_argument('--cuts', action="store", type=bool, default = False)
    
    args = parser.parse_args()
    MC_file_id, detector, cuts = args.simID, args.detector, args.cuts
    print("MC file ID: ", MC_file_id)
    print("detector: ", detector)
    print("fixed FCCD: ", str(FCCD))
    print("applying cuts: ", str(cuts))

    print("start...")

    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/processed/" #path to processed MC hdf5 files

   
    # MC_file = hdf5_path+"processed_detector_"+MC_file_id+'.hdf5'

    binwidth = 0.15 #keV

    #_____________PROCESS AND PLOT FCCDS_____________ 

    print("")
    print("Process for each DLF...")

    DLF_list = [0.0,0.25, 0.5, 0.75, 1.0]

    DLF_energies_list = []
    R_simdata_356_counts_list = []
    R_simdata_356_counts_cuts_list = []
    for DLF in DLF_list:

        print("")
        print("DLF: ", DLF)
        energies_TLDL, energy_data, energy_data_cuts, R_simdata_356_counts,R_simdata_356_counts_cuts = process_FCCDs(MC_file_id, FCCD, DLF, binwidth)
        #bins = np.arange(min(energies_TLDL), max(energies_TLDL) + binwidth, binwidth)
        DLF_energies_list.append(energies_TLDL)
        R_simdata_356_counts_list.append(R_simdata_356_counts)
        R_simdata_356_counts_cuts_list.append(R_simdata_356_counts_cuts)
    
    plt.close("all")


    #plot - NO CUTS
    fig, ax = plt.subplots()
    bins = np.arange(0, 450, binwidth)
    R_simdata_356_noTL = R_simdata_356_counts_list[-1] #for DLF = 1
    print("Rsimdata no Tl; ",R_simdata_356_noTL)
    for index, energies in enumerate(DLF_energies_list):
        DLF = DLF_list[index]
        plt.hist(energies, bins = bins, label ='MC: DLF: '+str(DLF)+' (scaled)', histtype = 'step', linewidth = '0.35',weights=(1/R_simdata_356_noTL)*np.ones_like(energies))

    plt.hist(energy_data, bins = bins, label ='Data', histtype = 'step', linewidth = '0.35')
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.xlim(0, 450)
    plt.yscale("log")
    plt.legend() 
    plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_allDLFs_scaledsame.png')


    fig, ax = plt.subplots()
    bins = np.arange(0, 450, binwidth)
    R_simdata_356_noTL = R_simdata_356_counts_list[-1] #for DLF = 1
    for index, energies in enumerate(DLF_energies_list):
        DLF = DLF_list[index]
        plt.hist(energies, bins = bins, label ='MC: DLF: '+str(DLF)+' (scaled)', histtype = 'step', linewidth = '0.35',weights=(1/R_simdata_356_noTL)*np.ones_like(energies))

    plt.hist(energy_data, bins = bins, label ='Data', histtype = 'step', linewidth = '0.35')
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.xlim(0, 100)
    plt.ylim(100, 5*10**5)
    plt.yscale("log")
    plt.legend()
    plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_allDLFs_scaledsame_zoomed.png')


    #PLOT - CUTS
    fig, ax = plt.subplots()
    bins = np.arange(0, 450, binwidth)
    R_simdata_356_noTL_cuts = R_simdata_356_counts_cuts_list[-1]
    print("Rsimdata no Tl cuts; ",R_simdata_356_noTL_cuts)
    for index, energies in enumerate(DLF_energies_list):
        DLF = DLF_list[index] 
        plt.hist(energies, bins = bins, label ='MC: DLF: '+str(DLF)+' (scaled, cuts)', histtype = 'step', linewidth = '0.35',weights=(1/R_simdata_356_noTL_cuts)*np.ones_like(energies))

    plt.hist(energy_data, bins = bins, label ='Data (cuts)', histtype = 'step', linewidth = '0.35')
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.xlim(0, 450)
    plt.yscale("log")
    plt.legend()
    plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_allDLFs_scaledsame_cuts.png')



    plt.show()

    print("done")


def process_FCCDs(MC_file_id, FCCD, DLF, binwidth):
    "process and plot for a list of different FCCDs"

    #hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/processed/"
    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/processed/" #path to processed MC hdf5 files


    # #_______plot full spectrum___________
    # print("plotting whole simulated spectrum...")

    MC_file = hdf5_path+"processed_detector_"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'.hdf5'
    print("MC file: ",MC_file)
    df =  pd.read_hdf(MC_file, key="procdf")
    energies = df['energy']
    #energies = energies*1000 #dont need anymore
    no_events = energies.size #=sum(counts)
    print("No. events: ", no_events) 
    #bins = np.arange(min(energies), max(energies) + binwidth, binwidth)
    bins = np.arange(0, 450, binwidth)

    # fig, ax = plt.subplots()
    # counts, bins, bars = plt.hist(energies, bins = bins) #, linewidth = '0.35')
    # plt.xlabel("Energy [keV]")
    # plt.ylabel("Counts")
    # plt.xlim(0, 450)
    # plt.yscale("log")
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # info_str = '\n'.join((r'# events = $%.0f$' % (no_events), r'binwidth = $%.2f$ keV' % (binwidth)))
    # ax.text(0.67, 0.97, info_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    # plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'.png')

    #________Fit peaks of interest_______
    print("")
    print("Fitting peaks of interest...")

    xmin_356, xmax_356 = 350, 362 #362 #360.5 for gammas #2 #360 #kev 
    plt.figure()
    counts, bins, bars = plt.hist(energies, bins = bins, histtype = 'step') #, linewidth = '0.35')
    popt, pcov, xfit = fit_peak_356_2("Energy (keV)", bins, counts, xmin_356, xmax_356)
    a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]
    amplitude356_sim = gaussian_and_bkg_2(b, a, b, c, d, e)
    plt.xlim(xmin_356, xmax_356) 
    plt.ylim(10, 10**7)
    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'_356keV.png')

    C_356, C_356_err = gauss_count(a, b, c, np.sqrt(pcov[0][0]),  np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
    print("gauss count 356keV: ", C_356, " +/- ", C_356_err )


    xmin_81, xmax_81 = 77, 84
    plt.figure()
    counts, bins, bars = plt.hist(energies, bins = bins, histtype = 'step') #, linewidth = '0.35')
    popt, pcov, xfit = fit_double_peak_81("Energy (keV)", bins, counts, xmin_81, xmax_81)
    a,b,c,d,e,f,g,h = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7] 
    plt.xlim(xmin_81, xmax_81) 
    plt.ylim(5*10**2, 5*10**6)
    #plt.ylim(10**3, 10**7) #gammas_81mmNEW
    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'_81keV.png')

    R = 2.65/32.9
    C_81, C_81_err = gauss_count(a, b, c, np.sqrt(pcov[0][0]),  np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
    C_79, C_79_err = gauss_count(R*a, d, e, R*np.sqrt(pcov[0][0]),  np.sqrt(pcov[3][3]), np.sqrt(pcov[4][4]), binwidth)
    print("gauss count 81: ", C_81, " +/- ", C_81_err )
    print("gauss count 79.6: ", C_79, " +/- ", C_79_err )

    print("")
    O_Ba133 = (C_79 + C_81)/C_356
    O_Ba133_err = O_Ba133*np.sqrt((C_79_err**2 + C_81_err**2)/(C_79+C_81)**2 + (C_356_err/C_356)**2)
    print("O_BA133 = " , O_Ba133, " +/- ", O_Ba133_err)

    plt.close('all')

    #__________compare against real data__________
    print("")
    print("plotting simulation against actual data...")

    #this code below is from Ba133_dlt_analysis.py
    detector = "I02160A" #read tier 2 runs for Ba data
    t2_folder = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v00.00/"
    keys, data = read_all_t2(t2_folder)
    data_size = data.size #all events
    key_data = obtain_key_data(data, keys, "e_ftp", data_size)
    
    t2_folder_h5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v00.00/"
    passed_cuts = json.load(open('/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/data/passed_cuts_data.json','r')) #passed cuts
    df_total_h5, df_total_cuts_h5 = read_all_dsp_h5(t2_folder_h5, passed_cuts)
    e_ftp_data = df_total_h5['e_ftp']
    e_ftp_data_cuts = df_total_cuts_h5['e_ftp']

    with open('/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/data/calibration_coef.json') as json_file:
        calibration_coefs = json.load(json_file)
        m = calibration_coefs['m']
        m_err = calibration_coefs['m_err']
        c = calibration_coefs['c']
        c_err = calibration_coefs['c_err']
        # a_quad = calibration_coefs['a_quad']
        # a_quad_err = calibration_coefs['a_quad_err']
        # b_quad = calibration_coefs['b_quad']
        # b_quad_err = calibration_coefs['b_quad_err']
        # c_quad = calibration_coefs['c_quad']
        # c_quad_err = calibration_coefs['c_quad_err']

    
    energy_data= (e_ftp_data-c)/m
    energy_data_cuts= (e_ftp_data_cuts-c)/m

    #calibrated_energy_data = (key_data-c)/m

    # #scale up data to same amplitude 356 peak as simulation - old
    # amplitude356_data = gaussian_and_bkg_2(356, 4.6*10**4, 356, 0.423, 2.64, 205) #from old plots
    # print("amplitude 356keV data: ", amplitude356_data)
    # print("amplitude 356keV simulation: ", amplitude356_sim)
    # R_simdata_356 = amplitude356_sim/amplitude356_data #ratio of sim to data for 356 peak amplitude
    # print(R_simdata_356)

    #NEW - instead scale up data to same peak integrated counts as simulation
    with open('/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/data/dlt_observables.json') as json_file:
        dlt_observables = json.load(json_file)
        C_356_data = dlt_observables['C_356_average']
    
    with open('/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/data/dlt_observables_cuts.json') as json_file:
        dlt_observables = json.load(json_file)
        C_356_data_cuts = dlt_observables['C_356_average']

    C_356_sim = C_356
    print("integral counts 356keV data: ", C_356_data)
    print("integral counts 356keV data cuts: ", C_356_data_cuts)
    print("integral counts 356keV simulation: ", C_356_sim)
    R_simdata_356_counts = C_356_sim/C_356_data #ratio of sim to data for 356 peak counts
    print(R_simdata_356_counts)
    R_simdata_356_counts_cuts = C_356_sim/C_356_data_cuts #ratio of sim to data for 356 peak counts
    print(R_simdata_356_counts_cuts)

    fig, ax = plt.subplots()
    bins_data = bins = np.arange(0, 450, binwidth)
    counts_data, bins_cal_data, bars_data = plt.hist(energy_data, bins=bins_data,  label = "Data", histtype = 'step', linewidth = '0.35')
    counts_data_cuts, bins_cal_data_cuts, bars_data_cuts = plt.hist(energy_data_cuts, bins=bins_data,  label = "Data (cuts)", histtype = 'step', linewidth = '0.35')
    #counts, bins, bars = plt.hist(energies, bins = bins, weights=(1/R_simdata_356)*np.ones_like(energies), label = "MC: FCCD "+str(FCCD)+"mm, DLF "+str(DLF)+" (scaled)", histtype = 'step', linewidth = '0.35')
    counts, bins, bars = plt.hist(energies, bins = bins, weights=(1/R_simdata_356_counts)*np.ones_like(energies), label = "MC: FCCD "+str(FCCD)+"mm, DLF "+str(DLF)+" (scaled)", histtype = 'step', linewidth = '0.35')
    counts, bins, bars = plt.hist(energies, bins = bins, weights=(1/R_simdata_356_counts_cuts)*np.ones_like(energies), label = "MC: FCCD "+str(FCCD)+"mm, DLF "+str(DLF)+" (scaled, cuts)", histtype = 'step', linewidth = '0.35')
    #counts_scaled = np.array(counts)*(1/R_simdata_356)
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.xlim(0, 450)
    plt.yscale("log")
    plt.legend(loc = "lower left")
    #plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_DATAcomparison_scaledMC.png")
    plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/TLDL_analysis/plots/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_DATAcomparison_scaledMC_cuts.png")



    return energies, energy_data, energy_data_cuts, R_simdata_356_counts,R_simdata_356_counts_cuts #for creating comparison graph


    # return energies, R_simdata_356_counts, energy_data #for creating comparison graph



if __name__ == "__main__":
    main()