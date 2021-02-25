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

#considers DL only, no TL
#bkg run code not included since it is insignificant for Ba - but look at old code if needed

def main(): 

    
    #print date and time for log: 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    parser = argparse.ArgumentParser(description='Fit and count MC gamma line for Ba for a particular detector, with cuts or not')
    parser.add_argument('--simID', action="store_true", default="IC160A_ba_top_81mmNEW8_01_newresolution")
    parser.add_argument('--detector', action="store_true", default="I02160A")
    parser.add_argument('--cuts', action="store", type=bool, default = False)
    args = parser.parse_args()
    MC_file_id, detector, cuts = args.simID, args.detector, args.cuts
    print("MC file ID: ", MC_file_id)
    print("detector: ", detector)
    print("applying cuts: ", str(cuts))

    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/processed/" #path to processed MC hdf5 files
    MC_file = hdf5_path+"processed_detector_"+MC_file_id+'.hdf5' #no FCCD
    binwidth = 0.15 #keV

    if not os.path.exists("detectors/"+detector+"/plots"):
        os.makedirs("detectors/"+detector+"/plots")


    #_____________PROCESS AND PLOT FCCDS_____________ 

    print("")
    print("Process each DLT (no TL)...")

    FCCD_list = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3] #make this an input argument
    FCCD_list = [0.744] #best fit, no cuts
    FCCD_list = [0.698] #best fit, cuts
    for FCCD in FCCD_list:
        print("")
        print("FCCD: ", FCCD)
        print("")
        energies_FCCD, R_simdata_356_FCCD = process_FCCDs(FCCD, MC_file_id, detector, cuts, hdf5_path, binwidth)
            

    print("done")


def process_FCCDs(FCCD, MC_file_id, detector, cuts, hdf5_path, binwidth):
    "process and plot for different FCCDs"

    #_______plot full spectrum___________
    print("plotting whole simulated spectrum...")

    if FCCD == 0:
        MC_file = hdf5_path+"processed_detector_"+MC_file_id+'.hdf5'
    else:
        MC_file = hdf5_path+"processed_detector_"+MC_file_id+'_FCCD'+str(FCCD)+'mm.hdf5'

    df =  pd.read_hdf(MC_file, key="procdf")
    energies = df['energy']
    energies = energies*1000
    no_events = energies.size #=sum(counts)
    print("No. events: ", no_events) 
    bins = np.arange(min(energies), max(energies) + binwidth, binwidth)


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
    plt.savefig("detectors/"+detector+"/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_356keV.png')

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
    plt.savefig("detectors/"+detector+"/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_81keV.png')

    R = 2.65/32.9
    C_81, C_81_err = gauss_count(a, b, c, np.sqrt(pcov[0][0]),  np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
    C_79, C_79_err = gauss_count(R*a, d, e, R*np.sqrt(pcov[0][0]),  np.sqrt(pcov[3][3]), np.sqrt(pcov[4][4]), binwidth)
    print("gauss count 81: ", C_81, " +/- ", C_81_err )
    print("gauss count 79.6: ", C_79, " +/- ", C_79_err )

    print("")
    O_Ba133 = (C_79 + C_81)/C_356
    O_Ba133_err = O_Ba133*np.sqrt((C_79_err**2 + C_81_err**2)/(C_79+C_81)**2 + (C_356_err/C_356)**2)
    print("O_BA133 = " , O_Ba133, " +/- ", O_Ba133_err)


    #fit other gaussian gamma peaks
    peak_ranges = [[159,162],[221.5,225],[274,279],[300,306],[381,386]] #Rough by eye
    peaks = [161, 223, 276, 303, 383]
    other_peak_counts = []
    other_peak_counts_err = []
    for index, i in enumerate(peak_ranges):

        print(str(peaks[index]), " keV")

        if peaks[index]==383 and FCCD == 3: #problems converging for this particular histogram
            C, C_err = float("nan"), float("nan")
            print("gauss count: ", C, " +/- ", C_err )
            other_peak_counts.append(C)
            other_peak_counts_err.append(C_err)
            continue

        
        plt.figure()
        xmin, xmax = i[0], i[1]
        popt, pcov, xfit = fit_peak_356_2("Energy (keV)", bins, counts, xmin, xmax)
        a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]
        counts, bins, bars = plt.hist(energies, bins=bins, histtype='step', color='grey')
        plt.xlim(xmin, xmax) 
        #plt.ylim(min(counts)/10, max(counts)*10)
        #plt.ylim(0.5*100, 5*10**5)
        #plt.ylim(10, 10**5) #0.05
        plt.yscale("log")
        plt.savefig("detectors/"+detector+"/plots/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_'+str(peaks[index])+'keV.png')
        C, C_err = gauss_count(a,b, c, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
        print("gauss count: ", C, " +/- ", C_err )
        other_peak_counts.append(C)
        other_peak_counts_err.append(C_err)


    plt.close('all')

    #__________compare against real data__________
    print("")
    print("plotting simulation against actual data...")

    t2_folder_h5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v00.00/"
    #t2_folder_lh5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v01.00/"

    with open("../data/detectors/"+detector+"/calibration_coef.json") as json_file:
        calibration_coefs = json.load(json_file)
        m = calibration_coefs['m']
        m_err = calibration_coefs['m_err']
        c = calibration_coefs['c']
        c_err = calibration_coefs['c_err']


    if cuts == False:
    
        df_total_h5 = read_all_dsp_h5(t2_folder_h5, cuts)
        e_ftp_data = df_total_h5['e_ftp']
        # df_total_lh5 = read_all_dsp_lh5(t2_folder_lh5,cuts)
        # trapE_data = df_total_lh5['trapE']

        energy_data = (e_ftp_data-c)/m
        #bins = np.arange(min(energy_data), max(energy_data) + binwidth, binwidth)
        #counts_energy_data, bins, bars = plt.hist(energy_data, bins=bins, label = "no cuts")

        #plot absolutes
        # bins_data = bins = np.arange(0, 450, binwidth)
        # fig, ax = plt.subplots()
        # counts_data, bins_cal_data, bars_data = plt.hist(calibrated_energy_data, bins=bins_data, label = "data", histtype = 'step', linewidth = '0.35')
        counts, bins, bars = plt.hist(energies, bins = bins, label = "MC: FCCD "+str(FCCD)+"mm", histtype = 'step', linewidth = '0.35')
        print("counts not scaled:")
        print(counts)
        # plt.xlabel("Energy [keV]")
        # plt.ylabel("Counts")
        # plt.xlim(0, 450)
        # plt.yscale("log")
        # plt.legend(loc = "lower left")
        # plt.savefig("/lfs/l1/legend/users/aalexander/HADES_detchar/Ba133_analysis/simulations/IC-legend/macro/ba_top/PostProc/plots/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DATAcomparison.png")

        #scale up data to same amplitude 356 peak as simulation
        with open("../data/detectors/"+detector+"/dlt_observables.json") as json_file:
            dlt_observables = json.load(json_file)
            C_356_data = dlt_observables['C_356_average']

        C_356_sim = C_356
        print("integral counts 356keV data: ", C_356_data)
        print("integral counts 356keV simulation: ", C_356_sim)
        R_simdata_356_counts = C_356_sim/C_356_data #ratio of sim to data for 356 peak counts
        print(R_simdata_356_counts)

        fig, ax = plt.subplots()
        #bins_data = bins = np.arange(0, 450, binwidth)
        counts_data, bins, bars_data = plt.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')
        counts, bins, bars = plt.hist(energies, bins = bins, weights=(1/R_simdata_356_counts)*np.ones_like(energies), label = "MC: FCCD "+str(FCCD)+"mm (scaled)", histtype = 'step', linewidth = '0.35')
        print("counts scaled")
        print(counts)
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.xlim(0, 450)
        plt.yscale("log")
        plt.legend(loc = "lower left")
        plt.savefig("detectors/"+detector+"/plots/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_datacomparison.png")

    else:
        passed_cuts = json.load(open('/lfs/l1/legend/users/aalexander/large_files/cuts/'+detector+'/passed_cuts_data.json','r')) #passed cuts
        df_total_cuts_h5 = read_all_dsp_h5(t2_folder_h5,cuts, passed_cuts = passed_cuts)
        e_ftp_data_cuts = df_total_cuts_h5['e_ftp']
        # df_total_cuts_lh5 = read_all_dsp_lh5(t2_folder_lh5, cuts, passed_cuts=passed_cuts)
        # trapE_data_cuts = df_total_cuts_lh5['trapE']
    
        energy_data_cuts= (e_ftp_data_cuts-c)/m
        #bins = np.arange(min(energy_data_cuts), max(energy_data_cuts) + binwidth, binwidth)
        #counts_energy_data_cuts, bins_cuts, bars_cuts = plt.hist(energy_data_cuts, bins=bins, label = "pile up cuts")

        #scale up data to same amplitude 356 peak as simulation
        with open("../data/detectors/"+detector+"/dlt_observables_cuts.json") as json_file:
            dlt_observables = json.load(json_file)
            C_356_data = dlt_observables['C_356_average']

        C_356_sim = C_356
        print("integral counts 356keV data: ", C_356_data)
        print("integral counts 356keV simulation: ", C_356_sim)
        R_simdata_356_counts = C_356_sim/C_356_data #ratio of sim to data for 356 peak counts
        print(R_simdata_356_counts)

        fig, ax = plt.subplots()
        #bins_data = bins = np.arange(0, 450, binwidth)
        counts_data_cuts, bins, bars_data = plt.hist(energy_data_cuts, bins=bins,  label = "Data (cuts)", histtype = 'step', linewidth = '0.35')
        counts, bins, bars = plt.hist(energies, bins = bins, weights=(1/R_simdata_356_counts)*np.ones_like(energies), label = "MC: FCCD "+str(FCCD)+"mm (scaled)", histtype = 'step', linewidth = '0.35')
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.xlim(0, 450)
        plt.yscale("log")
        plt.legend(loc = "lower left")
        plt.savefig("detectors/"+detector+"/plots/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_datacomparison_cuts.png")

    
    #Save count values to json file
    dlt_observables = {
        "C_356": C_356,
        "C_356_err" : C_356_err,
        "C_81" : C_81,
        "C_81_err" : C_81_err,
        "C_79" : C_79,
        "C_79_err" : C_79_err,
        "O_Ba133" : O_Ba133,
        "O_Ba133_err" : O_Ba133_err,
        "C_161" : other_peak_counts[0],
        "C_161_err" : other_peak_counts_err[0],
        "C_223" : other_peak_counts[1],
        "C_223_err" : other_peak_counts_err[1],
        "C_276" : other_peak_counts[2],
        "C_276_err" : other_peak_counts_err[2],
        "C_303" : other_peak_counts[3],
        "C_303_err" : other_peak_counts_err[3],
        "C_383" : other_peak_counts[4],
        "C_383_err" : other_peak_counts_err[4],
        "R_simdata_356_counts": R_simdata_356_counts
    }

    if cuts == False:
        with open("detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_dlt_observables.json", "w") as outfile: 
            json.dump(dlt_observables, outfile)
    else:
        with open("detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_dlt_observables_cuts.json", "w") as outfile: 
            json.dump(dlt_observables, outfile)


    #calculate DATA/MC for each energy bin and export
    #only do this for best fit FCCD
    if FCCD == 0.744 or FCCD == 0.698:
        print("")
        print("calculating data/MC ratios...")

        Data_MC_ratios = []
        Data_MC_ratios_err = []
        # print(len(counts_data))
        # print(len(bins_data))
        for index, bin in enumerate(bins[1:]):

            if cuts == False:
                data = counts_data[index]
            else:
                data = counts_data_cuts[index]

            MC = counts[index] #This counts has already been scaled by weights
            #MC = counts_scaled[index]

            if MC == 0:
                ratio = 0.
                error = 0.
            else:
                try: 
                    ratio = data/MC
                    try: 
                        error = np.sqrt(1/data + 1/MC)
                    except:
                        error = 0.
                except:
                    ratio = 0 #if MC=0 and dividing by 0
            Data_MC_ratios.append(ratio)
            Data_MC_ratios_err.append(error)

        Data_MC_ratios_df = pd.DataFrame({'ratio': Data_MC_ratios})
        Data_MC_ratios_df['ratio_err'] = Data_MC_ratios_err
        Data_MC_ratios_df['bin'] = bins[1:]
        print(Data_MC_ratios_df)

        if cuts == False:
            Data_MC_ratios_df.to_csv("detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DataMCRatios.csv", index=False)
        else:
            Data_MC_ratios_df.to_csv("detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DataMCRatios_cuts.csv", index=False)

    return energies, R_simdata_356_counts #for creating comparison graph



if __name__ == "__main__":
    main()