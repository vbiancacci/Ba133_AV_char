import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("mplstyle.txt")
from datetime import datetime
import argparse

#import fitting functions
import sys
sys.path.append('/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/')
from Ba133_data_AV_analysis import * 

#code to plot fit and count gamma line peaks for different FCCDs and DLFs
#bkg run code not included since it is insignificant for Ba - but look at old code if needed

def main(): 

    
    #print date and time for log: 
    t0 = datetime.now()
    dt_string = t0.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")


    parser = argparse.ArgumentParser(description='Fit and count MC gamma line for Ba for a particular detector, with cuts or not')
    #parser.add_argument('--simID', action="store_true", default="IC160A_ba_top_81mmNEW8_01_newresolution")
    parser.add_argument('--simID', action="store", type=str, default="IC160A-BA133-uncollimated-top-run0003-81z-newgeometry_g")
    parser.add_argument('--detector', action="store", type=str, default="I02160A")
    parser.add_argument('--cuts', action="store", type=bool, default = False)
    args = parser.parse_args()
    MC_file_id, detector, cuts = args.simID, args.detector, args.cuts
    print("MC file ID: ", MC_file_id)
    print("detector: ", detector)
    print("applying cuts: ", str(cuts))

    print("start...")

    #hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/processed/" #path to processed MC hdf5 files
    #MC_file = hdf5_path+"processed_detector_"+MC_file_id+'.hdf5' #no FCCD
    hdf5_path = "/lfs/l1/legend/users/aalexander/hdf5_output/raw_MC_combined/processed/" #path to processed MC hdf5 files

    binwidth = 0.15 #keV

    #initialise directories to save outputs if not already existing
    if not os.path.exists("detectors/"+detector+"/plots/"+MC_file_id):
        os.makedirs("detectors/"+detector+"/plots/"+MC_file_id)


    #_____________PROCESS AND PLOT FCCDS_____________ 

    print("")
    print("Process each FCCD and DLF...")

    #This configuration for getting best fit FCCD
    FCCD_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3.0] #make this an input argument?
    DLF_list =[1.0] 

    #This configuration for getting best fit TL
    # FCCD_list = [0.71] #=best fit
    # #FCCD_list = [0.67] #=best fit with cuts
    #FCCD_list = [0.73]
    FCCD_list = [0.69] #=best fit I02160A with cuts
    #FCCD_list = [1.06]
    #DLF_list = [0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5]
    DLF_list = [0.0, 0.5, 1.0, -0.25, -0.5, -0.75]

    #comparison graph for different DLFs
    energies_DLF_list = []
    R_DLF_list = []

    for FCCD in FCCD_list:
        for DLF in DLF_list:
            print("")
            print("FCCD: ", FCCD)
            print("DLF: ", DLF)
            print("")

            energies_FCCD, energy_data, R_simdata_356_FCCD = process_FCCDs(FCCD, DLF, MC_file_id, detector, cuts, hdf5_path, binwidth)
            energies_DLF_list.append(energies_FCCD)
            R_DLF_list.append(R_simdata_356_FCCD)
            #plt.hist(energies_FCCD, bins = bins, weights=(1/R_simdata_356_FCCD)*np.ones_like(energies_FCCD), label ='MC FCCD: ,'+str(FCCD)+' DLF: '+str(DLF)+' (scaled)', histtype = 'step', linewidth = '0.1')

    plt.close("all")


    #Plot comparison graph for best fit FCCD and varying DLFs
    print("plotting DLF comparison graph for best fit FCCD")
    fig, ax = plt.subplots()
    bins = np.arange(0, 450, binwidth)
    for index, energies_DLF in enumerate(energies_DLF_list):
        DLF, FCCD = DLF_list[index], FCCD_list[0]
        print("DLF: ", DLF)
        print("FCCD: ", FCCD)
        print("R for DLF i: ", R_DLF_list[index])
        plt.hist(energies_DLF, bins = bins, weights=(1/R_DLF_list[index])*np.ones_like(energies_DLF), label ='MC FCCD: '+str(FCCD)+' DLF: '+str(DLF)+' (scaled)', histtype = 'step', linewidth = '0.35')
    plt.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.25')
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.xlim(0, 450)
    plt.yscale("log")
    plt.legend(loc="lower left")
    if cuts == False:
        plt.savefig("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/plots/"+MC_file_id+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_allDLFs_datacomparison.png")
    else:
        plt.savefig("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/plots/"+MC_file_id+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_allDLFs_datacomparison_cuts.png")
    plt.show()

    print("done")
    print("time elapsed: ")
    print(datetime.now() - t0)


def process_FCCDs(FCCD, DLF, MC_file_id, detector, cuts, hdf5_path, binwidth):
    "process and plot for different FCCDs"

    #_______plot full spectrum___________
    print("plotting whole simulated spectrum...")

    MC_file = hdf5_path+"processed_detector_"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'.hdf5'    

    df =  pd.read_hdf(MC_file, key="procdf")
    energies = df['energy']
    #energies = energies*1000 - not needed anymore
    no_events = energies.size #=sum(counts)
    print("No. events: ", no_events) 
    bins = np.arange(0,450,binwidth)

    #________Fit peaks of interest_______
    print("")
    print("Fitting peaks of interest...")

    print("356keV:")
    xmin_356, xmax_356 = 352, 359.5 #350 #362
    if DLF == 1.0 and MC_file_id=="IC160A-BA133-uncollimated-top-run0003-81z-newgeometry-singlefile_g":
        xmin_356, xmax_356 = 352, 358 #350 #362
    plt.figure()
    counts, bins, bars = plt.hist(energies, bins = bins, histtype = 'step') #, linewidth = '0.35')
    popt, pcov, xfit = fit_peak_356_2("Energy (keV)", bins, counts, xmin_356, xmax_356)
    a,b,c,d,e = popt[0],popt[1],popt[2],popt[3],popt[4]
    amplitude356_sim = gaussian_and_bkg_2(b, a, b, c, d, e)
    plt.xlim(xmin_356, xmax_356) 
    #plt.ylim(10, 10**7)
    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.savefig("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/plots/"+MC_file_id+"/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'_356keV.png')

    C_356, C_356_err = gauss_count(a, b, c, np.sqrt(pcov[0][0]),  np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
    print("gauss count 356keV: ", C_356, " +/- ", C_356_err )


    print("81 keV:")
    xmin_81, xmax_81 = 77, 84
    plt.figure()
    counts, bins, bars = plt.hist(energies, bins = bins, histtype = 'step') #, linewidth = '0.35')
    popt, pcov, xfit = fit_double_peak_81("Energy (keV)", bins, counts, xmin_81, xmax_81)
    a,b,c,d,e,f,g,h = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7] 
    plt.xlim(xmin_81, xmax_81) 
    #plt.ylim(5*10**2, 5*10**6)
    #plt.ylim(10**3, 10**7) #gammas_81mmNEW
    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.savefig("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/plots/"+MC_file_id+"/"+MC_file_id+'_FCCD'+str(FCCD)+"mm_DLF"+str(DLF)+'_81keV.png')

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
    peak_ranges = [[159.5,161.5],[221.5,225],[274,279],[300,306],[381,384.5]] #Rough by eye
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

        # if peaks[index]==383: #problems converging for this particular histogram
        #     C, C_err = float("nan"), float("nan")
        #     print("gauss count: ", C, " +/- ", C_err )
        #     other_peak_counts.append(C)
        #     other_peak_counts_err.append(C_err)
        #     continue

        
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
        plt.savefig("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/plots/"+MC_file_id+"/"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+"_"+str(peaks[index])+'keV.png')
        C, C_err = gauss_count(a,b, c, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2]), binwidth)
        print("gauss count: ", C, " +/- ", C_err )
        other_peak_counts.append(C)
        other_peak_counts_err.append(C_err)


    plt.close('all')

    #__________compare against real data__________
    print("")
    print("plotting simulation against actual data...")

    #get calibration coefs
    with open("/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/detectors/"+detector+"/calibration_coef.json") as json_file:
    #with open("/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/detectors/"+detector+"/calibration_coef_trapE.json") as json_file:  
        calibration_coefs = json.load(json_file)
        m = calibration_coefs['m']
        m_err = calibration_coefs['m_err']
        c = calibration_coefs['c']
        c_err = calibration_coefs['c_err']

    
    t2_folder_h5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v00.00/"
    t2_folder_lh5 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v01.00/"

    if cuts == False:
    
        # e_ftp, .h5 files - dont exist for V05266A
        df_total_h5 = read_all_dsp_h5(t2_folder_h5, cuts)
        e_ftp_data = df_total_h5['e_ftp']
        energy_data = (e_ftp_data-c)/m

        #trapE, .lh5 files
        # df_total_lh5 = read_all_dsp_lh5(t2_folder_lh5,cuts)
        # trapE_data = df_total_lh5['trapE']
        # energy_data = (trapE_data-c)/m

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
        with open("/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/detectors/"+detector+"/dlt_observables.json") as json_file:
            dlt_observables = json.load(json_file)
            C_356_data = dlt_observables['C_356_average']

        C_356_sim = C_356
        print("integral counts 356keV data: ", C_356_data)
        print("integral counts 356keV simulation: ", C_356_sim)
        R_simdata_356_counts = C_356_sim/C_356_data #ratio of sim to data for 356 peak counts
        print(R_simdata_356_counts)

        fig, ax = plt.subplots()
        counts_data, bins, bars_data = plt.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')
        counts, bins, bars = plt.hist(energies, bins = bins, weights=(1/R_simdata_356_counts)*np.ones_like(energies), label = "MC: FCCD "+str(FCCD)+"mm, DLF: "+str(DLF)+" (scaled)", histtype = 'step', linewidth = '0.35')
        print("counts scaled")
        print(counts)
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.xlim(0, 450)
        plt.yscale("log")
        plt.legend(loc = "lower left")
        plt.savefig("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/plots/"+MC_file_id+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_datacomparison.png")

    else:
        passed_cuts = json.load(open('/lfs/l1/legend/users/aalexander/large_files/cuts/'+detector+'_ba_top_passed_cuts_data.json','r')) #passed cuts
        
        #e_ftp, .h5 files - dont exist for V05266A
        df_total_cuts_h5 = read_all_dsp_h5(t2_folder_h5,cuts, passed_cuts = passed_cuts)
        e_ftp_data_cuts = df_total_cuts_h5['e_ftp']
        energy_data_cuts= (e_ftp_data_cuts-c)/m
        
        #trapE, .lh5 files
        # df_total_cuts_lh5 = read_all_dsp_lh5(t2_folder_lh5, cuts, passed_cuts=passed_cuts)
        # trapE_data_cuts = df_total_cuts_lh5['trapE']
        #energy_data_cuts= (trapE_data_cuts-c)/m

        #scale up data to same amplitude 356 peak as simulation
        with open("/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/detectors/"+detector+"/dlt_observables_cuts.json") as json_file:
            dlt_observables = json.load(json_file)
            C_356_data = dlt_observables['C_356_average']

        C_356_sim = C_356
        print("integral counts 356keV data: ", C_356_data)
        print("integral counts 356keV simulation: ", C_356_sim)
        R_simdata_356_counts = C_356_sim/C_356_data #ratio of sim to data for 356 peak counts
        print(R_simdata_356_counts)

        fig, ax = plt.subplots()
        counts_data_cuts, bins, bars_data = plt.hist(energy_data_cuts, bins=bins,  label = "Data (cuts)", histtype = 'step', linewidth = '0.35')
        counts, bins, bars = plt.hist(energies, bins = bins, weights=(1/R_simdata_356_counts)*np.ones_like(energies), label = "MC: FCCD "+str(FCCD)+"mm, DLF: "+str(DLF)+" (scaled)", histtype = 'step', linewidth = '0.35')
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.xlim(0, 450)
        plt.yscale("log")
        plt.legend(loc = "lower left")
        plt.savefig("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/plots/"+MC_file_id+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_datacomparison_cuts.png")

    
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
        with open("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_dlt_observables.json", "w") as outfile: 
            json.dump(dlt_observables, outfile)
    else:
        with open("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_dlt_observables_cuts.json", "w") as outfile: 
            json.dump(dlt_observables, outfile)


    #calculate DATA/MC for each energy bin and export
    #only do this for best fit FCCD
    #if FCCD == 0.744 or FCCD == 0.698:
    #if FCCD == 0.73 or FCCD == 0.69:   #i02160a
    if FCCD == 1.06: #v05266a 
        print("")
        print("calculating data/MC ratios for best fit FCCD")

        Data_MC_ratios = []
        Data_MC_ratios_err = []

        for index, bin in enumerate(bins[1:]):

            if cuts == False:
                data = counts_data[index]
            else:
                data = counts_data_cuts[index]

            MC = counts[index] #This counts has already been scaled by weights

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
            Data_MC_ratios_df.to_csv("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_DataMCRatios.csv", index=False)
        else:
            Data_MC_ratios_df.to_csv("/lfs/l1/legend/users/aalexander/Ba133_AV_char/AV_analysis/detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_DataMCRatios_cuts.csv", index=False)

    if cuts == False:
        return energies, energy_data,  R_simdata_356_counts #for creating comparison graph
    else: 
        return energies, energy_data_cuts,  R_simdata_356_counts #for creating comparison graph

if __name__ == "__main__":
    main()