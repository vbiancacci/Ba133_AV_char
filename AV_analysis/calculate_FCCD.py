import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("mplstyle.txt")
from datetime import datetime
import json
import argparse

#import fitting functions
import sys
sys.path.append('../data/')
from Ba133_data_AV_analysis import * 

#

def main(): 

    #print date and time for log: 
    t0 = datetime.now()
    dt_string = t0.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("")
    print("date and time =", dt_string)	
    print("")

    parser = argparse.ArgumentParser(description='Infer FCCD for a particular detector, with cuts or not')
    #parser.add_argument('--simID', action="store_true", default="IC160A_ba_top_81mmNEW8_01_newresolution")
    parser.add_argument('--simID', action="store_true", default="IC160A-BA133-uncollimated-top-run0003-81z-newgeometry_g")
    parser.add_argument('--detector', action="store_true", default="I02160A")
    parser.add_argument('--cuts', action="store", type=bool, default = False)
    args = parser.parse_args()
    MC_file_id, detector, cuts = args.simID, args.detector, args.cuts
    print("MC file ID: ", MC_file_id)
    print("detector: ", detector)
    print("applying cuts: ", str(cuts))

    print("start...")

    #Get O_ba133 for each FCCD
    FCCD_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3.0] #mm
    O_Ba133_list = []
    O_Ba133_err_list = []
    DLF = 1.0 #considering 0 TL

    for FCCD in FCCD_list:

        #print("FCCD: ", FCCD)
        
        if cuts == False:
            with open("detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+'mm_DLF'+str(DLF)+'_dlt_observables.json') as json_file:
                dlt_obs = json.load(json_file)
                O_Ba133 = dlt_obs['O_Ba133']
                O_Ba133_list.append(O_Ba133)
                O_Ba133_err = dlt_obs['O_Ba133_err']
                O_Ba133_err_list.append(O_Ba133_err)
        else:
            with open("detectors/"+detector+"/"+MC_file_id+"_FCCD"+str(FCCD)+'mm_DLF'+str(DLF)+'_dlt_observables_cuts.json') as json_file:
                dlt_obs = json.load(json_file)
                O_Ba133 = dlt_obs['O_Ba133']
                O_Ba133_list.append(O_Ba133)
                O_Ba133_err = dlt_obs['O_Ba133_err']
                O_Ba133_err_list.append(O_Ba133_err)


    #Get ratio for data
    if cuts == False:
        with open("../data/detectors/"+detector+"/dlt_observables.json") as json_file:
            dlt_obs = json.load(json_file)
            O_Ba133_data = dlt_obs['O_Ba133']
            O_Ba133_err_data = dlt_obs['O_Ba133_err']
    else:
        with open("../data/detectors/"+detector+"/dlt_observables_cuts.json") as json_file:   
            dlt_obs = json.load(json_file)
            O_Ba133_data = dlt_obs['O_Ba133']
            O_Ba133_err_data = dlt_obs['O_Ba133_err']  


    #plot and fit exp decay
    xdata, ydata = FCCD_list, O_Ba133_list
    yerr = O_Ba133_err_list
    aguess = max(ydata)
    bguess = 1
    cguess = min(ydata)
    p_guess = [aguess,bguess,cguess]
    #bounds=([0, 0, 0, 0, -np.inf], [np.inf]*5)
    popt, pcov = optimize.curve_fit(exponential_decay, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf") #, bounds = bounds)
    print(popt)
    a,b,c = popt[0],popt[1],popt[2]
    a_err, b_err, c_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])

    fig, ax = plt.subplots()
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "simulations", elinewidth = 1, fmt='x', ms = 3.0, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    yfit = exponential_decay(xfit,*popt)
    plt.plot(xfit, yfit, "g", label = "fit: a*exp(-bx)+c") 

    #calculate FCCD of data - invert eq
    FCCD_data = (1/b)*np.log(a/(O_Ba133_data-c))
    FCCD_data_err = np.sqrt((1/(a**2*b**4*(c-O_Ba133_data)**2))*(a**2*(b_err**2*(c-O_Ba133_data)**2)*(np.log(-a/(c-O_Ba133_data))**2) + b**2*(c_err**2+O_Ba133_err_data**2) + a_err**2*b**2*(c-O_Ba133_data)**2)) #wolfram alpha
    print('FCCD of data extrapolated: '+str(FCCD_data) +" +/- "+ str(FCCD_data_err))
    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, exponential_decay, popt)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3g \pm %.3g$' % (b, np.sqrt(pcov[1][1])), r'$c=%.3g \pm %.3g$' % (c, np.sqrt(pcov[2][2])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'FCCD of data (extrapolated)=$%.3g \pm %.3g$ mm' % (FCCD_data, FCCD_data_err)))
    plt.text(0.02, 0.275, info_str, transform=ax.transAxes, fontsize=9,verticalalignment='top', bbox=props) #ax.text..ax.tra

    #plot data line
    plt.plot(xfit, [O_Ba133_data]*(len(xfit)), label = 'data') 
    plt.plot(xfit, [O_Ba133_data+O_Ba133_err_data]*(len(xfit)), label = 'data bounds', color = 'grey', linestyle = 'dashed', linewidth = '1.0') 
    plt.plot(xfit, [O_Ba133_data-O_Ba133_err_data]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0') 


    plt.ylabel(r'$O_{Ba133} = (C_{79.6} + C_{81})/C_{356}$')
    plt.xlabel("FCCD (mm)")
    plt.xlim(0,FCCD_list[-1])
    #plt.title("")
    plt.legend(loc="upper right", fontsize=8)
    
    if cuts == False:
        plt.savefig("detectors/"+detector+"/plots/"+MC_file_id+"_FCCD_OBa133.png")
    else:
        plt.savefig("detectors/"+detector+"/plots/"+MC_file_id+"_FCCD_OBa133_cuts.png")

    plt.show()

    #Save interpolated fccd for data to a json file
    FCCD_data_dict = {"FCCD": FCCD,"FCCD_err": FCCD}
    if cuts == False:
        with open("detectors/"+detector+"/"+MC_file_id+"_FCCD_data.json", "w") as outfile: 
            json.dump(FCCD_data_dict, outfile)
    else:
        with open("detectors/"+detector+"/"+MC_file_id+"_FCCD_data_cuts.json", "w") as outfile: 
            json.dump(FCCD_data_dict, outfile)

    print("done")

def exponential_decay(x, a, b ,c):
    f = a*np.exp(-b*x) + c
    return f

if __name__ == "__main__":
    main()