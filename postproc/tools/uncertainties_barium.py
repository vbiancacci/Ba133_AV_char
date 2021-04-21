import numpy as np

def uncertainty(counts_79_81keV, counts_356keV):
#values from Bjoern's thesis - Barium source
    gamma_line=0.69
    geant4=2.
    source_thickness=0.02
    source_material=0.01
    endcap_thickness=0.28
    detector_cup_thickness=0.07
    detector_cup_material=0.03

#compute statistical error
    se=StatisticalError()
    MC_statistics=np.sqrt(se)*100

#sum squared of all the contributions
    tot_error=np.sqrt(
        gamma_line**2+
        geant4**2+
        source_thickness**2+
        source_material**2+
        endcap_thickness**2+
        detector_cup_thickness**2+
        detector_cup_material**2+
        MC_statistics**2
        )

    return tot_error


def StatisticalError():
#error on a peak is sqrt(N) where N is the # counts of the peak    ??
#counts_79_81keV is the sum of the # counts of the peaks at 79 keV and 81 keV
#counts_356keV is the # counts of the peak at 356keV

    sigma_79_81keV=np.sqrt(counts_79_81keV)
    sigma_356keV=np.sqrt(counts_356keV)
    ratio=sigma_79_81keV/sigma_356keV
    
   se=((sigma_79_81keV/counts_356keV)**2+
      ((counts_79_81keV*sigma_356keV)/(counts_356keV**2))**2)
    
    return se
