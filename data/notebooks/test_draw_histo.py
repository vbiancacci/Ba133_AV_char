# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import json
# import sys
# sys.path.append('/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/')
# from Ba133_data_AV_analysis import * 


# detector = "V05266A"

# t2_folder_lh5_v01 = "/lfs/l1/legend/detector_char/enr/hades/char_data/"+detector+"/tier2/ba_HS4_top_dlt/pygama/v01.00/"
# df_total_lh5_v01 = read_all_dsp_lh5(t2_folder_lh5_v01,cuts=False)
# trapE_data_v01 = df_total_lh5_v01['trapE']
# with open("/lfs/l1/legend/users/aalexander/Ba133_AV_char/data/detectors/"+detector+"/calibration_coef_trapE.json") as json_file:  
#     calibration_coefs = json.load(json_file)
#     m = calibration_coefs['m']
#     m_err = calibration_coefs['m_err']
#     c = calibration_coefs['c']
#     c_err = calibration_coefs['c_err']
# energy_data = (trapE_data_v01-c)/m


# #compare with tier 3:


# plt.figure()
# plt.hist(trapE_data_v00,bins=1000,label="v00")
# plt.hist(trapE_data_v01,bins=1000,label="v01")
# plt.legend()
# plt.show()
