import numpy as np
import pandas as pd
import math
import sys 
import h5py
import random
import glob
from datetime import datetime
import json


conf_path = "V02160A.json"
with open(conf_path) as json_file:
    json_geometry = json.load(json_file)
    geometry = json_geometry['geometry'] #cavity radius
    
    R_b = radius_in_mm = geometry["radius_in_mm"] #crystal main/bottom radius
    H = height_in_mm = geometry["height_in_mm"] # = cryystal height 
    well_gap_in_mm = geometry["well"]["gap_in_mm"] #radius cavity
    r_c = well_radius_in_mm = geometry["well"]["radius_in_mm"] #radius cavity
    print(geometry["taper"])
    taper_top_outer_angle_in_deg = geometry["taper"]["top"]["outer"]["angle_in_deg"] 
    H_u = taper_top_outer_height_in_mm = geometry["taper"]["top"]["outer"]["height_in_mm"] #height o

R_u = R_b - H_u*math.tan(taper_top_outer_angle_in_deg*np.pi/180) #radius of top crystal
h_c = H - well_gap_in_mm #cavity height

print(R_u)