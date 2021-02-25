import sys
import uproot
import math
import numpy as np
import sympy
from pprint import pprint
import yaml

#   IC160A


if(len(sys.argv) != 3):
    print('Usage: python3 IC160A_m_q.py constants_IC160A.conf  [A value]')
    sys.exit()

conf_path = sys.argv[1]
A = float(sys.argv[2])    #where the fccd (or transition layer) ends
#output= str(sys.argv[4])


with open(conf_path) as fid:
    config = yaml.load(fid, Loader=yaml.Loader)
pprint(config)

R_b     = config['R_b']
R_u     = config['R_u']
H_u     = config['H_u']
offset  = config['offset']

'''

     B______________     ________
     / ___________ |     | _______
    / / E        | |     | |
   / /           | |     | |
C / /            | |     | |
  | | F          | |     | |
  | |            . .     . .
  | |            . .     . .
  | |
. .
. .


 '''

#B(R_u,offset)
#C(R_b,offset+H_u)
#E(r_E?,offset+A)
#F(R_b-A,z_F?)



#line passing trough C and B
m=H_u/(R_b-R_u)
#q calculating in B
q=(offset-m*R_u)


#translation of the line by vector v 
theta=math.pi/2-np.arctan(m)   #orthogonal to m direction
y_v=A*math.sin(theta)
x_v=A*math.cos(theta)

#x=x'+x_v
#y=y'-y_v
#y'=mx'+(q+y_v+m*x_v)
q_A=q+y_v+m*x_v

#r_E:
r_E=((offset+A)-q_A)/m

#z_F:
z_F=m*(R_b-A)+q_A

with open('m_q.conf', 'w') as file_output:
    file_output.write("m   : {}  #m functions \n".format(m))
    file_output.write("q   : {}  #q external function \n".format(q))
    file_output.write("q_A   : {}  #q_A internal function \n".format(q_A))
    file_output.write("r_E     : {}    #extremity r point  \n".format(r_E))
    file_output.write("z_F     : {}    #extremity z point \n".format(z_F))

