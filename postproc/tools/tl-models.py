import numpy as np

class NoValidTLModel(Exception):
    pass

def models (modelname,fDLT,fFCCD):
    conditions=(x<fFCCD & x>0)+(x>=fFCCD)  #?
    #no transtion layer
    if modelname=="notl":
        CCEs = np.where(x <= fFCCD, 0, CCEs)
    #linear model
    elif modelname=="l":
        CCEs = np.where(x <= fDLT, 0, CCEs)
        CCEs = np.where((x>fDLT)&(x<fFCCD), 1./(fFCCD-fDLT)*x-fDLT/(fFCCD-fDLT), CCEs)
    #error function model
    elif modelname=="erf":
        M=3
        CCEs=np.where(conditions, 0.5+0.5*erf((x-(1+fDLT)/2*fFCCD)/((1-fDLT)*fFCCD/M*sqrt(2)), CCEs)
    #cosh model
    elif modelname=="cosh":
        CCES=np.where(conditions, pow((cosh(x/fFCCD)-1)/(cosh(1)-1),0.55/pow(1-fDLT,1.3)), CCEs )
    #polynomial model
    elif modelname=="pol":
        CCES=np.where(contidions,pow(x/fFCCD,((log(0.2*fDLT)-log(3-fDLT))/log(fDLT)))*(3-2*x/fFCCD), CCEs)
    #error- no existing model
    else:
        raise NoValidTLModel('Choose:\n notl (no trantion layer) \n l (linear function) \n erf (error function) \n cosh (cosh function \n pol (plynomial function)')

    return CCEs
