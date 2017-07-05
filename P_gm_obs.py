import numpy as np
from scipy.special import jn
from scipy.integrate import simps

def h(x):
    return -x*jn(1,x) - 2.*jn(0,x)

def P_gm_band(theta, shear, lmax,lmin):
    
    delta_l = np.log(lmax/lmin)
    
    return 2*np.pi/delta_l*simps(shear*(h(lmin*theta)-h(lmax*theta)),np.log(theta))
