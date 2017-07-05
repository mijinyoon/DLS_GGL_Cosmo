import numpy as np
from scipy.special import jn
from scipy.integrate import simps


def f(x):
    return x*jn(1,x)

def P_gg_band(theta, w, lmax,lmin):
    
    delta_l = np.log(lmax/lmin)

    return 2*np.pi/delta_l*simps(w*(f(lmin*theta)-f(lmax*theta)),np.log(theta))
