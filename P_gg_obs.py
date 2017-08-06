import numpy as np
from scipy.special import jn
from scipy.integrate import simps
import pylab

def f(x):
    return x*jn(1,x)

def P_gg_band(theta, w, lmax,lmin):
    """
    pylab.plot(theta, f(lmin*theta), label = 'lmin')
    pylab.xscale('log')
    #pylab.yscale('log')
    pylab.legend()
    pylab.show()
    
    pylab.plot(theta, f(lmax*theta), label = 'lmax')
    pylab.xscale('log')
    #pylab.yscale('log')
    pylab.legend()
    pylab.show()
    
    pylab.plot(theta, f(lmax*theta)-f(lmin*theta), label ='diff')
    pylab.xscale('log')
    #pylab.yscale('log')
    pylab.legend()
    pylab.show()
    """
    delta_l = np.log(lmax/lmin)
    #print delta_l
    return 2.*np.pi/delta_l*simps(w*(f(lmax*theta)-f(lmin*theta)),np.log(theta))
