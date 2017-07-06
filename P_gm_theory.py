import numpy as np
import P_nonlin
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology.funcs import angular_diameter_distance
from astropy.cosmology.funcs import comoving_distance
from astropy.cosmology.funcs import scale_factor
from astropy import constants as const

cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

# redshift to comoving distance
Ok0 = 0

def fk_x(x):
    
    if Ok0 == 0.:
        return x
    elif Ok0 > 0.:
        return np.sinh((Ok0)**0.5*cosmo.H0*x/(cosmo.H0*(Ok0)**0.5))
    elif Ok0 < 0.:
        return np.sin((-Ok0)**0.5*cosmo.H0*x/(cosmo.H0*(-Ok0)**0.5))

def g(x,p_z):
    
    gi = np.zeros(len(x))
    
    for i in range(1,len(x))
        
        x' = x[i:]
        gi[i] = simps(p_z[i:]*fk_x(x'-x[i])/fk_x(x'),x')
                                          
    return gi

def P_gm_ell(b, ell,p_z, z ):
                                     
    x = comoving_distance(z,cosmo)
    a = scale_factor(z,cosmo)
    fk = angular_diameter_distance(z, cosmo)*(1+z) # (1+z) factor: converted to comoving scale.
    
    factor = 3.*cosmo.H0**2*cosmo.Om0/(2.*const.c.to('km/s')**2)
    print factor
                      
    return b*factor*simps(p_z*g(x,p_z)/a/fk*P_nonlin((ell+1./2.)/fk), x)




