import numpy as np
import P_nonlin
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.funcs as cosmofunc
from astropy import constants as const


cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

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
    
    for i in range(1,len(x)):
        
        x_prime = x[i:]
        gi[i] = simps(p_z[i:]*fk_x(x_prime-x[i])/fk_x(x_prime),x_prime)
                                          
    return gi

def P_gm_ell(ell,b, z_lens,p_z_lens,z_source, p_z_source):
    #print ell
    
    x_source = cosmo.comoving_distance(z_source)
    x_lens = cosmo.comoving_distance(z_lens)
    A_factor = cosmo.scale_factor(z_lens)
    print x_source
    print x_lens
    print A_factor
    fk = cosmo.angular_diameter_distance(z_lens)*(1.+z_lens) # (1+z) factor: converted to comoving scale.
    factor = 3.*cosmo.H0**2*cosmo.Om0/(2.*const.c.to('km/s')**2)
    
    print fk
    print factor
    
    PS = np.zeros(len(z_lens))
    
    for i in range(len(z_lens)):
        #print 'z:',z[i]
        PS[i] = P_nonlin.P_nonlin(((ell+1./2.)/fk[i]), z_lens[i])
        #print factor

    PS2 = b*factor*simps(p_z_lens*g(x_source,p_z_source)/A_factor/fk*PS,x_lens)
    #print PS
    #print PS2

    return ell, PS2.value



