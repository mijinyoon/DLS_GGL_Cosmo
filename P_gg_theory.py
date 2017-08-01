import numpy as np
import P_nonlin
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM



cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

def dz_dx_func(z):
    h = 10**(-6)
    return     2.*h/(cosmo.comoving_distance(z+h).value - cosmo.comoving_distance(z-h).value)

def P_gg_ell(ell, b, p_z, z ):
    print ell
    x = cosmo.comoving_distance(z).value
    fk = (cosmo.angular_diameter_distance(z)*(1.+z)).value # (1+z) factor: converted to comoving scale.
    dz_dx = dz_dx_func(z)
    p_x = p_z*dz_dx
    
    PS = np.zeros(len(z))
    for i in range(len(z)):
        #print 'z:',z[i]
        PS[i] = P_nonlin.P_nonlin(((ell+1./2.)/fk[i]), z[i])
    
    return ell, b**2*simps(p_x**2/fk**2*PS,x)

def P_band(ell, P_ell):
    lmin = ell[0]
    lmax = ell[len(ell)-1]
    delta_l = np.log(lmax/lmin)
    
    return 1./delta_l*simps(ell*P_ell, ell)
