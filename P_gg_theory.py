import numpy as np
import P_nonlin
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.funcs as cosmofunc



cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)


def P_gg_ell(b, ell,p_z, z ):

    x = cosmofunc.comoving_distance(z,cosmo)
    fk = cosmofunc.angular_diameter_distance(z, cosmo)*(1.+z) # (1+z) factor: converted to comoving scale.


    return b**2*simps(p_z**2/fk**2*P_nonlin.P_nonlin(((ell+1./2.)/fk), z),x)

def P_band(ell, P_ell):
    lmin = ell[0]
    lmax = ell[len(ell)-1]
    delta_l = np.log(lmax/lmin)
    
    return 1./delta_l*simps(ell*P_ell, ell)