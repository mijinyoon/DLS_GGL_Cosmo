import numpy as np
import P_nonlin
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.funcs as cosmofunc
from astropy import constants as const
from scipy.interpolate import interp1d
#import pylab

cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

Ok0 = 0.

def fk_x(x):
    
    if Ok0 == 0.:
        return x
    elif Ok0 > 0.:
        return np.sinh((Ok0)**0.5*cosmo.H0*x/(cosmo.H0*(Ok0)**0.5))
    elif Ok0 < 0.:
        return np.sin((-Ok0)**0.5*cosmo.H0*x/(cosmo.H0*(-Ok0)**0.5))

def dz_dx_func(z):
    h = 10**(-6)
    return     2.*h/(cosmo.comoving_distance(z+h).value - cosmo.comoving_distance(z-h).value)

def g(z,p_z):
    
    x= cosmo.comoving_distance(z).value
    
    dz_dx = dz_dx_func(z)
    
    p_x = p_z*dz_dx
    
    gi = np.zeros(len(x))
    
    for i in range(1,len(x)):
        
        x_prime = x[i:]
        gi[i] = simps(p_x[i:]*fk_x(x_prime-x[i])/fk_x(x_prime),x_prime)

    return x, gi

def g_func(x_input,x_min, x_max, g_interp):
        
    if x_input < x_min :
        return g_interp(x_min)
    elif x_input > x_max:
        return g_interp(x_max)
    else:
        return g_interp(x_input)

vg_func = np.vectorize(g_func)


def P_gm_ell(ell,b, z_lens,p_z_lens,z_source, p_z_source):
    print ell
    
    x_lens = (cosmo.comoving_distance(z_lens)).value
    dz_dx = dz_dx_func(z_lens)

    A_factor = cosmo.scale_factor(z_lens)
    p_x_lens = p_z_lens*dz_dx
    
    fk = (cosmo.angular_diameter_distance(z_lens).value*(1.+z_lens)) # (1+z) factor: converted to comoving scale.
    factor = 3.*cosmo.H0**2*cosmo.Om0/(2.*const.c.to('km/s')**2)
    
    x_source, g_source = g(z_source,p_z_source)
    #pylab.plot(z_source, g_source)
    #pylab.show()
    g_interp = interp1d(x_source, g_source, kind = 'cubic')
    g_at_lens_distance = vg_func(x_lens ,min(x_source), max(x_source), g_interp)

    index = np.where(g_source > 10**(-5))[0]
    z_source = z_source[index]
    x_source = x_source[index]
    g_source = g_source[index]
    #print x_source
    
    #pylab.plot(z_source, g_source)
    #pylab.show()

    g_interp = interp1d(x_source, g_source, kind = 'cubic')
    
    g_at_lens_distance = vg_func(x_lens ,min(x_source), max(x_source), g_interp)
    
    #pylab.plot(z_lens, x_lens)
    #pylab.show()

    PS = np.zeros(len(z_lens))
    
    for i in range(len(z_lens)):
        #print 'z:',z[i]
        PS[i] = P_nonlin.P_nonlin(((ell+1./2.)/fk[i]), z_lens[i])
        #print factor


    return ell, (b*factor*simps(p_x_lens*g_at_lens_distance/A_factor/fk*PS,x_lens)).value



