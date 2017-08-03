import numpy as np
import pyccl as ccl
import pylab

k = 10**(-5.+np.arange(10))

p = ccl.Parameters(Omega_c = 0.25, Omega_b = 0.05, h = 0.7, sigma8 = 0.8 ,n_s = 0.96)
cosmo = ccl.Cosmology(p)
"""
z = 2.
a = 1./(1.+z)
pk = ccl.nonlin_matter_power(cosmo, a = a, k = k)

print pk
"""
data = np.loadtxt("z.dat")
z_l = data[:,0]
p_z_l = data[:,1]
b_l = np.ones(len(z_l))+z_l

z_s = data[:,2]
p_z_s = data[:,3]


pylab.plot(z_l, p_z_l)

pylab.plot(z_s, p_z_s)
pylab.show()

lens = ccl.ClTracerNumberCounts(cosmo, has_rsd = False, has_magnification= False, n = (z_l,p_z_l), bias = (z_l,b_l) )
source = ccl.ClTracerLensing(cosmo, has_intrinsic_alignment = False, n = (z_s, p_z_s))

ell = np.arange(10,2001)

p_ell_gg = ccl.angular_cl(cosmo,lens, lens, ell)
p_ell_gm = ccl.angular_cl(cosmo,lens, source, ell)

pylab.plot(ell, p_ell_gm, label= 'P_gm')
pylab.plot(ell, p_ell_gg, label= 'P_gg')

pylab.legend()
pylab.xscale('log')
pylab.yscale('log')
pylab.xlim(10,2000)
pylab.ylim(10**(-9),10**(-4))
pylab.tick_params(bottom= 'on', top = 'on', left = 'on',right = 'on', direction = 'in', which = 'both')
pylab.show()
