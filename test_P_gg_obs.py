#from P_gm_obs import *
#from P_gm_theory import *
from P_gg_obs import *
#from P_gg_theory import *
import numpy as np
import pylab
from scipy.interpolate import interp1d
import pyccl as ccl

ell_min_range = np.round(2*10**(2+np.arange(5)/5.))
ell_max_range = np.append(ell_min_range[1:]-1  ,2001.)
ell_mid = np.sqrt(ell_min_range*ell_max_range)
ell = np.arange(200,2001)
print ell_min_range
print ell_max_range
print ell_max_range - ell_min_range +1
#print ell




p = ccl.Parameters(Omega_c = 0.25, Omega_b = 0.05, h = 0.7, sigma8 = 0.8 ,n_s = 0.96)
cosmo = ccl.Cosmology(p)

p_z_l = np.zeros(500)
for field_num in range(1,6):
    data = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/data/p_of_z_lens/F%s_p_of_z_z1z2z3.txt"%field_num)
    z_l = data[:,0]
    p_z_l += data[:,1]
    b_l = np.ones(500)

p_z_l = p_z_l/np.sum(p_z_l)/0.01
pylab.plot(z_l[0:101],p_z_l[0:101])
pylab.show()

lens = ccl.ClTracerNumberCounts(cosmo, has_rsd = False, has_magnification= False, n = (z_l,p_z_l), bias = (z_l,b_l) )




nwbin = 20 -2

w_sum = np.zeros(nwbin)
error_w_jackknife_sum = np.zeros(nwbin)
error_w_poisson_sum = np.zeros(nwbin)
n_field = 5.
for field_num in range(1,6):
    
    data = np.loadtxt('/usr/local/athena_1.7/test_z1z2z3/F%s/w_theta_0_0_LS.dat'%field_num)
    theta = data[:,0][:-2]
    w_sum += data[:,1][:-2]
    error_w_jackknife_sum += data[:,2][:-2]**2/n_field**2
    error_w_poisson_sum += data[:,3][:-2]**2/n_field**2

w = w_sum/n_field
error_w_jackknife = np.sqrt(error_w_jackknife_sum/n_field)
error_w_poisson = np.sqrt(error_w_poisson_sum/n_field)
"""
pylab.errorbar(theta, w, yerr= error_w_jackknife, fmt = '.')

pylab.xlabel(r'$\theta $[arcmin]')
pylab.ylabel(r'$w(\theta)$')
pylab.xscale('log')
pylab.yscale('log')
#pylab.xlim(0.5, 200)
#pylab.ylim(-0.05, 0.15)
pylab.ylim(0.001, 1.)
#pylab.hlines(0, 0.5, 200)
pylab.tick_params(bottom= 'on', top = 'on', left = 'on',right = 'on', direction = 'in', which = 'both')
pylab.show()
"""
print theta
theta = theta/60.*np.pi/180. # theta converted to radian
print theta




log_w_interp = interp1d(np.log(theta), np.log(w))
theta_min = min(theta)
theta_max = max(theta)

theta_new = (theta_min*np.exp(np.linspace(0.,np.log(theta_max*0.9/theta_min), num = 1000)))
#print theta_new

log_w_new = log_w_interp(np.log(theta_new))
w_new = np.exp(log_w_new)
#print log_w_new
#print w_new

"""
pylab.errorbar(theta, w, yerr= error_w_jackknife)
pylab.plot(theta_new*1.1, w_new)


pylab.xlabel(r'$\theta $[arcmin]')
pylab.ylabel(r'$w(\theta)$')
pylab.xscale('log')
pylab.yscale('log')

pylab.tick_params(bottom= 'on', top = 'on', left = 'on',right = 'on', direction = 'in', which = 'both')
pylab.show()

"""
b = 1.5
p_obs_band = np.zeros(len(ell_min_range))
#p_theory_band = np.zeros(len(ell_min_range))
for i in range(len(ell_min_range)):
    print i
    p_obs_band[i] = P_gg_band(theta_new, w_new,ell_max_range[i], ell_min_range[i] )
"""
for i in range(len(ell_min_range)):
    print i
    
    delta_l = np.log(ell_max_range[i]/ell_min_range[i])
    #print delta_l
    
    ell_temp_range = np.arange(int(ell_min_range[i]), int(ell_max_range[i]+1),1)
    #print ell_temp_range
    p_theory_band[i] = b**2*simps(ell_temp_range*ccl.angular_cl(cosmo,lens, lens, ell_temp_range),ell_temp_range)/delta_l

"""
p_ell = b**2*ccl.angular_cl(cosmo,lens, lens, ell)

pylab.plot(ell_mid, p_obs_band/(2*np.pi), 'o', label = 'obs')

#pylab.plot(ell_mid, p_theory_band/(2*np.pi), 'o', label = 'theory')
pylab.plot(ell, p_ell*ell**2/(2*np.pi),label = 'theory')
pylab.ylabel(r'$\ell^2 P^{gg}(\ell)/2 \pi$' )
pylab.xlabel(r'$\ell$')
pylab.legend()
pylab.xscale('log')
pylab.yscale('log')
pylab.xlim(200,2000)
pylab.tick_params(bottom= 'on', top = 'on', left = 'on',right = 'on', direction = 'in', which = 'both')

pylab.show()

