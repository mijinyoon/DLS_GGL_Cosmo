import numpy as np
import pyccl as ccl
import scipy.stats as stats
from numpy.linalg import inv
from scipy.integrate import simps
from multiprocessing import Pool

file = open("/Users/mjyoon/Desktop/mjyoon/DLS_Cosmo/mcmc_result.txt", 'w')
ell = np.arange(250,2500)
#ell bin range
ell_min_range = np.round(2.5*10**(2+np.arange(6)/5.))
ell_max_range = ell_min_range[1:]-1
ell_min_range =ell_min_range[:-1]
ell_mid = np.round(np.sqrt(ell_min_range*ell_max_range))
ell = np.arange(100,4001)
print ell_min_range
print ell_max_range
print ell_mid
ell_gm_mid = ell_mid
ell_gg_mid = ell_mid
bias_lim = np.array([0., 2.0, 0.01])
Oc_lim = np.array([0.15,0.3,0.0025])
Ob_lim = np.array([0.,0.1,0.0005])
hubble_lim  = np.array([0.6, 0.8,0.007])
sigma8_lim = np.array([0.7, 0.9,0.008])
ns_lim = np.array([0.9,1.1,0.0096])



p_z_l = np.zeros(500)
b_l = np.ones(500)

for field_num in range(1,6):
    data = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/data/p_of_z_lens/F%s_p_of_z_z1z2z3.txt"%field_num)
    z_l = data[:,0]
    p_z_l += data[:,1]



p_z_s = np.zeros(500)
z_s = z_l

for field_num in range(1,6):
    data = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/data/p_of_z_source/F%s_p_of_z_SL.txt"%field_num)
    p_z_s += data

p_z_l = p_z_l/np.sum(p_z_l)/0.01
p_z_s = p_z_s/np.sum(p_z_s)/0.01

cov = np.loadtxt("covariance_Pgm_Pgg.txt")
cov_inv = inv(cov)

p_gm_obs =  np.loadtxt("Pgm_mean.txt")[:,3]
p_gg_obs = np.loadtxt("Pgg_mean.txt")

"""
def P_band(p_ell, lmax,lmin):
    
    delta_l = np.log(lmax/lmin)

    return simps(p_ell*ell, ell)
"""

def cal_likelihood(inputs):
    b , Omega_c , Omega_b , h , sigma8 ,n_s  = inputs
    print b
    p = ccl.Parameters(Omega_c = Omega_c, Omega_b = Omega_b , h = h, sigma8 = sigma8 ,n_s = n_s)
    cosmo = ccl.Cosmology(p)

    lens = ccl.ClTracerNumberCounts(cosmo, has_rsd = False, has_magnification= False, n = (z_l,p_z_l), bias = (z_l,b_l) )
    source = ccl.ClTracerLensing(cosmo, has_intrinsic_alignment = False, n = (z_s, p_z_s))


    #p_ell_gm_th = b*ccl.angular_cl(cosmo,lens, source, ell)
    #p_ell_gg_th = b**2*ccl.angular_cl(cosmo,lens, lens, ell)

    p_band_gm_th = b*ccl.angular_cl(cosmo,lens, source, ell_mid)*ell_gm_mid**2
    p_band_gg_th = b**2*ccl.angular_cl(cosmo,lens, lens, ell_mid)*ell_gg_mid**2

    Delta_p_gm  = p_gm_obs - p_band_gm_th
    Delta_p_gg = p_gg_obs - p_band_gg_th

    Delta_p = np.hstack((Delta_p_gm , Delta_p_gg ))
    print np.shape(Delta_p)

    Likelihood = np.exp(-1./2.*np.dot(np.dot(Delta_p, cov_inv), Delta_p.T))

    print Likelihood
    return Likelihood






def param_comparison(param_old, param_candid):
    
    lkhood_old = cal_likelihood(param_old)
    lkhood_new = cal_likelihood(param_candid)

    alpha = lkhood_new/lkhood_old

    if alpha >= 1.:
        return True

    else:
        rand = np.random.uniform(0.,1.)
        if rand <= alpha:
            return True
        else:
            return False


def param_gen(input_params) :
    
    bias_center,Oc_center, Ob_center, hubble_center, sigma8_center, ns_center = input_params
    
    bias = stats.truncnorm((bias_lim[0]-bias_center)/bias_lim[2], (bias_lim[1]-bias_center)/bias_lim[2], bias_center, bias_lim[2]).rvs(1)
    Oc = stats.truncnorm((Oc_lim[0]-Oc_center)/Oc_lim[2], (Oc_lim[1]-Oc_center)/Oc_lim[2], Oc_center, Oc_lim[2]).rvs(1)
    Ob = stats.truncnorm((Ob_lim[0] - Ob_center)/Ob_lim[2], (Ob_lim[1] - Ob_center)/Ob_lim[2], Ob_center, Ob_lim[2]).rvs(1)
    hubble = stats.truncnorm(( hubble_lim[0] -  hubble_center)/ hubble_lim[2], ( hubble_lim[1] -  hubble_center)/ hubble_lim[2],  hubble_center,  hubble_lim[2]).rvs(1)
    sigma8 = stats.truncnorm((sigma8_lim[0] - sigma8_center)/sigma8_lim[2], (sigma8_lim[1] - sigma8_center)/sigma8_lim[2], sigma8_center, sigma8_lim[2]).rvs(1)
    ns = stats.truncnorm((ns_lim[0] - ns_center)/ns_lim[2], (ns_lim[1] - ns_center)/ns_lim[2], ns_center, ns_lim[2]).rvs(1)
    
    return np.array([bias, Oc, Ob, hubble, sigma8, ns])

def running_mcmc(N_iter):
    
    param_ref = np.array([1.,  0.25,  0.05, 0.7, 0.8 ,0.96])
    print param_ref

    for i in range(N_iter):
    
        param_test = param_gen(param_ref)
        
        if param_comparison(param_ref, param_test) == True:
            param_ref = param_test
            file.writelines(["%f " %item  for item in param_ref])
            file.writelines("\n")
    
        else:
            file.writelines(["%f " %item  for item in param_ref])
            file.writelines("\n")




running_mcmc(1000)



