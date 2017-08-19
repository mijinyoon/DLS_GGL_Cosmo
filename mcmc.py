import numpy as np
import pyccl as ccl
import scipy.stats as stats
from numpy.linalg import inv
from scipy.integrate import simps
from multiprocessing import Pool
from datetime import datetime

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


ell_min_range = np.round(2*10**(2+np.arange(5)/5.))
ell_max_range = np.append(ell_min_range[1:]-1  ,2001.)
ell_mid = np.sqrt(ell_min_range*ell_max_range)


ell_gg_mid = ell_mid
bias_lim = np.array([0.1, 4.0, 0.01])
Oc_lim = np.array([0.1,0.9,0.0025])
#Om_lim = np.array([0.1,0.9,0.0025])

Ob_lim = np.array([0.02,0.03,0.0005])
hubble_lim  = np.array([0.55, 0.9,0.007])
sigma8_lim = np.array([0.6, 1.0,0.008])
ns_lim = np.array([0.7,1.3,0.001])

p_z_l = np.zeros(250)
b_l = np.ones(250)

for field_num in range(1,6):
    data = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/data/p_of_z_lens/F%s_p_of_z_z1z2z3.txt"%field_num)
    z_l = data[:,0][0:250]
    p_z_l += data[:,1][0:250]
    del data



p_z_s = np.zeros(250)
z_s = z_l

for field_num in range(1,6):
    data = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/data/p_of_z_source/F%s_p_of_z_SL.txt"%field_num)
    p_z_s += data[0:250]
    del data


p_z_l = p_z_l/np.sum(p_z_l)/0.01
p_z_s = p_z_s/np.sum(p_z_s)/0.01

cov = np.loadtxt("covariance_Pgm_Pgg_new.txt")
cov_inv = inv(cov)

p_gm_obs =  np.loadtxt("Pgm_mean.txt")[:,3]
p_gg_obs =  np.array([0.05200286,  0.0738437,   0.09889041,  0.14413876,  0.16433688])

"""
def P_band(p_ell, lmax,lmin):
    
    delta_l = np.log(lmax/lmin)

    return simps(p_ell*ell, ell)
"""

def cal_likelihood(inputs):
    
    b , Omega_c , Omega_b , h , sigma8 ,n_s  = inputs
    #print "set up input0"
    p = ccl.Parameters(Omega_c = Omega_c, Omega_b = Omega_b , h = h, sigma8 = sigma8 ,n_s = n_s)
    cosmo = ccl.Cosmology(p)
    #print "set up input1"
    lens = ccl.ClTracerNumberCounts(cosmo, has_rsd = False, has_magnification= False, n = (z_l,p_z_l), bias = (z_l,b_l) )
    source = ccl.ClTracerLensing(cosmo, has_intrinsic_alignment = False, n = (z_s, p_z_s))
    #print "set up input2"
    #p_ell_gm_th = b*ccl.angular_cl(cosmo,lens, source, ell)
    #p_ell_gg_th = b**2*ccl.angular_cl(cosmo,lens, lens, ell)
    p_band_gm_th = ccl.angular_cl(cosmo,lens, source, ell_gm_mid)
    #print "calculated pgm theory"
    p_band_gg_th = ccl.angular_cl(cosmo,lens, lens, ell_gg_mid)
    #print "calculated pgg theory"
    Delta_p_gm  = p_gm_obs - p_band_gm_th*b*ell_gm_mid**2
    Delta_p_gg = p_gg_obs - p_band_gg_th*b**2*ell_gg_mid**2

    Delta_p = np.hstack((Delta_p_gm , Delta_p_gg ))
    
    logLikelihood = - np.dot(np.dot(Delta_p, cov_inv), Delta_p.T))/2.
    
    del p, cosmo, lens, source, p_band_gm_th, p_band_gg_th, Delta_p, Delta_p_gg, Delta_p_gm
    return logLikelihood






def param_comparison(loglkhood_old, param_candid):
    print "test start"
    #lkhood_old = cal_likelihood(param_old)
    loglkhood_new = cal_likelihood(param_candid)
    print "calculated likelihood"
    #print 'lkhood_new', lkhood_new
    alpha = np.exp(loglkhood_new -loglkhood_old)
    
    #print 'alpha', alpha

    if alpha >= 1.:
        return True, loglkhood_new

    else:
        rand = np.random.uniform(0.,1.)
        #print 'rand', rand
        
        if rand <= alpha:
            return True, loglkhood_new
        else:
            return False, loglkhood_old


def param_gen(input_params, core_num) :
    
    bias_center,Oc_center, Ob_center, hubble_center, sigma8_center, ns_center = input_params
    
    np.random.seed(seed = datetime.now().microsecond*core_num)
    
    bias = stats.truncnorm((bias_lim[0]-bias_center)/bias_lim[2], (bias_lim[1]-bias_center)/bias_lim[2], bias_center, bias_lim[2]).rvs(1)[0]
    Oc = stats.truncnorm((Oc_lim[0]-Oc_center)/Oc_lim[2], (Oc_lim[1]-Oc_center)/Oc_lim[2], Oc_center, Oc_lim[2]).rvs(1)[0]
    Ob = stats.truncnorm((Ob_lim[0]-Ob_center)/Ob_lim[2], (Ob_lim[1] - Ob_center)/Ob_lim[2], Ob_center, Ob_lim[2]).rvs(1)[0]
    hubble = stats.truncnorm(( hubble_lim[0] -  hubble_center)/ hubble_lim[2], ( hubble_lim[1] -  hubble_center)/ hubble_lim[2],  hubble_center,  hubble_lim[2]).rvs(1)[0]
    sigma8 = stats.truncnorm((sigma8_lim[0] - sigma8_center)/sigma8_lim[2], (sigma8_lim[1] - sigma8_center)/sigma8_lim[2], sigma8_center, sigma8_lim[2]).rvs(1)[0]
    ns = stats.truncnorm((ns_lim[0] - ns_center)/ns_lim[2], (ns_lim[1] - ns_center)/ns_lim[2], ns_center, ns_lim[2]).rvs(1)[0]
    
    A = np.array([bias, Oc, Ob, hubble, sigma8, ns])
    del bias, Oc, Ob, hubble, sigma8, ns
    return A

def running_mcmc(core_num,N_iter=10000):
    
    param_ref = np.array([1.,  0.2,  0.025, 0.7, 0.8 ,0.96])
    store = []
    loglikelihood_ref = cal_likelihood(param_ref)
    #print param_ref
    #print likelihood_ref
    
    for i in range(N_iter):

        print i
        param_test = param_gen(param_ref,core_num)
        #print "param_generated"
        #print param_test
        
        test_result, loglikelihood_ref = param_comparison(loglikelihood_ref, param_test)
        #print "param_tested"
        
        if test_result == True:
            
            print 'accept'
            param_ref = param_test
            store += [param_ref]
        #print "param stored1"

    
        else:
            print 'reject'
            store += [param_ref]
#print "param stored2"
        #print store
    file_name = "mcmc%s_%s"%(N_iter,core_num)
    np.savetxt("/Users/mjyoon/Desktop/mjyoon/DLS_Cosmo/%s_new2.txt"%file_name,np.array(store), fmt = '%f')

pool = Pool(10)
pool.map(running_mcmc, np.arange(1,11))
