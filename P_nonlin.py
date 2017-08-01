import numpy as np
import P_lin
import pylab
from scipy.integrate import simps
"""
h = 0.7

#data1 = np.loadtxt("P_nonlin.txt")
data1 = np.loadtxt("ps_takahashi.txt")

k_data = data1[:,0]*h
k_test = k_data

P_data_1 = data1[:,2]
P_data_1 = P_data_1/(k_data**3/(2*np.pi**2))

z = 1.
"""
data = np.loadtxt("z_ksigma_neff_C_james_lens.txt")
z_array = data[:,0]
k_sigma_array = data[:,1]
n_eff_array = data[:,2]
C_array = data[:,3]

def sigma_sq(R,z):
    
    k = 10**(-10.+ np.arange(200)/10.)
    Delta_L_sq = P_lin.P(k, z)*k**3/(2*np.pi**2)
    
    return simps(Delta_L_sq*np.exp(-k**2*R**2), np.log(k))

def vsigma_sq(R,z):
    
    vec_sigma_sq = np.zeros(len(R))
    
    for i in range(len(R)):
    
        vec_sigma_sq[i] = sigma_sq(R[i],z)
    
    return vec_sigma_sq

def d_ln_sigma_sq(x,z):
    step = 1e-10
    return (np.log(sigma_sq(x+step,z))-np.log(sigma_sq(x-step,z)))/(np.log(x+step) - np.log(x-step))

def d2_ln_sigma_sq(x,z):

    step = 1e-6

    t1 = (np.log(sigma_sq(x+2*step,z))-np.log(sigma_sq(x,z)))/(np.log(x+2*step)- np.log(x))
    t2 = (np.log(sigma_sq(x,z))   -np.log(sigma_sq(x-2*step,z)))/(np.log(x)- np.log(x-2*step))
    t3 = np.log(x+step) - np.log(x-step)
    return (t1 - t2)/ t3

"""
#k_test = 10**(-5.+ np.arange(400)/50.)
k_test = k_data
#print k_test
#print vsigma_sq(1./k_test)

PS_test = P_lin.P(k_test,z)
P_data_1 = P_data_1/(k_data**3/(2*np.pi**2))

pylab.plot(k_test,  PS_test*k_test**3/(2*np.pi**2), color = 'red' )
pylab.plot(k_data, P_data_1*k_data**3/(2*np.pi**2), color = 'blue')

pylab.xscale('log')
pylab.yscale('log')
pylab.show()

mult_factor = P_data_1/PS_test
pylab.plot(k_test,mult_factor)
pylab.xscale('log')
pylab.xlim([10**(-3),10])
pylab.ylim([0,5])
pylab.show()

"""



def Omega_m(Omega_m0,Omega_lambda0,z):
    
    return Omega_m0*(1.+z)**3/(Omega_lambda0 + (1.- Omega_lambda0-Omega_m0)*(1.+z)**2+ Omega_m0*(1.+z)**3)

def Delta_Q_sq(Omega_m0,Omega_b,Omega_c, Omega_lambda0, h, n_s,sigma_8,k, z, k_sigma, n_eff, C):
    
    Delta_L_sq = P_lin.P(k,z, Omega_m0,Omega_b,Omega_c,Omega_lambda0,h, n_s,sigma_8)*k**3/(2*np.pi**2)
    
    y = k/k_sigma
    f_y = y/4. +y**2/8.
    
    alpha_n = abs(6.0835+ 1.3373*n_eff - 0.1959*n_eff**2 - 5.5274*C)
    #print 'alpha', alpha_n
    beta_n = 2.0379 - 0.7354*n_eff + 0.3157*n_eff**2 + 1.2490*n_eff**3 + 0.3980*n_eff**4 - 0.1682*C
    #print 'beta', beta_n
    Delta_sq = Delta_L_sq*(1.+Delta_L_sq)**beta_n/(1.+alpha_n*Delta_L_sq)*np.exp(-f_y)
    
    return Delta_sq

def Delta_H_sq(Omega_m0,Omega_lambda0,k,z, k_sigma, n_eff, C):
    
    y = k/k_sigma
    Omega_m_z = Omega_m(Omega_m0,Omega_lambda0,z)
    
    f1 = Omega_m_z**(-0.0307)
    f2 = Omega_m_z**(-0.0585)
    f3 = Omega_m_z**(0.0743)

    #a_n = 10**(1.5222  +2.8553*n_eff +2.3706*n_eff**2 +0.9903*n_eff**3 +0.2250*n_eff**4 -0.6038*C +0.1749*Omega_w(z)*(1.+w))
    #b_n = 10**(-0.5642 +0.5864*n_eff +0.5716*n_eff**2                                   -1.5474*C +0.2279*Omega_w(z)*(1.+w))

    a_n = 10**(1.5222  +2.8553*n_eff +2.3706*n_eff**2 +0.9903*n_eff**3 +0.2250*n_eff**4 -0.6038*C )
    b_n = 10**(-0.5642 +0.5864*n_eff +0.5716*n_eff**2                                   -1.5474*C )
    c_n = 10**(0.3698  +2.0404*n_eff +0.8161*n_eff**2                                   +0.5869*C )
    gamma_n =  0.1971  -0.0843*n_eff                                                    +0.8460*C
    
    mu_n = 0.
    nu_n = 10**(5.2105 + 3.6902*n_eff)

    
    
    Delta_sq_prime = a_n*y**(3.*f1)/(1.+ b_n*y**f2 + (c_n*f3*y)**(3.- gamma_n))
    Delta_sq = Delta_sq_prime/ (1. + mu_n*y**(-1) + nu_n*y**(-2))
    
    return Delta_sq


def P_nonlin(k,z,h=0.7,Omega_m0= 0.3,Omega_lambda0= 0.7, Omega_b= 0.05, Omega_c =0.25,n_s = 0.96,sigma_8 = 0.8 ):
    """
    k_test2 = (np.arange(10000)+1.)/1000.
    index = np.where(abs(vsigma_sq(1./k_test2,z) - 1.) == min(abs(vsigma_sq(1./k_test2,z) - 1.)))[0]
    k_sigma = k_test2[index]
    R_star = 1./k_sigma
    n_eff = - d_ln_sigma_sq(R_star,z) -3.
    C = - d2_ln_sigma_sq(R_star,z)
    #return z, k_sigma, n_eff, C
    """
    
    index = np.where(z == z_array)[0]
    if len(index) != 1:
        print 'error'
    k_sigma = k_sigma_array[index]
    n_eff = n_eff_array[index]
    C = C_array[index]
    
    #A = np.diff(np.log(vsigma_sq(1./k_test2)))/np.diff(np.log(1./k_test2))
    #C = - (np.diff(A)/np.diff(np.log(1./k_test2[1:])))[index-2]
    

    P_Q = Delta_Q_sq(Omega_m0,Omega_b,Omega_c,Omega_lambda0,h, n_s,sigma_8,k,z, k_sigma, n_eff, C)*2*np.pi**2/k**3
    P_H = Delta_H_sq(Omega_m0,Omega_lambda0,k,z, k_sigma, n_eff, C)*2*np.pi**2/k**3
    
    P = P_Q + P_H
    
    return P
"""

P = P_nonlin(k_test,z, h=0.7,Omega_m0= 0.3,Omega_lambda0= 0.7, Omega_b= 0.05, Omega_c =0.25,n_s = 0.96,sigma_8 = 0.8 )

#pylab.plot(k_test, P_H*k_test**3/(2*np.pi**2), label = 'H')
#pylab.plot(k_test, P_Q*k_test**3/(2*np.pi**2), label = 'Q')

pylab.plot(k_test, P*k_test**3/(2*np.pi**2), label = 'H + Q')
pylab.plot(k_data, P_data_1*k_data**3/(2*np.pi**2), label = 'P_nl')

pylab.xscale('log')
pylab.yscale('log')
pylab.xlim([10**(-3),100])
pylab.ylim([10**(-6), 2*10**3])
pylab.legend()
pylab.show()


mult_factor = P_data_1/P
pylab.plot(k_test,mult_factor)
pylab.xscale('log')
pylab.xlim([10**(-3),100])
pylab.show()
"""

