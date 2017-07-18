import numpy as np
import P_lin
import pylab
from scipy.integrate import simps

def sigma_sq(R):
    
    k = 10**(-10.+ np.arange(200)/10.)
    Delta_L_sq = P_lin.P(k)*k**3/(2*np.pi**2)
    
    return simps(Delta_L_sq*np.exp(-k**2*R**2), np.log(k))
vsigma_sq = np.vectorize(sigma_sq)

def d_fun(x):
    h = 1e-10
    return (fun(x+h)-fun(x-h))/(2*h)


k_test = 10**(-10.+ np.arange(400)/20.)
print k_test
print vsigma_sq(1./k_test)
#pylab.plot(k_test,vsigma_sq(1./k_test) )
#pylab.xscale('log')
#pylab.yscale('log')
#pylab.show()

print min(abs(vsigma_sq(1./k_test)-1.))

index = np.where(abs(vsigma_sq(1./k_test) - 1.) == min(abs(vsigma_sq(1./k_test) - 1.)))[0]
print 'index', index
k_sigma = k_test[index]
print 'k_sigma', k_sigma

n_eff = - (np.diff(vsigma_sq(1./k_test))/np.diff(1./k_test))[index-1] -3.
print n_eff
A = np.diff(vsigma_sq(1./k_test))/np.diff(1./k_test)
C = - (np.diff(A)/np.diff(1./k_test[1:]))[index-2]
print C

Omega_m = 0.3
def Delta_Q_sq(k):
          
    Delta_L_sq = P_lin.P(k)*k**3/(2*np.pi**2)
    y = k/k_sigma
    f_y = y/4. +y**2/8.
    
    alpha_n = abs(6.0835+ 1.3373*n_eff - 0.1959*n_eff**2 - 5.5274*C)
    beta_n = 2.0379 - 0.7354*n_eff + 0.3157*n_eff**2 + 1.2490*n_eff**3 + 0.3980*n_eff**4 - 0.1682*C
    
    Delta_sq = Delta_L_sq*((1.+Delta_L_sq)**beta_n/(1.+alpha_n*Delta_L_sq))*np.exp(-f_y)
    
    return Delta_sq

def Delta_H_sq(k):
          
    Delta_L_sq = P_lin.P(k)*k**3/(2*np.pi**2)
    y = k/k_sigma
    
    f1 = Omega_m**(-0.0307)
    f2 = Omega_m**(-0.0585)
    f3 = Omega_m**(0.0743)

    #a_n = 10**(1.5222  +2.8553*n_eff +2.3706*n_eff**2 +0.9903*n_eff**3 +0.2250*n_eff**4 -0.6038*C +0.1749*Omega_w(z)*(1.+w))
    #b_n = 10**(-0.5642 +0.5864*n_eff +0.5716*n_eff**2                                   -1.5474*C +0.2279*Omega_w(z)*(1.+w))
    a_n = 10**(1.5222  +2.8553*n_eff +2.3706*n_eff**2 +0.9903*n_eff**3 +0.2250*n_eff**4 -0.6038*C )
    b_n = 10**(-0.5642 +0.5864*n_eff +0.5716*n_eff**2                                   -1.5474*C )
    c_n = 10**(0.3698  +2.0404*n_eff +0.8161*n_eff**2                                   +0.5869*C)
    gamma_n =  0.1971  -0.0843*n_eff                                                    +0.8460*C
    
    Delta_sq = a_n*y**(3.*f1)/(1.+ b_n*y**f2 + (c_n*f3*y)**(3.- gamma_n))

    return Delta_sq


def P_nonlin(k,z):

    Delta_sq = Delta_Q_sq(k) + Delta_H_sq(k)
    P = Delta_sq*2.*np.pi**2/k**3
    
    return P

PS_test = P_nonlin(k_test,0)
pylab.plot(k_test, PS_test)
pylab.xscale('log')
pylab.yscale('log')
pylab.show()
