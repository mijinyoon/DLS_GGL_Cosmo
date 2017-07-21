import numpy as np
from scipy.integrate import simps

#Eisenstein and Hu (1998)


c = 2.998 *10**5 #km/s

H0 = 70 #km/s Mpc^-1
h = 0.7
omega_lambda = 0.7
n_s = 0.96 #
sigma_8 = 0.8 #

#rho_crit = 2.77*10**11*h**2 #Solar mass/Mpc^3
omega_b = 0.05
omega_c = 0.25
omega_m = omega_b +omega_c

w_m = omega_m*h**2
w_b = omega_b*h**2
w_c = omega_c*h**2

f_c = omega_c/omega_m
f_b = omega_b/omega_m
f_cb = (omega_c+ omega_b)/omega_m

theta_2p7 = 2.728/2.7

z_eq = 2.50*10**4*w_m*theta_2p7**(-4) #


R_eq = 31.5*w_b*theta_2p7**(-4)*(z_eq/10**3)**(-1) #
k_eq = 7.46*10**(-2)*w_m*theta_2p7**(-2) #

d1 = 0.313*(w_m)**(-0.419)*(1.+0.607*(w_m)**0.674) #
d2 = 0.238*(w_m)**0.223 #

z_d = 1291.* (w_m)**0.251/(1.+0.659*(w_m)**0.828)*(1+d1*(w_b)**d2)#
R_d = 31.5*w_b*theta_2p7**(-4)*(z_d/10**3)**(-1) #

s = 2./(3.*k_eq) * (6./R_eq)**0.5 *np.log(((1.+R_d)**0.5+(R_d+R_eq)**0.5)/(1.+(R_eq)**0.5))#



def f(z):
    
    return(omega_m*(1+z)**3 + (1-omega_m-omega_lambda)*(1+z)**2 + omega_lambda)**(0.5)

def Omega(z):
    
    return omega_m*(1+z)**3/f(z)**2

def Omega_lambda_1(z):
    
    return omega_lambda/f(z)**2

def D1(z):
    
    return 1./(1+z)*5./2.*Omega(z)*( Omega(z)**(4./7.) - Omega_lambda_1(z) + ( 1. + Omega(z)/2.)*(1. +Omega_lambda_1(z)/70.))**(-1)

def D(z):
    return D1(z)/D1(0)

#print D1(0)

def T_tilde(k,alpha,beta):
    
    q = k/(13.41*k_eq)#
    C = 14.2/alpha + 386./(1.+ 69.9*q**1.08)#
        
    return np.log(np.exp(1.)+1.8*beta*q)/(np.log(np.exp(1.)+1.8*beta*q)+C*q**2)#


def T_c(k):

    f = 1./(1.+(k*s/5.4)**4) #
    
    a1 =(46.9*w_m)**0.670*(1.+(32.1*w_m)**(-0.532)) #
    a2 = (12.0*w_m)**0.424*(1.+(45.0*w_m)**(-0.582))#
    
    alpha_c = a1**(-f_b)*a2**(-(f_b)**3) #

    b1 = 0.944*(1.+(458.*w_m)**(-0.708))**-1 #
    b2 = (0.395*w_m)**(-0.0266) #
    
    beta_c = (1. + b1*((f_c)**b2 - 1.))**(-1)#

    return f*T_tilde(k,1.,beta_c) + (1.-f)*T_tilde(k,alpha_c, beta_c)


def T_b(k):

    def G(y):
        return y*(-6.*np.sqrt(1.+y) +(2.+3.*y)*np.log(((1.+y)**0.5 +1. )/((1+y)**0.5 -1. )))
    
    alpha_b = 2.07*k_eq*s*(1.+R_d)**(-3./4.)*G((1.+ z_eq)/(1.+ z_d))#
    
    beta_node = 8.41*(w_m)**0.435 #
    s_tilde = s/(1.+(beta_node/(k*s))**3)**(1./3.) #
    k_silk = 1.6*(w_b)**0.52*(w_m)**0.73*(1.+(10.4*w_m)**(-0.95))#
    beta_b = 0.5  + f_b + (3. -2.*f_b)*np.sqrt((17.2*w_m)**2+1.) #
    
    return (T_tilde(k,1.,1.)/(1.+(k*s/5.2)**2) + alpha_b/(1.+(beta_b/(k*s))**3)*np.exp(-(k/k_silk)**1.4))*np.sin(k*s_tilde)/(k*s_tilde)


def T(k):
    return f_b*T_b(k) + f_c*T_c(k)

def T0(k):
    #zero-baryon
    
    Gamma = omega_m*h #
    q = k/h*theta_2p7**2/Gamma #
    L0  = np.log(2.*np.exp(1.) + 1.8*q) #
    C0 = 14.2 + 731./(1.+ 62.5*q)

    return L0/(L0 +C0*q**2)

def T1(k):
    #baryon included
    
    alpha_gamma = 1. - 0.328*np.log(431.*w_m)*f_b + 0.38*np.log(22.3*w_m)*f_b**2
    
    Gamma_eff = omega_m*h*(alpha_gamma + ((1. -alpha_gamma)/(1+(0.43*k*s)**4))) #
    q = k/h*theta_2p7**2/Gamma_eff
    L0  = np.log(2.*np.exp(1) + 1.8*q)
    C0 = 14.2 + 731./(1.+ 62.5*q)

    return L0/(L0 +C0*q**2)

def W(x):
    #return np.exp(-x**2/2.)
    return 3.*(np.sin(x) - x*np.cos(x))/x**3


def P(k,z):
    
    n_tilde = 0.
    delta_H = 1.94*10**(-5)*omega_m**(-0.785-0.05*np.log(omega_m))*np.exp(-0.95*n_tilde-0.169*n_tilde**2)
    PS = 2.*np.pi**2/k**3*(c*k/H0)**(3.+n_s)*delta_H**2*T(k)**2
    PS_norm = simps(1./(2*np.pi**2)*k**3*W(k*8./h)**2*PS, np.log(k))
    #print 'D(z)',D(z)
    #print 'D1(z)',D1(z)
    PS = PS/PS_norm*sigma_8**2*D(z)**2
          
    return PS

