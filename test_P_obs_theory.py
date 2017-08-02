#from P_gm_obs import *
from P_gm_theory import *
#from P_gg_obs import *
from P_gg_theory import *
import numpy as np
from multiprocessing import Pool
from datetime import datetime

#ell_min_range = np.round(2*10**(2+np.arange(5)/5.))
#ell_max_range = np.append(ell_min_range[1:]-1  ,2000.)
#ell = np.arange(200,210)

#print ell_min_range
#print ell_max_range
#print ell_max_range - ell_min_range +1
#print ell


b = 1.
#z = np.zeros(500)
#p_z = np.zeros(500)
"""
for field_num in range(1,6):
    data = np.loadtxt("z.dat")
    #data = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/data/p_of_z_lens/F%s_p_of_z_z1z2z3.txt"%field_num)
    z = data[:,0]
    #print z #checking to exclude the first point when z = 0.
    p_z += data[:,1]
    
"""
data = np.loadtxt("z.dat")
z_l = data[:,0]
p_z_l = data[:,1]

z_s = data[:,2]
p_z_s = data[:,3]


#z = z[10:101]
#p_z = p_z[10:101]
p_z_l = p_z_l/np.sum(p_z_l)/(z_l[1]-z_l[0])
p_z_s = p_z_s/np.sum(p_z_s)/(z_s[1]-z_s[0])

def P_gg_cal(ell):
    return P_gg_ell(ell, b,p_z_l,z_l)

def P_gm_cal(ell):
    return P_gm_ell(ell, b,z_l,p_z_l,z_s, p_z_s)

"""
for ell in range(10,2001):
    print P_gm_cal(ell)
"""
def mp_handler():
    A =  datetime.now()
    pool = Pool(20)
    ell_range =np.arange(10,2001)
    with open("P_ell_gg_lens.txt","w") as f:
        for P_gg_theory_zcomb in pool.imap(P_gg_cal, ell_range):
            f.write("%d %E\n"%(P_gg_theory_zcomb))
    B = datetime.now()
    print (B - A)


def mp_handler2():
    A =  datetime.now()
    pool = Pool(20)
    ell_range =np.arange(10,2001)
    with open("P_ell_gm.txt","w") as f:
        for P_gm_theory_zcomb in pool.imap(P_gm_cal, ell_range):
            f.write("%d %E\n"%(P_gm_theory_zcomb))
    B = datetime.now()
    print (B - A)

if __name__=='__main__':
    mp_handler2()

