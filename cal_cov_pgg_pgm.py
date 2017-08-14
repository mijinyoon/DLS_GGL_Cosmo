import numpy as np
import pylab
Nfield = 5

Pgm_jk = [None]*Nfield
Pgg_jk = [None]*Nfield


for field_num in range(1,6):
    
    Pgm_jk[field_num-1] = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/DLS_Cosmo/Pgm_jk/pgm_F%s.dat"%field_num)
    Pgg_jk[field_num-1]  = np.loadtxt("/Users/mjyoon/Desktop/mjyoon/DLS_Cosmo/Pgg_jk/pgg_F%s.dat"%field_num)
    
    
Pgm_allfield_jk = np.vstack((Pgm_jk[0],Pgm_jk[1],Pgm_jk[2],Pgm_jk[3],Pgm_jk[4]))
Pgm_mean = np.mean(Pgm_allfield_jk, axis = 0)
print np.shape(Pgm_mean)
print Pgm_mean
Pgg_allfield_jk = np.vstack((Pgg_jk[0],Pgg_jk[1],Pgg_jk[2],Pgg_jk[3],Pgg_jk[4]))
Pgg_mean = np.mean(Pgg_allfield_jk, axis = 0)
print np.shape(Pgg_mean)
print Pgg_mean
                            
jk_num = 400*5
print 'jk_num',jk_num

delta_Pgm = (Pgm_allfield_jk - Pgm_mean)
delta_Pgg = (Pgg_allfield_jk - Pgg_mean)


delta_P = np.hstack((delta_Pgm, delta_Pgg))
print np.shape(delta_P)
cov = np.dot(delta_P.T, delta_P)/(jk_num -1.)
print np.shape(cov)
#print cov
"""
#np.savetxt("covariance_Pgm_Pgg_F%s.txt"%field_num, cov)
pylab.imshow(cov[0:5,0:5])
pylab.colorbar()
pylab.title('Covariance Pgm F%s'%field_num)
pylab.show()

pylab.imshow(cov[5:10,5:10])
pylab.colorbar()
pylab.title('Covariance Pgg F%s'%field_num)
pylab.show()
"""
pylab.imshow(cov)
pylab.title('Covariance P')
pylab.colorbar()
pylab.savefig('Covariance_pgm_pgg.png')
pylab.show()

pylab.imshow(cov[0:5,0:5])
pylab.title('Covariance Pgm')
pylab.colorbar()
pylab.savefig('Covariance_pgm.png')
pylab.show()


pylab.imshow(cov[5:10,5:10])
pylab.title('Covariance Pgg')
pylab.colorbar()
pylab.savefig('Covariance_pgg.png')
pylab.show()

np.savetxt("covariance_Pgm_Pgg.txt", cov)
