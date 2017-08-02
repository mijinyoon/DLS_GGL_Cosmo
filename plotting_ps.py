import numpy as np
import pylab

data = np.loadtxt("P_ell_gm.txt")
ell = data[:,0]
P = data[:,1]
pylab.plot(ell, P, label = 'P_gm')



data = np.loadtxt("P_ell_gg_lens.txt")

ell = data[:,0]
P2 = data[:,1]

pylab.plot(ell, P2, label= 'P_gg')



pylab.legend()
pylab.xscale('log')
pylab.yscale('log')
pylab.xlim(10,2000)
pylab.ylim(10**(-9),10**(-4))

pylab.tick_params(bottom= 'on', top = 'on', left = 'on',right = 'on', direction = 'in', which = 'both')
pylab.show()

"""
pylab.plot(ell, P3/P2)
pylab.show()
"""
