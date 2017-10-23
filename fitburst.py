import sqlutil as sqlutil
from astropy.io import ascii
import matplotlib.pyplot as plt
import emcee
import numpy as np
import scipy.optimize as op
import corner


data = ascii.read('data/sn.csv',format='csv')

nullmask = data['averagemag.'] != 'null'
data = data[nullmask]

N = len(data['averagemag.'])

x = np.array(data['JD'])
y = np.array(data['averagemag.'],dtype=float)



#generate some errors
yerr = 0.1+0.5*np.random.rand(N)

plt.scatter(x,y)
plt.show()


def model(x,b_0,alpha,A,T):
	out = np.array([])
	for i in range(0,len(x)):
		print(i)
		if x[i] < T:
			out = np.append(out,b_0)
		else:
			out = np.append(out, b_0 +  A * np.exp(-alpha*(x[i]-T)))
	return out


def lnlike(theta,x,y,yerr):
	b_0,A,alpha,T = theta
	inv_sigma2 = 1.0/(yerr**2)
	modelval = model(x,b_0,alpha,A,T)
	return -0.5*(np.sum((y-modelval)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnprior(theta):
	b_0, A, alpha, T = theta
	if 0.0 < b_0 < 50.0 and 0.0 < A < 50.0 and 0.0 < alpha < 50 and 2456871.00 < T < 2557175.078:
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)





ndim, nwalkers = 4, 200
pos = [[20.0,15.0,0.5,2456871.253] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

sampler.run_mcmc(pos, 1000)

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

fig = corner.corner(samples)
fig.savefig("triangle.png")

