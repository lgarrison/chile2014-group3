import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import emcee


#############################################################################
# COMPUTE_EI.py 
# <TODO>

# author: Anita Mehrotra, anitamehrotra@fas.harvard.edu
# date: January 18, 2014
# department: CSE SEAS, Harvard University
# project: CHILE 2014
#############################################################################


# Celestial object model, denoted M_e
# 	- variable stars = Gaussian Process
#	- supernovae = template
def model(theta):
    return theta*3 + 100

# Generative probabilistic model (using Bayes, MCMC); contains the following fxns
# define log likelihood
def lnlike(theta, Di):
    mu, sigma = theta #parameters
    inv_sigma2 = 1./(2*np.square(sigma))
    C = (-len(Di)/2.)*(np.log(2*np.pi) + np.log(np.square(sigma)))
    return C - inv_sigma2 * (sum(np.square(Di - mu)))

# define log probability
def lnprob(theta, Di):
    lnprior = 1 # assume flat prior (uniform distribution), proportional to 1
    return lnprior*lnlike(theta, Di)

# compute Phat
def Phat(d):
    N = len(thetas)
    phat = np.zeros(N)
    for j in xrange(N):
        #phat[j] = (1./N)*(sum(d[:,j]))
        phat[j] = (1./N)*(sum(d[j]))
    #return np.mean(d, axis=1)
    return phat

# compute expected information at time e
def EI(*args):
    ln_Ptilde = args[0]
    return np.mean(ln_Ptilde)


def main():

    # load data
    #input_file = sys.argv[1]
    input_file = '/Users/anita/Documents/CHILE/data/test_lc/Obj133_148.304044_1.378136'
    jd, mag, mag_err= np.loadtxt(input_file,unpack=True, usecols=[0, 1, 2])
    submag = mag[0:16]

    # initialize variables
    nwalkers = 100
    ndim = 2
    nsamples = 500
    burn_in = 50

    # use MCMC to determine parameters of generative model (assuming normal)
    initial = np.mean(submag)
    pos = [initial + np.random.rand(ndim)*1e-4 for i in range(N)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[submag])
    results = sampler.run_mcmc(pos, nsamples)
    
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    samples[:,1] = np.exp(samples[:,1])

    # parameters for distribution
    mu_mcmc, sigmasq_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                     zip(*np.percentile(samples, [16,50,84],
                                                        axis=0)))

    # define pdf and draw theta values
    thetas = np.array(np.random.normal(mu_mcmc[0], sigmasq_mcmc[0], 3))

    # generate "new" data
    d = np.array(model(thetas))

    # compute Ptilde
    Ptilde = Phat(d)
    lnPtilde = np.log(Ptilde)

    EI_e = EI(lnPtilde)

    print "expected information at time e = ", EI_e

    return 0


if __name__ == '__main__':
    main()




