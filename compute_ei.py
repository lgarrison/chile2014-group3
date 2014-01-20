import numpy as np
import sys
import emcee

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


#############################################################################
# COMPUTE_EI.py 

# <TODO>

# Celestial object models for the following types:
# 	- variable stars = Gaussian Process
#	- supernovae = template

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
    N = len(d)
    phat = np.zeros(N)
    for j in xrange(N):
        if isinstance(d[j], float):
            phat[j] = (1./N)*d[j]
        else:
            #phat[j] = (1./N)*(sum(d[:,j]))
            phat[j] = (1./N)*sum(d[j])
    #return np.mean(d, axis=1)
    return phat

# compute expected information at time e
def EI(ln_Ptilde):
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
    steps = np.arange(0,nsamples)

    # use MCMC to determine parameters of generative model (assuming normal)
    initial = np.mean(submag)
    pos = [initial + np.random.rand(ndim)*1e-4 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[submag])
    results = sampler.run_mcmc(pos, nsamples)
    
    # plot samples to determine burn-in; default is 30
    mychain = sampler.chain

    print "Displaying results for $\mu$..."
    for j in xrange(nwalkers):
        plt.plot(steps, mychain[j][:,0], 'b-')
    plt.title('$\mu$')
    plt.xlabel('steps')
    plt.show()

    print "Displaying results for $\sigma^2$..."
    for j in np.arange(nwalkers):
        plt.plot(steps, mychain[j][:,1], 'g-')
    plt.title('$\sigma^2$')
    plt.xlabel('steps')
    plt.show()

    # determine final set of samples
    burn_in = 30 # change depending on plots of theta
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
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

    print "Expected Information at time e = ", EI_e

    return 0


if __name__ == '__main__':
    main()




