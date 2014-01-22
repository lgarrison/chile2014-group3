import sys
import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import norm
import pandas as pd

from sklearn.gaussian_process import GaussianProcess
import pdb
from gaussian_periodic import *


#############################################################################
# COMPUTE_EI.py 

# <TODO>

# Celestial object models for the following types:
# 	- variable stars = Gaussian Process
#	- supernovae = template

# author: Anita Mehrotra, anitamehrotra@fas.harvard.edu
# date: January 21, 2014
# department: CSE SEAS, Harvard University
# project: CHILE 2014
#############################################################################


# covert Lomb-Scargle significance to proportions
def convert_sig(sig):
    """
    Parameters
    ----------
    sig: a numpy array containing float64's
    """
    return sig/sum(sig)

# compute EI
def EI(ln_Ptilde):
    """
    Parameters
    ----------
    ln_Ptilde: a multidimensional array containing values for different
        theta_j at different times e
    """
    return np.mean(ln_Ptilde)

# compute Phat
def Phat(d):
    """
    Parameters
    ----------
    d: a multidimensional array containing values for a particular 
        theta_j at different times e
    """
    return np.mean(d, axis=1)

# create full list of thetas, uncertainties
def full_list(*args):
    """
    Parameters
    ----------
    # TODO
    """
    items, prob, size_full_list = args[0], args[1], args[2]
    full_list = []
    curr_index = 0
    vals = np.floor(prob*size_full_list)
    
    for i in xrange(len(vals)):
        curr_index = curr_index + vals[i]
        intermediate = np.ones(int(vals[i]))*items[i]
        full_list.append(intermediate)
        if i==len(vals)-1:
            intermediate = np.ones(int(size_full_list-curr_index))*items[i]
            full_list.append(intermediate)
    
    return np.hstack(full_list)

# compute indices
def compute_indices(*args):
    """
    Parameters
    ----------
    # TODO
    """
    items, prob, size_full_list = args[0], args[1], args[2]
    
    full_list_items = np.zeros(size_full_list)
    indices = np.zeros(2*len(items))
    vals = np.floor(prob*size_full_list)
    
    # compute & store indices
    for i in xrange(len(vals)):
        
        # starting values
        if i == 0:
            indices[i] = 0 #start if i=0
        else:
            indices[i*2] = indices[i*2-1] + 1
        
        # ending values
        if i == len(vals)-1:
            indices[i*2+1] = size_full_list-1
        else:
            indices[i*2+1] = indices[i*2] + vals[i] - 1 #end
    
    return indices.astype(int)


def main():

    # load data
    data_input_file = sys.argv[1]
    theta_input_file = sys.argv[2]
    e = np.array(sys.argv[3])
    #data_input_file  = '/Users/anita/Documents/CHILE/data/test_lc/Obj133_148.304044_1.378136'
    #theta_input_file = '/Users/anita/Documents/CHILE/data/test_theta.txt'
    #e =  np.array([.02,.03, .1])
    
    jd, mag, mag_err = np.loadtxt(data_input_file, unpack=True, usecols=[0, 1, 2])
    obj_id, theta0, theta1, theta2, dp0, dp1, dp2, sig0, sig1, sig2 = np.loadtxt(theta_input_file, unpack=True, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    

    ##### this portion for testing only #####
    first = 16
    jd = jd[:first]
    mag = mag[:first]
    mag_err = mag_err[:first]
    ##### end testing portion #####

    # initial vars
    N = 50  # number of thetas i.e. i = {1, 2, ..., N}
    M = 100 # number of new data points d_j i.e. j = {1, 2, ..., M}
    ptildes = np.zeros([N, len(e)])
    EI_e = []

    # work with one object for now
    obj_id = obj_id[0]
    theta0 = theta0[0]
    theta1 = theta1[0]
    theta2 = theta2[0]
    dp0 = dp0[0]
    dp1 = dp1[0]
    dp2 = dp2[0]
    sig0 = sig0[0]
    sig1 = sig1[0]
    sig2 = sig2[0]

    sig = np.array([sig0, sig1, sig2], dtype=float)
    uncert = np.array([dp0, dp1, dp2], dtype=float)
    thetas = np.array([theta0, theta1, theta2], dtype=float)

    # compute probabilities 
    prob = sig/sum(sig)

    # produce full lists; needed to run Gaussian Process
    allthetas = full_list(thetas, prob, N)
    alluncertainty = full_list(uncert, prob, N)
    indices = compute_indices(thetas, prob, N)

    # run Gaussian Process
    for i in xrange(N):
        print "i =", i # track progress
        period = allthetas[i]
        period_uncertainty = alluncertainty[i]
        
        gp = GaussianPeriodic(jd, mag, mag_err, period, period_uncertainty, verbose=True)
        querytimes = jd[-1] + (jd[-1] - jd[0])*e
        
        # produce "new data" (samples from Gaussian Process)
        dj = gp.sample_d(M,querytimes)

        # compute phat per theta
        ptildes[i,:] = Phat(dj)

    # compute Expected Information per e
    for k in xrange(len(thetas)):
        start = indices[k]
        end = indices[k]+1
        ptildes_per_theta = ptildes[start:end]
        lnptildes = np.log(ptildes_per_theta)
        EI_e.append(EI(lnptildes))

    # normalize Expected Information, and output to csv
    #normEI = (EI_e - min(EI_e))/(max(EI_e) - min(EI_e))
    ee = np.concatenate(([e], [EI_e]), axis=0)
    np.savetxt("expected_information.csv", np.asarray(ee), delimiter=',')

    print "See expected_information.csv for output."

    return 0


if __name__ == '__main__':
    main()




