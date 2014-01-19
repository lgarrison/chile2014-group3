r"""
=========================================================
Periodic Gaussian Processes Regression
=========================================================

WARNING: This class probably won't work if you don't fix this!
_arg_max_reduced_likelihood_function() in sklearn's gaussian_process.py
contains an error.  This section:
    constraints = []
    for i in range(self.theta0.size):
        constraints.append(lambda log10t:
                           log10t[i] - np.log10(self.thetaL[0, i]))
        constraints.append(lambda log10t:
                           np.log10(self.thetaU[0, i]) - log10t[i])
should read
    constraints = []
    for i in range(self.theta0.size):
        constraints.append(lambda log10t, i=i:
                           log10t[i] - np.log10(self.thetaL[0, i]))
        constraints.append(lambda log10t, i=i:
                           np.log10(self.thetaU[0, i]) - log10t[i])
(Note the i=i)


"""
# print(__doc__)

# Author: Lehman Garrison <lgarrison@cfa.harvard.edu>

import numpy as np
from sklearn.gaussian_process import GaussianProcess
import pandas as pd
import numpy as np
import pdb
from scipy.stats import norm

np.random.seed(5)

def periodic_exponential(theta, d):
    """
    A periodic exponential correlation model.
    
    Parameters
    ----------
    theta : array_like
        An array with shape 2, where theta[...,0] is the period, and
        theta[...,1] is the width of the exponential.

    dx : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    """

    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)
    
    if np.isnan(theta[...,1]):
        print "Warning: theta1 nan"
        return np.zeros(d.shape[0])
    try:
        return np.exp(-2*np.sum(np.sin(d/theta[...,0] * np.pi) ** 2, axis=1)/theta[...,1]**2)
    except:
        pdb.set_trace()


class GaussianPeriodic:
    """
    Periodic Gaussian Process model class.
    Initializing the object fits the model and prepares P()
    to be queried.

    Parameters
    ----------
    t: double array_like
        Times of observations

    mag: double array_like
        Observed magnitudes
        
    mag_err: double array_like
        Error in the magnitude observations
        
    period: double
        Best guess of the period
        
    period_err: double
        Error in the period
        
    verbose: bool, optional
        Set to True to see the progress of building the model.
        Default is False.
        
    N_samp: int, optional
        The number of period samples in determining the error
        due to period uncertainty.  Default 30

    """
    def __init__(self, t, mag, mag_err, period, period_err, verbose=False, N_samp=30):
        self.X = t
        self.y = mag
        self.dy = mag_err
        self.period = period
        self.period_uncertainty = period_err
        self.verbose = verbose
        self.N_samp = N_samp
        
        self._init_model()
        
           
    
    def _init_model(self):
        """
        Initializes the model. (internal use)
        
        """
        self.gp = GaussianProcess(normalize=False, regr='constant', corr=periodic_exponential, theta0=(self.period,1),thetaL=(self.period,3e-1), thetaU=(period,5),nugget=(self.dy / self.y) ** 2,random_start=100, verbose=self.verbose)
        self.gp.fit(np.atleast_2d(self.X).T, self.y)
        
        self.period_trials = np.random.normal(self.period,self.period_uncertainty,self.N_samp)
    
    
    def P(self, d, t):
        """
        Returns the probability of observing data value(s) d at time(s) t,
        using the fitted model that was created when this object was initialized.
        The errors come from the regression itself and from propagation of the
        uncertainty in the period.
        
        Parameters
        ----------
        d: double array_like
            data values
        
        t: double array_like
            times of observations
        
        Returns
        -------
        prob: double array_like
            The probability of observing value d at time t
        """
        
        if len(d) != len(t):
            raise ValueError('d and t must have the same size')
            return None
        
        t = np.atleast_2d(t).T
        
        # Calc regression mean and uncertainty
        y_pred, MSE = self.gp.predict(t, eval_MSE=True)
        sigma = np.sqrt(MSE)
        
        # Calc uncertainty from period
        y_err = self.gp.predict(((t-self.X[0])*(self.period_trials/self.period)+t[0]).reshape(-1,1)).reshape(len(y_pred),self.N_samp) - y_pred.reshape(len(y_pred),1)
        yplus_uncert = np.ma.masked_less(y_err,0.).mean(axis=1)
        yminus_uncert = -np.ma.masked_greater(y_err,0.).mean(axis=1)
        
        y_above = d > y_pred
        yplus_uncert.mask |= ~y_above
        yminus_uncert.mask |= y_above
        yplus_uncert = yplus_uncert.filled(0.)
        yminus_uncert = yminus_uncert.filled(0.)
        
        sigma = sigma + yplus_uncert + yminus_uncert
        
        prob = norm.pdf(d, loc=y_pred, scale=sigma)
        
        return prob
    
    

if __name__ == '__main__':
    print 'Testing GaussianPeriodic...'
    
    filename = r'..\test_lc\Obj027_148.812762_3.354973'
    period = .608057
    period_uncertainty = .01
    first = 16
    jd, mag, mag_err= np.loadtxt(filename,unpack=True, usecols=[0, 1, 2])
    jd = jd[:first]
    mag = mag[:first]
    mag_err = mag_err[:first]
    
    gp = GaussianPeriodic(jd, mag, mag_err, period, period_uncertainty, verbose=True)
    querytimes = jd[-1] + (jd[-1] - jd[0])*np.array([.02,.03, .1])
    print gp.P([mag[-1]]*len(querytimes), querytimes)
