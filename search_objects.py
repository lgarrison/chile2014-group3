import astropy
import astroquery
import pandas as pd
import numpy as np

def main():
	#N: is the number of observations
	#rms: rms of the photometry
	#median_err: is the median photometric uncertainty
	#dr_rms: is the rms in the position (not very useful)
	objects = pd.read_table('Variables_var', header=None, names=['obj', 'ra', 'dec', 'N', 'mean', 'median', 'rms', 'median_err', 'skewness', 'chi2', 'sigma', 'dr_rms'])
	
	# use astroquery here, using objects['ra'] and objects['dec'] as the positions
	
	return 0

if __name__ == '__main__':
	exit(main())
