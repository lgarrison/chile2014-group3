import numpy as np
import scipy
import scipy.stats as st
import pandas as pd
import sys

import astropy
import astroquery

# astropy modules
import astropy.coordinates as coord
from astropy import units as u
from astropy.utils.data import REMOTE_TIMEOUT

# astroquery modules
from astroquery.simbad import Simbad 
from astroquery.nrao import Nrao
from astroquery.ukidss import Ukidss
from astroquery.vizier import Vizier
from astroquery import fermi #### CHECK ####
from astroquery import ogle  #### CHECK ####

#############################################################################
# TODO About the code, how to run it

# SEARCH_OBJECTS.PY searches for information from the following databases:
# Simbad
# NRAO
# UKIDSS #not stable
# Vizier
#############################################################################



# SEARCH_CATALOGS function searches for information from databases
def search_catalogs(n, catalog_name, position):
    #n = len(data)
    flag = 0
    
    for i in np.arange(n):
        try:
            with REMOTE_TIMEOUT.set_temp(30):
                cc = catalog_name.query_region(position[i])
            if len(cc) == 0:
                continue
            else:
                cc_table = cc
                flag = i
                break
        except (urllib2.URLError, urllib.error.URLError, timeout):
            print "This remote file couldn't be found, or it took too long to query the data :( Try running the script again."
            continue
        
    # if empty, exit; otherwise, grow table       
    if flag == 0:
        print '\n'
        print 'No information available from ', str(catalog_name)
        print 'The table for this database will be empty.'
        return None
    
    else:   
        for j in np.arange(flag,n):
            try:
                with REMOTE_TIMEOUT.set_temp(30):
                    cc_also = catalog_name.query_region(converted_str[j])
                if len(cc_also) == 0:
                    continue
                else:
                    cc_table.add_row(cc_also[0])
            except timeout:
                continue
        return cc_table


def main():

	# initialize vars
	radius   = '2d0m0s' #### CHECK (do we need this?) ####
	catalogs = [Simbad, Nrao, Ukidss]
	names    = ['Simbad', 'Nrao', 'Ukidss']
	n 		 = len(objects)

	# read in data & convert posititon to arcdeg/min/sec
	objects   = pd.read_table('Variables_var', header=None, 
		names=['obj', 'ra', 'dec', 'N', 'mean', 'median', 'rms', 
		'median_err', 'skewness', 'chi2', 'sigma', 'dr_rms'])
	converted = coord.ICRS(ra=objects['ra'], dec=objects['dec'], 
		unit=(u.degree, u.degree))
	converted_str = converted.to_string()

	# run
	for k in np.arange(len(catalogs)):
	    table_per_catalog = search_catalogs(n, catalogs[k], converted_str)
	    if table_per_catalog is None:
	        continue
	    else:
	    	# if table is not empty, write to csv
	        filename = names[k] + '_test.csv'
	        ascii.write(table_per_catalog, filename, format='csv')

	return 0

if __name__ == '__main__':
	exit(main())






