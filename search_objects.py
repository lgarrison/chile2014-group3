import numpy as np
import scipy
import scipy.stats as st
import pandas as pd
import sys
import urllib2
import socket

import astropy
import astroquery

# astropy classes
import astropy.coordinates as coord
from astropy import units as u
from astropy.io import ascii
from astropy.utils.data import REMOTE_TIMEOUT

# astroquery classes
from astroquery.ned import Ned
from astroquery.simbad import Simbad 
from astroquery.nrao import Nrao
from astroquery.ukidss import Ukidss
from astroquery.vizier import Vizier
import astroquery.exceptions as exceptions

#############################################################################
# SEARCH_OBJECTS.py performs a conesearch around an object and retrieves  
# information from all catalogs in the following databases:
# 	- NED
# 	- Simbad
# 	- NRAO
# 	- UKIDSS #instability exists with this database
# 	- Vizier

# INPUT DATA
# The data file should have the following columns in the following order:
# obj = object ID
# ra = RA
# dec = DEC
# N = number of observations
# mean = average of magnitudes
# median = median of magnitudes
# rms = rms of the photometry
# median_err = median photometric uncertainty
# skewness = skewness of magnitudes
# chi2 = chi-squared statistic of magnitudes
# sigma = variance of magnitudes
# dr_rms = rms in the position (not very useful)

# ASSUMPTIONS & NOTES
# 1. Data  is saved as a file named 'Variabes_var'; in the same directory as  script.
# 2. Radius is 5 arcseconds
# 3. Number of objects returned per inputted object is less than/equal to 15 
# 4. If information is retrieved, a csv file is outputted for that database
# 5. There are a lot of warnings that are outputted, but this is okay. Modify code
#   to suppress if it's annoying enough.

# author: Anita Mehrotra, anitamehrotra@fas.harvard.edu
# date: January 13, 2014
# department: CSE SEAS, Harvard University
# project: CHILE 2014
#############################################################################


# SEARCH_CATALOGS function searches for information from databases listed above
def search_catalogs(*args):
    
    n, catalog_name, position, radius = args[0], args[1], args[2], args[3]
    flag = 0
    
    for i in xrange(n):
        try:
            with REMOTE_TIMEOUT.set_temp(30):
                cc = catalog_name.query_region(position[i], radius=radius)
            if len(cc) == 0:
                continue
            else:
                cc_table = cc
                flag = i
                break
        except (urllib2.URLError, socket.timeout, exceptions.RemoteServiceError):
            print "This remote file/object couldn't be found, or it took too long to query the data :("
            print " Try running the script again."
            break
        
    # if empty, exit; otherwise, grow table       
    if flag == 0 and len(cc) == 0:
        print "\n"
        print "No information available from ", str(catalog_name)
        print "The table for this database will be empty."
        return None
    else:   
        for j in xrange(flag,n):
            try:
                with REMOTE_TIMEOUT.set_temp(30):
                    cc_also = catalog_name.query_region(position[j], radius=radius)
                if len(cc_also) == 0:
                    continue
                else:
                    cc_table.add_row(cc_also[0])
            except (socket.timeout, exceptions.RemoteServiceError):
                continue
        return cc_table


def main():
	
    # read in data & convert posititon to arcdeg/min/sec
    objects   = pd.read_table('Variables_var', header=None, names=['obj', 'ra', 'dec', 'N', 'mean', 'median', 'rms', 'median_err', 'skewness', 'chi2', 'sigma', 'dr_rms'])
    converted = coord.ICRS(ra=objects['ra'], dec=objects['dec'], unit=(u.degree, u.degree))
    converted_str = converted.to_string()

    # initialize vars
    radius   = '0d0m2s'
    #catalogs = [Ned, Simbad, Nrao, Ukidss, Vizier]
    #names    = ['NED', 'Simbad', 'NRAO', 'UKIDSS', 'VizieR']
    catalogs = [Simbad, Nrao, Ukidss, Vizier]
    names    = ['Simbad', 'NRAO', 'UKIDSS', 'VizieR']
    n = len(objects)
    m = len(catalogs)

    # execute search_catalogs()
    for k in xrange(m):
        print "\n"
        print "Searching ", names[k], "..."

        catalogs[k].ROW_LIMIT = 15
        table_per_catalog = search_catalogs(n-6100, catalogs[k], converted_str, radius)
        
        if n%10 == 0:
            print "\n"
            print n

        if table_per_catalog is None:
            continue
        else:
            filename = names[k] + '_test.csv'
            ascii.write(table_per_catalog, filename, format='csv')
	
    print "Finished!"
    
    return 0


if __name__ == '__main__':
	main()

	exit(main())
