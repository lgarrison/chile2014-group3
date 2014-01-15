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
# SEARCH_OBJECTS.py performs a conesearch around a given object and retrieves  
# information from all catalogs in the following databases:
# 	- NED
# 	- Simbad
# 	- NRAO
# 	- UKIDSS
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
# 1. Data is saved as a file named 'Variabes_var'; in the same directory as  script.
# 2. Radius is 2 arcseconds
# 3. Number of objects returned per inputted object is less than/equal to 15 
# 4. If information is retrieved, a csv file is outputted for that database
# 5. There are a lot of warnings that are outputted, but this is okay. Modify code
#   to suppress if it's annoying enough.

# EXCEPTIONS
# All exceptions are written to a file 'exceptions_stdout.txt.'
#   - urllib2.URLError = The remote file couldn't be found
#   - socket.timeout   = This object couldn't be found
#   - exceptions.RemoteServiceError = It took too long to connect with the database server & query the data
#   - Exception: Query Failed

# author: Anita Mehrotra, anitamehrotra@fas.harvard.edu
# date: January 15, 2014
# department: CSE SEAS, Harvard University
# project: CHILE 2014
#############################################################################


# SEARCH_CATALOGS function searches for information from databases listed above
def search_catalogs(*args):

    n, catalog_name, position, radius = args[0], args[1], args[2], args[3] #length of args = 4
    flag      = 0  # flag = line number of first non-zero row returned from query
    str_table = [] # str_table = list to store queries if they are strings
    f         = open('exceptions_stdout.txt', 'w')
    writeflag = 0

    # find the first non-zero row of information to begin the table (cc) for the catalog
    for i in xrange(n):
        try:
            with REMOTE_TIMEOUT.set_temp(30):
                cc = catalog_name.query_region(position[i], radius=radius)
            if len(cc) == 0:
                continue
            else:
                cc_table = cc
                if type(cc) is str: #append to a list if cc_table is a string
                    str_table.append(cc_table)
                flag = i 
                print "flag = ", str(flag) 
                break
        except Exception:
            print "An exception has occurred! :("
            print "Check out the README or comments above for reasons."
            print "A simple solution is to try re-running."
            break
        
    # if table is empty, exit; otherwise, grow table     
    if flag == 0 and len(cc) == 0:
        print "\n"
        print "No information available from ", str(catalog_name)
        print "The table for this database will be empty."
        writeflag = -1
        return (writeflag, None)
    else:
        for j in xrange(flag+1,n):
            try:
                with REMOTE_TIMEOUT.set_temp(30):
                    cc_also = catalog_name.query_region(position[j], radius=radius)
                    if j%5 == 0: print "j = ", str(j) #track progress
                if len(cc_also) == 0:
                    continue
                else:
                    if type(cc_table) is str:
                        cc_table.append(str(cc_also)) #return a list of results if the output is a string
                        writeflag = 1
                    else:
                        cc_table.add_row(cc_also[0])
            except Exception as e:
                exception_msg = "j (row number in original data file) = " + str(j) + ":\t" + str(e) + "\n"
                f.write(exception_msg) #write exceptions to a text file
                pass
        return (writeflag, cc_table)

    f.close()


def main():
	
    # read in data & convert posititon to arcdeg/min/sec
    objects   = pd.read_table('Variables_var', header=None, names=['obj', 'ra', 'dec', 'N', 'mean', 'median', 'rms', 'median_err', 'skewness', 'chi2', 'sigma', 'dr_rms'])
    converted = coord.ICRS(ra=objects['ra'], dec=objects['dec'], unit=(u.degree, u.degree))
    converted_str = converted.to_string()

    # initialize vars
    radius   = '0d0m2s'
    #catalogs = [Ned, Simbad, Nrao, Ukidss, Vizier]
    #names    = ['NED', 'Simbad', 'NRAO', 'UKIDSS', 'VizieR']
    catalogs = [Vizier]
    names    = ['VizieR']   
    n = len(objects)
    m = len(catalogs)

    # execute search_catalogs()
    for k in xrange(m):
        print "\n"
        print "Searching ", names[k], "..." #track progress

        catalogs[k].ROW_LIMIT = 15
        writeflag, table_per_catalog = search_catalogs(n-6180, catalogs[k], converted_str, radius)

        if table_per_catalog is None:
            continue

        else:
            if writeflag == 0:
                # cc_table is a table of different datatypes
                filename = names[k] + '_information.csv'
                ascii.write(table_per_catalog, filename, format='csv')

            elif writeflag == 1:
                # cc_table is a list of strings
                filename = names[k] + '_information.txt'
                ff = open(filename, 'w')
                for info in table_per_catalog:
                    ff.write("%s\n" % info)
                ff.close()

    print "Finished searching all databases!"
    
    return 0


if __name__ == '__main__':
	main()
