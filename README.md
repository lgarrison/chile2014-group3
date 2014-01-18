chile2014-group3
================

#### FILES
  - search_objects.py
  - search\_objects\_parallel.py
  - parallelize.sh
  
SEARCH_OBJECTS.PY* performs a conesearch around an object (where the object is given by an identifier and position (RA, DEC)). The script automates retrieval of information from all catalogs in the following databases: NED, Simbad, NRAO, UKIDSS, and Vizier.

SEARCH\_OBJECTS\_PARALLEL.PY* is a slightly modified version of search\_objects.py that allows for distribution of work across multiple processors. 

PARALLELIZE.SH is the bash script used for executing search\_objects\_parallel.py.

#### TO EXECUTE
To execute search\_objects.py, from the command line:



    $ python search_objects.py <input_filename>

To execute search\_objects\_parallel.py, from the command line:



    $ ./parallelize.sh ./<input_filename>
    
*Both search\_objects.py and search\_objects\_parallel.py use astroquery, which require the DEVELOPER version of astropy. If you have pip, do 



    $ pip install git+http://github.com/astropy/astroquery.git#egg=astroquery
    
See the README from https://github.com/astropy/astroquery for alternative methods of downloading & installing astroquery.

#### INPUT DATA
The data file should have the following columns in the following order:
  - obj = object ID
  - ra = RA
  - dec = DEC
  - N = number of observations
  - mean = average of magnitudes
  - median = median of magnitudes
  - rms = rms of the photometry
  - median_err = median photometric uncertainty
  - skewness = skewness of magnitudes
  - chi2 = chi-squared statistic of magnitudes
  - sigma = variance of magnitudes
  - dr_rms = rms in the position (not very useful)

#### ASSUMPTIONS AND NOTES
  1. Data saved as a file named 'Variables_var' & is in the same directory as  script.
  2. Radius is 2 arcseconds
  3. Number of objects returned per inputted object is less than/equal to 15
  4. There are a lot of warnings that are outputted, but this is okay. Modify code to suppress if it's annoying enough.
  5. The code can be very slow (and particularly NED)! Consider working on one database at a time (lines 115-116).

#### OUTPUT
  1. A .txt or .csv file with information from queries, per database.
  2. All exceptions are written to a file 'CATALOG-NAMEexceptions_stdout.txt.'
    - urllib2.URLError = The remote file couldn't be found
    - socket.timeout   = This object couldn't be found
    - exceptions.RemoteServiceError = It took too long to connect with the database server & query the data
    - Exception: Query Failed
