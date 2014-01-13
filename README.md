chile2014-group3
================

SEARCH_OBJECTS.PY performs a conesearch around an object (where the object is given by an identifier and position (RA, DEC)). The script automates retrieval of information from all catalogs in the following databases:
  - NED
  - Simbad
  - NRAO
  - UKIDSS
  - Vizier

INPUT DATA
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

ASSUMPTIONS
  1. Data  is saved as a file named 'Variables_var'; in the same directory as  script.
  2. Radius is 5 arcseconds
  3. Number of objects returned per inputted object is less than/equal to 15 

OUTPUT

If information is retrieved, a csv file is outputted for that database
