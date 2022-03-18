"""
A set of functions for measuring rotation periods.
"""

import matplotlib.pyplot as plt
import numpy as np
from starspot import rotation_tools as rt


#########################################################################################################
####################################### Simple definitions ##############################################
#########################################################################################################
def sigma_clip(x, y, yerr):
    x, y, yerr = np.array(x), np.array(y), np.array(yerr)
    
    # Initial removal of extreme outliers.
    m = rt.sigma_clip(y, nsigma=7)
    x, y, yerr = x[m], y[m], yerr[m]

    # Remove outliers using Sav-Gol filter
    smooth, mask = rt.filter_sigma_clip(x, y)
    resids = y - smooth
    stdev = np.std(resids)
    return x[mask], y[mask], yerr[mask], stdev

def norma_one(x, y, yerr):
    #Median normalize
    mu = np.median(y)
    yn = (y / mu)
    yerrn = yerr / mu
    xn = x
    
    return xn, yn, yerrn

def concat_lk(x, y, yerr):
    xc    = np.concatenate(x)
    yc    = np.concatenate(y)
    yerrc = np.concatenate(yerr)
    
    return xc, yc, yerrc




