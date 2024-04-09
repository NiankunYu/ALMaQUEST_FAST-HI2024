"""
functions to deal with RFIs
coder: Niankun Yu @ 2022.12.23   (niankunyu@bao.ac.cn)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob,sys, io, os
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import FASTfunc_basic as fast_basic

def flag_rfi(tmpflux, n_sigma=None):
    """
    set the values beyond +-1.5*std as np.nan
    Coder: Zheng Zheng @ 2020
    Edit by Niankun Yu @ 2022.12.07

    Input
    -------
    tmpflux: temperature array,

    Output
    -------
    tmpflux: temperature array after setting values beyond +-1.5*std as np.nan
    """
    if n_sigma==None:
        n_sigma=1.5
    medflux = np.median(tmpflux)
    stdflux = np.std(tmpflux)
    tmpflux[tmpflux > medflux+n_sigma*stdflux] = np.nan
    tmpflux[tmpflux < medflux-n_sigma*stdflux] = np.nan
    return tmpflux

def flag_rfiTime(flux0, freq, freqc):
    """
    coder: Zheng Zheng @ 2020
    Edit by Niankun Yu @ 2022.12.23

    remove the bad time sequency through the median value of a given time sequence
    """
    index_freq = np.where((freq>freqc-2) & (freq<freqc+2))[0]
    print("The index of frequency is:", index_freq)
    flux_mean = np.mean(flux0[:, index_freq, :], axis= 1)
    flux_med = np.median(flux_mean, axis = 0)
    flux_std = np.std(flux_mean, axis = 0)
    flux_delta = np.abs(flux_mean - flux_med)
    index_good = np.where((flux_delta[:, 0]<2.0*flux_std[0]) & (flux_delta[:, 1]<2.0*flux_std[1]))[0]
    flux = flux0[index_good, :, :]
    return flux, index_good



