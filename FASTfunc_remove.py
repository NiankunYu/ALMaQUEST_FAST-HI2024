"""
manual setting the mask range, and remove bad cycles
coder: Niankun Yu @ 2022.12.27   (niankunyu@bao.ac.cn)
"""
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from PyAstronomy import pyasl
import sys, glob, os, time, io
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from copy import copy

def get_mask(name, year, N_cycle = None):
    """
    get the mask range of a give object,
    coder: Niankun Yu
    """
    if (name =="7815-12705") & (year != "2022"):
        vel_mask_arr = np.asarray([[[7650, 9500], [7900, 9850]],\
        [[7700, 9450], [8000, 9900]],\
        [[9400, ], [9850, ]],\
        [[7750, 9450], [7950, 9900]],\
        [[7700, 9450], [8000, 9900]],\
        [[9400, ], [9850, ]]], dtype=object)
    else:
        assert N_cycle is not None
        vel_mask_arr = np.empty([N_cycle*2, 2], dtype=object)

    return vel_mask_arr

def get_stringRemove(name, year):
    """
    get the string we want to remove
    coder: Niankun Yu @ 2022.12.27
    """
    if (name =="7815-12705") & (year != "2022"):
        stringRemove = ""
    # elif (name =="8728-3701") & (year == "2022"):
    #     stringRemove = ["_M01_04", "_M14_04",]
    else:
        stringRemove = ""
    return stringRemove

def get_cycle(sp_files, vel_mask_arr, stringRemove = ""):
    """
    get the cycle we want to remove from the given string
    coder: Niankun Yu @ 2022.12.27
    """
    if stringRemove == "":
        return sp_files, vel_mask_arr
    else:
        sp_files2 = sp_files.copy()
        for i in range(len(stringRemove)):
            sp_files = [ x for x in sp_files if stringRemove[i] not in x ]
        dropIdx = []
        for i in range(len(stringRemove)):
            for j in range(len(sp_files2)):
                if stringRemove[i] in sp_files2[j]:
                    dropIdx.extend([j])
        vel_mask_new = np.delete(vel_mask_arr, dropIdx, 0)
        return sp_files, vel_mask_new