"""
coder: Niankun Yu @ 0222.12.05  (niankunyu@bao.ac.cn)
Based on the method of Zheng Zheng @ 2020 
(crafts:/home/zz/python/onoff/; crafts:/home/zz/python/tools/,
crafts:/home/zz/python/tools/read_onoff_tools.py)

separate the data of source on and source off, cal on and cal off
"""

import glob,sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import medfilt
import pandas as pd
from PyAstronomy import pyasl
from astropy import constants as const
# from apply_cal8b import apply_cal
# from FASTfunc_calibration import apply_cal
import FASTfunc_basic as fast_basic


def read_onoff_files(path, beam, plot_fig):
    """
    coder: Zheng Zheng @ 2020
    Edit by Niankun Yu @ 2022, add the mjd and modify the figure plotting
    read the onoff files of the same beam under one file directory, and give its frequency, mjd, and flux
    Meanwhile, plot the data of each fits file (median data vs time array)
    combine the function of Pei Zuo and Zheng Zheng (fast_onoff_tools.read_onoff_files). Rewrite by Niankun Yu @ 2022.09.16

    Input
    -------
    path: str, path of the fits file
    beam: str, the beam name
    plot_fig: str, the figure name we want to save

    Output
    -------
    freq: 1d array, the frequency array
    fluxall: 
    mjdall: 1d array, the mjd value
    """
    files = np.sort(glob.glob(path+'*_onoff-'+beam+'_W_*.fits'))
    if len(files)==0:
        print('no files!')
        return [],[]
    print('Reading files:')
    [print(i) for i in files]

    for indf, file in enumerate(files):
        print(indf+1, '/',len(files), file)
        with fits.open(file) as hdu:
            obsdata = hdu[1].data
        if indf==0:
            chanbw=obsdata['chan_bw'][0]
            nchan=obsdata['nchan'][0]
            freq0=obsdata['freq'][0]
            freq=np.arange(nchan)*chanbw+freq0
            mjdall = obsdata['utobs']
            fluxall=obsdata['data'][:,:,0:2]
        else:
            flux = obsdata['data'][:,:,0:2]
            fluxall = np.concatenate((fluxall,flux),0)
            mjd = obsdata['utobs']
            mjdall = np.concatenate((mjdall,mjd),0)
        if len(plot_fig) > 0:
            str_cycle = fast_basic.get_strCycle(indf)
            fast_basic.plot_tempTime(obsdata, plot_fig+'_'+str_cycle+'_temp_vs_time.png')
    return freq, fluxall, mjdall

def separate_onoff(freq, fluxall, mjdall, beam, samptime, switchtime, inttime, plot_fig = "", write_fits = ""):
    """
    Coder:Zheng Zheng @ 2020
    Edit by Niankun Yu @ 2022, 
    cut the frequencu range,
    seperate the data of source on and source off,
    Inverse results of source on and source off for beam M01 and other beams (such as M14)

    Input
    -------
    freq: 1d array of the observed frequency.
    fluxall: 3d array of the observed data. 
    mjdall: 1d array of the observation time. 
    beam: str, FAST beam, such as "M01"
    samptime, switchtime: float, the sampling time, the switch time
    inttime: int, the integration time of source on for one cycle
    freq_min, freq_max: flaot, the minimum and maximum frequency 
    plot_fig: str, plot the figure or not. If len(plot_fig)>0, write
    write_fits: str, write the fits file or not. If len(write_fits)>0, write
    CALon_factor: float, CALon_factor = CALon_time/samptime --- the ratio of cal on and sampling time

    Output
    -------
    freq: 1d array of the frequency with a range of freq_min, freq_max.
    flux_on: 3d array of the observed data for source on, same frequency range as freq
    flux_off: 3d array of the observed data for source off, same frequency range as freq
    mjd_on: 1d array of mjd for source on
    mjd_off: 1d array of mjd for source off
    """
    ncycle = round((fluxall.shape[0]*samptime+switchtime)/(inttime+switchtime)/2)
    ttime = np.arange(fluxall.shape[0])*samptime
    for indcycle in range(ncycle):
        print('cycle #',indcycle+1,'/',ncycle)
        str_cycle = fast_basic.get_strCycle(indcycle)
        ############# get the index of source on and source off
        indon=np.where((ttime>indcycle*2.*(inttime+switchtime)) & (ttime<indcycle*2.*(inttime+switchtime)+inttime))[0]
        indoff=np.where((ttime>(indcycle*2.+1.)*(inttime+switchtime)) & (ttime<(indcycle*2.+1.)*(inttime+switchtime)+inttime))[0]
        if beam == 'M01':
            flux_on_tmp = fluxall[indon,:,:]
            flux_off_tmp = fluxall[indoff,:,:]
            mjd_on_tmp = mjdall[indon]
            mjd_off_tmp = mjdall[indoff]
        else:
            flux_on_tmp = fluxall[indoff,:,:]
            flux_off_tmp = fluxall[indon,:,:]
            mjd_on_tmp = mjdall[indoff]
            mjd_off_tmp = mjdall[indon]
        
        if indcycle == 0:
            flux_on = flux_on_tmp
            flux_off = flux_off_tmp
            mjd_on = mjd_on_tmp
            mjd_off = mjd_off_tmp
        else:
            flux_on = np.concatenate((flux_on, flux_on_tmp),axis=0)
            flux_off = np.concatenate((flux_off, flux_off_tmp),axis=0)
            mjd_on = np.concatenate((mjd_on, mjd_on_tmp),axis=0)
            mjd_off = np.concatenate((mjd_off, mjd_off_tmp),axis=0)
        if len(plot_fig) > 0:
            pdf_path = "/".join(plot_fig.split("/")[0:-1])+"/"
            srcName_on = plot_fig.split("/")[-1]+'_'+str_cycle+"_on"
            srcName_off = plot_fig.split("/")[-1]+'_'+str_cycle+"_off"
            fast_basic.plot_waterfall_2pols(srcName_on, freq, flux_on_tmp, pdf_path = pdf_path)
            fast_basic.plot_waterfall_2pols(srcName_off, freq, flux_off_tmp, pdf_path = pdf_path)
        if len(write_fits) > 0:
            fits_file = write_fits+'_'+str_cycle+"_onoff.fits"
            fast_basic.write_fits1(freq, flux_on_tmp, flux_off_tmp, mjd_on_tmp, mjd_off_tmp, fits_file)
    if len(write_fits) > 0:
        fits_file_all = write_fits+"_onoffALL.fits"
        fast_basic.write_fits1(freq, flux_on, flux_off, mjd_on, mjd_off, fits_file_all)
    return freq, flux_on, flux_off, mjd_on, mjd_off

