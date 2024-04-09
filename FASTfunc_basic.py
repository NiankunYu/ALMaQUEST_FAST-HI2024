"""
coder: Niankun Yu @ 2022.12.05  (niankunyu@bao.ac.cn)

the basic functions used in fast data process

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


def get_src(file_source_list, indgal, freq0=None):
    """
    coder: Niankun Yu @ 2022.12.05
    get the basic information of the observation of each galaxy

    Input
    --------------
    file_source_list: str, the file name, including its path, of the source list file
    indgal: int, the number of the corresponding source
    freq0: float, the frequency of the line at the rest frame. Default: 1420 MHz of the H I 21 cm emission line

    Output
    --------------
    name: str, galaxy name
    freqc: float, the central frequency of the source
    date: str, the observation date
    dateKY: str, the date of the KY file (馈源舱文件), sometimes it maybe slighlty different from date 
    inttime: int, the integration time of source on for a single cycle.

    """
    df_gal = pd.read_csv(file_source_list, delim_whitespace=1, index_col = "col1")
    name = df_gal['plateifu'][indgal]
    z = df_gal['redshift'][indgal]
    c = const.c.to('km/s').value
    # V_c0 = c*z
    if freq0 == None:
        freq0 = 1420.405752
    freqc = freq0/(1.+z)
    date = str(df_gal['date'][indgal])
    dateKY = str(df_gal['dateKY'][indgal])
    inttime = df_gal['inttime'][indgal]
    N_cycle = df_gal["cycle"][indgal]
    return name, freqc, date, dateKY, inttime, N_cycle

def get_path(file_path):
    """
    coder: Niankun Yu @ 2022.12.05
    get the file path

    Input
    --------------
    file_path: str, the name including the path of file, 

    Output
    --------------
    outpath, datapathprefix, dataKY, projectname: str, the path of output, data, KY file, and project name

    """
    pardict = {'toolspath':'tmp', 'workprefix':'tmp', 'project':'tmp'} 
    with open(file_path) as parfile:
        for line in parfile:
            name, value = line.split('=')
            value = value.strip()
            name = name.strip()
            pardict[name] = value
    outpath = pardict['outpath']
    datapathprefix = pardict['datapath']
    dataKY = pardict["dataKY"]
    projectname = pardict['projectname']
    return outpath, datapathprefix, dataKY, projectname

def get_observationSetting(beams = None, samptime=None, switchtime =None, CALon_time=None):
    """
    coder: Niankun Yu @ 2022.12.05
    return the observation settings

    Input
    --------------
    beams: list of FAST beam
    samptime, switchtime, CALon_time: float, the sampling time, the switch time, the cal on time
           default sampling time 0.1 second, the cal on is 1/10 of the sampling time, 
           the switch time between source on and source off is 30 seconds.
    """
    if beams == None:
        beams = ['M01','M14']
    if samptime==None:
        samptime = 0.100663296  #s
    if switchtime==None:
        switchtime = 30.        #s
    if CALon_time==None:
        CALon_time = 24315904 * 4e-9
    CALon_factor = CALon_time/samptime
    return beams, samptime, switchtime, CALon_time, CALon_factor

def get_strCycle(ind_cycle):
    """
    keep two digital number of the ind_cycle
    coder: Zheng Zheng @ 2020
    Edit by Niankun Yu @ 2022.12.28
    """
    if ind_cycle>=9:
        str_cycle = str(ind_cycle+1) 
    else:
        str_cycle = "0"+str(ind_cycle+1) 
    return str_cycle

def get_cycle(file_str):
    """
    coder: Niankun Yu @ 2022.12.20
    get the beam and cycle from a given string    
    """
    file_name = file_str.split("/")[-1]
    str_ind = file_name.find('_M')
    str_cycle = file_name[str_ind+1:str_ind+7]
    return str_cycle

def get_beamN(beam):
    """
    coder: Niankun Yu @ 2022.12.08
    get the int number of each individual beam

    Input
    -------
    beam: str, the beam name, such as "M01"

    Output
    -------
    beamN: the int value of the given beam

    """
    beams = {'M01':0,'M02':1,'M03':2,'M04':3,'M05':4,
             'M06':5,'M07':6,'M08':7,'M09':8,'M10':9,
             'M11':10,'M12':11,'M13':12,'M14':13,'M15':14,
             'M16':15,'M17':16,'M18':17,'M19':18}
    beamN = beams[beam]
    return beamN

def get_beamPos():
    """
    coder: Niankun Yu@ 2022.12.12
    Based on Jiang et al. 2020, and mjd2radec of Zheng Yinghui and Zheng Zheng.
    get the position of each individual beams

    Output
    -------
    beams: a dictionary, which contains beam_name, beam_offset_ra, and beam_offset_dec for all 19 beams
    """
    pi = np.pi
    bra = np.array([0.,5.74,2.88,-2.86,-5.74,-2.88,2.86,
                        11.5,8.61,5.75,0.018,-5.71,-8.6,-11.5,
                        -8.63,-5.77,-0.0181,5.73,8.61])
    bdec = np.array([0.,0.00811,-4.97,-4.98,-0.0127,4.97,4.98,
                        0.0116,-4.96,-9.93,-9.94,-9.96,-4.99,-0.03,
                        4.95,9.93,9.94,9.95,4.98])
    bang = np.arctan(bdec/bra)/pi * 180.0
    ind2 = np.where(bra<0)[0]
    ind4 = np.where((bra>0) & (bdec<0))[0]
    bang[ind2] = bang[ind2] + 180.
    bang[ind4] = bang[ind4] + 360.
    bang[0]=0.
    bang_rot = bang + 0 ############# the rotation angle is 0 for FAST onoff observation, 23.5 for drift scan.
    beamdist = np.sqrt(bra**2 + bdec**2)
    beams = {'beam_name':np.array(['M01','M02','M03','M04','M05','M06','M07',
                    'M08','M09','M10','M11','M12','M13','M14',
                    'M15','M16','M17','M18','M19']), \
                'beam_offset_ra':beamdist*np.cos(bang_rot/180.*pi),\
                'beam_offset_dec':beamdist*np.sin(bang_rot/180.*pi)}
    return beams


def cut_data(freq, fluxall, freq_min=None, freq_max=None):
    """
    coder: Niankun Yu @ 2022.12.05
    cut the frequency range to freq_min, freq_max

    Input
    -------
    freq: 1d array of the observed frequency.
    fluxall: 3d array of the observed data. 

    Output
    -------
    freq_cut: 1d array of the observed frequency.
    fluxall_cut: 3d array of the observed data. 
    """
    if freq_min==None:
        freq_min = np.median(freq)-50
    if freq_max==None:
        freq_max = np.median(freq)+50
    indfreq = np.where((freq>=freq_min)&(freq<=freq_max))[0]
    freq_cut = freq[indfreq]
    fluxall_cut = fluxall[:, indfreq, :]
    return freq_cut, fluxall_cut

def radec_deg2hmsdms(ra_deg, dec_deg):
    """
    convert ra dec in unit of deg to h m s, d m s
  
    Input
    ----------
    ra_deg: float, 
    dec_deg: float.

    Output
    ----------
    ra_hms
    dec_dms

    Coder: Niankun Yu @ 2022.09.19
    """
    sexa = pyasl.coordsDegToSexa(ra_deg, dec_deg, fmt=('%02d %02d %4.1f  ', '%s%02d %02d %04.1f'))
    # print(sexa, sexa[0:10], sexa[11:])
    ra_hms = sexa[0:10]
    dec_dms = sexa[11:]
    return ra_hms, dec_dms

def MaskData_Interpolate2d(data_xx, interp_method = None):
    """
    remove the nan value in the data, and interpolate to derive a new one
    For 2d array

    Input
    --------
    data_xx: 2d array, such as the data of xx polarization
    interp_method: str, the interpolation way used to fill in the nan value.
         such as "nearest", "linear"

    Output
    --------
    data_xx_maArr_new: 2d array, after removing the nan value and interpolate

    Coder: Niankun Yu @ 2022.10.19
    """
    from scipy import interpolate
    if interp_method == None:
        interp_method = 'nearest'
    tLen, fLen = np.shape(data_xx)
    data_xx_maArr = np.ma.array(data_xx, mask = np.isnan(data_xx))
    interp_xx = np.arange(0, fLen)
    interp_yy = np.arange(0, tLen)
    itp_xx, itp_yy = np.meshgrid(interp_xx, interp_yy)
    itp_x1 = itp_xx[~data_xx_maArr.mask]
    itp_y1 = itp_yy[~data_xx_maArr.mask]
    data_xx_new = data_xx_maArr[~data_xx_maArr.mask]
    # data_xx_maArr_new = interpolate.griddata((itp_x1, itp_y1), data_xx_new.ravel(), (itp_xx, itp_yy), method='linear')
    data_xx_maArr_new = interpolate.griddata((itp_x1, itp_y1), data_xx_new.ravel(), (itp_xx, itp_yy), method = interp_method)
    return data_xx_maArr_new

def MaskData_Interpolate3d(data, interp_method = None):
    """
    remove the nan value in the data, and interpolate to derive a new one
    For 3d array

    Input
    --------
    data: 3d array, such as the data of xx and yy polarization

    Output
    --------
    data_new: 2d array, after removing the nan value and interpolate

    Coder: Niankun Yu @ 2022.10.19
    """
    from copy import copy
    tLen, fLen, sLen = np.shape(data)
    data_xx = data[:, :, 0:1]
    data_xx_rs = data_xx.reshape(tLen, fLen)
    data_xx_maArr_new = MaskData_Interpolate2d(data_xx_rs, interp_method)
    data_yy = data[:, :, 1:2]
    data_yy_rs = data_yy.reshape(tLen, fLen)
    data_yy_maArr_new = MaskData_Interpolate2d(data_yy_rs, interp_method)
    data_new = data.copy()
    data_new[:, :, 0:1] = data_xx_maArr_new.reshape(tLen, fLen, 1)
    data_new[:, :, 1:2] = data_yy_maArr_new.reshape(tLen, fLen, 1)
    return data_new 

def get_p1p2Avg(data_on, data_off):
    """
    get the average value of xx and yy polarization
    Coder: Niankun Yu @ 2022.12.08

    Input
    --------    
    data_on: 2d array, data of source on with the last dimension showing the data of XX and YY polarization
    data_off: 2d array, data of source off with the last dimension showing the data of XX and YY polarization

    Output
    --------
    onoff_avg: 1d array, the average value of source on - source off

    """
    data_on_xx = data_on[:, 0]
    data_on_yy = data_on[:, 1]
    data_off_xx = data_off[:, 0]
    data_off_yy = data_off[:, 1]
    onoff_xx = data_on_xx - data_off_xx
    onoff_yy = data_on_yy - data_off_yy
    onoff_avg = (onoff_xx+onoff_yy)/2.0
    return onoff_xx, onoff_yy, onoff_avg

def get_Vdoppler(ra, dec, mjd):
    """
    get the term for Doppler correction

    Input
    ----------
    ra, dec: float, ra, dec of the FAST observation
    mjd: array, the observation time

    Output
    ----------
    vh_med: float, median velocity of Doppler correction (V' = V+vh_med)

    Coder: Pei Zuo
    Edit by Niankun Yu @ 2022.10.19
    """
    from PyAstronomy import pyasl
    from astropy import constants as const
    ################ the geographic position of FAST
    dawodangelong = 106.8566667    # East positive, deg
    dawodangnlat  = 25.65294444    # North positive, deg
    altitude      = 1110.0288      # Altitude, m 
    if ra.shape[0] == dec.shape[0] == mjd.shape[0]:
        nin = ra.shape[0]
        porbh = np.zeros(nin, dtype = np.double)
        porbhjd = np.zeros(nin, dtype = np.double)
        for nr in range(nin):
            ############## the input should Julian Date
            vh, hjd = pyasl.helcorr(dawodangelong, dawodangnlat, altitude,\
                                    ra[nr], dec[nr], mjd[nr]+2.4e6)
            porbh[nr] = vh
            porbhjd[nr] = hjd  
    ######### because the source on time is short (<= 5mins), we take the median value of the Doppler correction
    ######### by V' = V+vh_med
    vh_med = np.median(porbh)
    return vh_med

def get_bic(y_data, k, y_true=None):
    """
    calculate the BIC of a given fitting
    coder: Niankun Yu @ 2022.12.28

    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    https://stackoverflow.com/questions/47442102/how-to-find-the-best-degree-of-polynomials
    k = number of variables in the model
    n = number of observations
    sse = sum(residuals**2)
    BIC = n*ln(sse/n) + k*ln(n) 
    """
    n = len(y_data)
    if np.any(y_true)==None:
        y_true = np.asarray([np.nanmedian(y_data)]*len(y_data))
    sse = 0
    for i in range(len(y_data)):
        sse = sse+(y_data[i]-y_true[i])**2
    bic = n*np.log(sse/n)+k*np.log(n)
    return bic

def get_Freq2Vel(freq, vh_med=None):
    """
    convert the frequency in unit of MHz to velocity in units of km/s
    Meanwhile, we do the Doppler correction.
    
    Input
    ----------
    ra, dec: float, ra, dec of the FAST observation
    mjd: array, the observation time

    Output
    ----------
    vel_radio, vel_optical, vel_cor: velocity array,
          radio velocity, optical velocity, optical velocity after Doppler correction

    Coder: Pei Zuo
    Edit by Niankun Yu @ 2022.10.18
    """
    if vh_med==None:
        vh_med = 0
        print("We do not perform Doppler correction so far.")
    ######### convert frequency to velocity, in radio frame
    c_kms = const.c.to('km/s').value  
    freq0 = 1420.405751
    vel_radio = c_kms*(freq0-freq)/freq0
    ######### derive the velocity in optical frame
    vel_optical = c_kms*(freq0-freq)/freq
    ######### derive the velocity after Doppler correction      
    vel_cor = vel_optical + vh_med  
    return vel_radio, vel_optical, vel_cor

def plot_tempTime(obsdata, file_fig):
    """
    coder: Niankun Yu @ 2022.12.05
    plot the median value vs time array for the input data

    obsdata: 3d array
    file_fig: str, the path and name of the figure file
    """
    plt.close()
    mpl.rcParams['figure.subplot.top']=0.98 
    mpl.rcParams['figure.subplot.bottom']=0.1
    mpl.rcParams['figure.subplot.left']=0.05
    mpl.rcParams['figure.subplot.right']=0.98
    mpl.rcParams['figure.subplot.hspace']=0.2
    mpl.rcParams['figure.subplot.wspace']=0.2
    fig = plt.figure(figsize=[6, 6])
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(obsdata['data'].shape[0]), np.median(obsdata['data'][:,:,0],axis=1),'r.')
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', length=6, width=3., direction="in", labelsize=14)
    ax1.tick_params(axis="both", which="minor", length=3, width=2, direction="in")
    ax1.set_xlabel(r'Time', fontsize=18)
    ax1.set_ylabel(r'Data_Median', fontsize=18)
    plt.savefig(file_fig, bbox_inches = "tight")
    plt.close()

def plot_waterfall_2pols(srcName, freq, data, pdf_path = None, freq_cen=None,\
    freq_min = None, freq_max = None, vmin_xx = None, vmax_xx = None, vmin_yy = None, vmax_yy = None):
    """
    plot the waterfall figure for the first two polarizations (XX & YY)

    Input:
    -------------
    srcName: str, source name
    freq: 1d array, frequency 
    data: 3d array, data array -- [time, frequency, polarization]
    pdf_path: str, the path we want to save the figure, default: ./
    freq_cen: float, the central frequency of the potential source, default: 1350 MHz
    freq_min: float, the minimum value of showed frequency
    freq_max: float, the maximum value of showed frequency
    vmin_xx: float, the minimum value of the imshow function for XX polarization
    vmax_xx = None, vmin_yy = None, vmax_yy

    Output:
    -------------
    None
    save the figure

    contributor: Yu Niankun @ 2022.07
    """
    
    if pdf_path == None:
        pdf_path = "./"
    if freq_cen == None:
        freq_cen = 1350
    if freq_min == None:
        freq_min = min(freq)
    if freq_max == None:
        freq_max = max(freq)
    tLen, fLen, sLen = np.shape(data)
    data_xx = data[:, :, 0:1]
    data_yy = data[:, :, 1:2]
    ######### reshape the data into two dimensional 
    data_xx_rs = data_xx.reshape(tLen, fLen)
    data_yy_rs = data_yy.reshape(tLen, fLen)
    if (vmin_xx == None) or (vmax_xx == None) or (vmin_yy == None) or (vmax_yy == None):
        ######### calculate the median value and scatter of data for each polarization
        med_xx = np.nanmedian(data_xx_rs)
        scatter_xx = np.nanstd(data_xx_rs)
        med_yy = np.nanmedian(data_yy_rs)
        scatter_yy = np.nanstd(data_yy_rs)
        if vmin_xx == None:
            vmin_xx = med_xx-0.5*scatter_xx
        if vmax_xx == None:
            vmax_xx = med_xx+2*scatter_xx
        if vmin_yy == None:
            vmin_yy = med_yy-0.5*scatter_yy
        if vmax_yy == None:
            vmax_yy = med_yy+2*scatter_yy
    extent = [freq_min, freq_max, tLen, 0]
        
    plt.close()
    mpl.rcParams['figure.subplot.top']=0.98 
    mpl.rcParams['figure.subplot.bottom']=0.1
    mpl.rcParams['figure.subplot.left']=0.05
    mpl.rcParams['figure.subplot.right']=0.98
    mpl.rcParams['figure.subplot.hspace']=0.2
    mpl.rcParams['figure.subplot.wspace']=0.0
    fig = plt.figure(figsize=[8, 16])
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    ax1_img = ax1.imshow(data_xx_rs, aspect='auto', cmap='coolwarm', extent=extent, \
                vmin = vmin_xx, vmax = vmax_xx)
    ax1.vlines(freq_cen, 0, tLen, color="black", linestyle="-", linewidth=2, zorder=1)
    ax1.set_xlabel("Frequency (MHz)", fontsize=16)
    ax1.set_ylabel("Second (0.1 s)", fontsize=16)
    ax1.set_title(srcName+'XX', fontsize=20)
    fig.colorbar(ax1_img, ax=ax1)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', length=10, width=3., direction="in", labelsize=16)
    ax1.tick_params(axis="both", which="minor", length=5, width=1, direction="in", labelsize=16)
        
    ax2_img = ax2.imshow(data_yy_rs, aspect='auto', cmap='coolwarm', extent=extent, \
        vmin = vmin_yy, vmax = vmax_yy)
    ax2.vlines(freq_cen, 0, tLen, color="black", linestyle="-", linewidth=2, zorder=1)
    ax2.set_xlabel("Frequency (MHz)", fontsize=16)
    ax2.set_ylabel("Second (0.1 s)", fontsize=16)
    ax2.set_title(srcName+'YY', fontsize=20)
    fig.colorbar(ax2_img, ax=ax2)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='major', length=10, width=3., direction="in", labelsize=16)
    ax2.tick_params(axis="both", which="minor", length=5, width=1, direction="in", labelsize=16)

    plt.savefig(pdf_path+srcName+"_waterfall.pdf", bbox_inches="tight")
    plt.close()

def plot_TempFlux(pd_cal, plot_fig, freqc=None):
    """
    coder: Niankun Yu @ 2022.12.13
    plot the spectrum of temperature, flux as a function of frequency and velocity

    """
    freq = pd_cal["frequency"]
    temp_xx = pd_cal["temp_xx"]
    temp_yy = pd_cal["temp_yy"]
    temp = pd_cal["temp"]
    flux_xx = pd_cal["flux_xx"]
    flux_yy = pd_cal["flux_yy"]
    flux = pd_cal["flux"]
    V_radio = pd_cal["V_radio"]
    V_opt = pd_cal["V_opt"]
    V_optDop = pd_cal["velocity"]

    plt.close()
    mpl.rcParams['figure.subplot.top']=0.98 
    mpl.rcParams['figure.subplot.bottom']=0.1
    mpl.rcParams['figure.subplot.left']=0.05
    mpl.rcParams['figure.subplot.right']=0.98
    mpl.rcParams['figure.subplot.hspace']=0.1
    mpl.rcParams['figure.subplot.wspace']=0.2
    fig = plt.figure(figsize=[16, 16])
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    ax1.step(freq, temp_xx, where = "mid", color="black", linestyle = "-", label = "XX")
    ax1.step(freq, temp_yy, where = "mid", color="blue", linestyle = "-", label = "YY")
    ax1.step(freq, temp, where = "mid", color="red", linestyle = "-", label = "(XX+YY)/2")
    ax1.legend(loc="upper left")

    ax2.step(freq, flux_xx, where = "mid", color="black", linestyle = "-", label = "XX")
    ax2.step(freq, flux_yy, where = "mid", color="blue", linestyle = "-", label = "YY")
    ax2.step(freq, flux, where = "mid", color="red", linestyle = "-", label = "(XX+YY)/2")

    ax3.step(V_radio, temp, where = "mid", color="black", linestyle = "-", label = "Radio")
    ax3.step(V_opt, temp, where = "mid", color="blue", linestyle = "-", label = "Optical")
    ax3.step(V_optDop, temp, where = "mid", color="red", linestyle = "-", label = "Optical+Doppler")
    ax3.legend(loc="upper left")

    ax4.step(V_radio, flux, where = "mid", color="black", linestyle = "-", label = "Radio")
    ax4.step(V_opt, flux, where = "mid", color="blue", linestyle = "-", label = "Optical")
    ax4.step(V_optDop, flux, where = "mid", color="red", linestyle = "-", label = "Optical+Doppler")

    ax_list = [ax1, ax2, ax3, ax4]
    xlabel_list = [r"Freq (MHz)", r"Freq (MHz)", r"Vel (km s$^{-1}$)", r"Vel (km s$^{-1}$)"]
    ylabel_list = [r"Temp (K)", r"Flux (mJy)", r"Temp (K)", r"Flux (mJy)"]
    for i in range(len(ax_list)):
        p = ax_list[i]
        p.minorticks_on()
        p.tick_params(axis='both', which='major', length=10, width=3., direction="in", labelsize=16)
        p.tick_params(axis="both", which="minor", length=5, width=1, direction="in", labelsize=16)
        p.set_xlabel(xlabel_list[i], fontsize=16)
        p.set_ylabel(ylabel_list[i], fontsize=16)

    ######### show the signal range
    if freqc != None:
        temp_med = np.nanmedian(temp)
        temp_std = np.nanstd(temp)
        flux_med = np.nanmedian(flux)
        flux_std = np.nanstd(flux)
        ############# 
        vel_radio, vel_optical, vel_cor =  get_Freq2Vel(freqc)
        ax1.vlines(freqc, temp_med-5*temp_std, temp_med+5*temp_std, color="green", zorder=2, linewidth=3)
        ax2.vlines(freqc, flux_med-5*flux_std, flux_med+5*flux_std, color="green", zorder=2, linewidth=3)
        ax3.vlines(vel_optical, temp_med-5*temp_std, temp_med+5*temp_std, color="green", zorder=2, linewidth=3)
        ax4.vlines(vel_optical, flux_med-5*flux_std, flux_med+5*flux_std, color="green", zorder=2, linewidth=3)

    plt.savefig(plot_fig, bbox_inches="tight")
    plt.close()

    
def write_fits1(freq, flux_on, flux_off, mjd_on, mjd_off, fits_file):
    """
    coder: Niankun Yu @ 2022.12.06
    write the frequency, data of on and off, mjd of on and off into one fits file

    Input
    -------
    freq: 1d array of the frequency with a range of freq_min, freq_max.
    flux_on: 3d array of the observed data for source on, same frequency range as freq
    flux_off: 3d array of the observed data for source off, same frequency range as freq
    mjd_on: 1d array of mjd for source on
    mjd_off: 1d array of mjd for source off
    fits_file: str, the fits file we want to write the data in, including its path

    No output
    """
    outhdr = fits.Header()
    outhdu1=fits.PrimaryHDU(data=freq,header=outhdr)
    outhdu1.header['extname']='freq'
    outhdu2=fits.ImageHDU(data=flux_on,name='ON_flux')
    outhdu3=fits.ImageHDU(data=flux_off,name='OFF_flux')
    outhdu4=fits.ImageHDU(data=mjd_on,name='ON_mjd')
    outhdu5=fits.ImageHDU(data=mjd_off,name='OFF_mjd')
    hdulist=fits.HDUList([outhdu1,outhdu2,outhdu3,outhdu4,outhdu5])
    hdulist.writeto(fits_file, overwrite=True)

def read_fits1(fits_file):
    """
    coder: Niankun Yu @ 2022.12.08
    read the onoff fits file written by write_fits1

    Input
    -------
    fits_file: str, the fits file we want to write the data in, including its path

    Output
    -------
    freq: 1d array of the frequency with a range of freq_min, freq_max.
    flux_on: 3d array of the observed data for source on, same frequency range as freq
    flux_off: 3d array of the observed data for source off, same frequency range as freq
    mjd_on: 1d array of mjd for source on
    mjd_off: 1d array of mjd for source off

    """
    with fits.open(fits_file) as hdu:
        freq = hdu[0].data
        flux_on = hdu[1].data
        flux_off = hdu[2].data
        mjd_on = hdu[3].data
        mjd_off = hdu[4].data
        hdu.close()
    return freq, flux_on, flux_off, mjd_on, mjd_off

def write_fits2(freq, Temp_on, Temp_off, mjd_on, mjd_off, fits_file):
    """
    coder: Niankun Yu @ 2022.12.06
    write the frequency, data of on and off, mjd of on and off into one fits file

    Input
    -------
    freq: 1d array of the frequency with a range of freq_min, freq_max.
    flux_on: 3d array of the observed data for source on, same frequency range as freq
    flux_off: 3d array of the observed data for source off, same frequency range as freq
    mjd_on: 1d array of mjd for source on
    mjd_off: 1d array of mjd for source off
    fits_file: str, the fits file we want to write the data in, including its path

    No output
    """
    outhdr = fits.Header()
    outhdu1=fits.PrimaryHDU(data=freq,header=outhdr)
    outhdu1.header['extname']='freq'
    outhdu2=fits.ImageHDU(data=Temp_on,name='ON_temp')
    outhdu3=fits.ImageHDU(data=Temp_off,name='OFF_temp')
    outhdu4=fits.ImageHDU(data=mjd_on,name='ON_mjd')
    outhdu5=fits.ImageHDU(data=mjd_off,name='OFF_mjd')
    hdulist=fits.HDUList([outhdu1,outhdu2,outhdu3,outhdu4,outhdu5])
    hdulist.writeto(fits_file, overwrite=True)

def read_fits2(fits_file):
    """
    read the temperature fits file written by write_fits2

    Input
    -------
    fits_file: str, the fits file we want to write the data in, including its path

    Output
    -------
    freq: 1d array of the frequency with a range of freq_min, freq_max.
    flux_on: 3d array of the observed data for source on, same frequency range as freq
    flux_off: 3d array of the observed data for source off, same frequency range as freq
    mjd_on: 1d array of mjd for source on
    mjd_off: 1d array of mjd for source off
    """
    with fits.open(fits_file) as hdu:
        freq = hdu[0].data
        temp_on = hdu[1].data
        temp_off = hdu[2].data
        mjd_on = hdu[3].data
        mjd_off = hdu[4].data
        hdu.close()
    return freq, temp_on, temp_off, mjd_on, mjd_off

def write_txtCal(freq, on_temp, off_temp, on_flux, off_flux, vel_radio, vel_optical, vel_cor, plot_fig = "", write_txt = "", freqc=None):
    """
    write the on, off temperature (flux into the files)

    Input
    -------
    fits_file: str, the fits file we want to write the data in, including its path

    Output
    -------
    pd_cal: the dataframe containing the frequecy, temperature (K), flux (mJy), velocity (km/s)

    """
    onoff_temp_xx, onoff_temp_yy, onoff_temp = get_p1p2Avg(on_temp, off_temp)
    onoff_flux_xx, onoff_flux_yy, onoff_flux = get_p1p2Avg(on_flux, off_flux)
    ####### build the dataframe
    pd_cal = pd.DataFrame({"frequency":freq, "temp_xx": onoff_temp_xx, "temp_yy": onoff_temp_yy, "temp": onoff_temp,\
        "flux_xx": onoff_flux_xx, "flux_yy": onoff_flux_yy, "flux": onoff_flux,\
            "V_radio": vel_radio, "V_opt": vel_optical, "velocity": vel_cor})
    if len(plot_fig)>0:
        plot_TempFlux(pd_cal, plot_fig, freqc)
    if len(write_txt)>0:
        pd_cal.to_csv(write_txt, sep=",")
    return pd_cal

def read_txtCal(file_txt):
    """
    read the spectra from the function of write_txtCal
    coder: Niankun Yu @ 2022.12.13

    Input
    -------
    fits_file: str, the fits file we want to write the data in, including its path

    Output
    -------
    """
    dd0_names = ("index", "frequency", "temp_xx", "temp_yy", "temp", "flux_xx", "flux_yy", "flux", "V_radio", "V_opt", "velocity")
    dd0 = pd.read_csv(file_txt, names = dd0_names, skiprows = 1, sep = ",")
    frequency = dd0["frequency"]
    velocity = dd0["velocity"]
    flux  = dd0["flux"]
    dd1 = pd.DataFrame({"frequency":frequency, "velocity": velocity, "flux":flux, "weight": np.asarray([1.0]*len(flux))})
    dd1 = dd1.sort_values(by=['velocity'], ascending=True).reset_index(drop=True)
    return dd1


    

