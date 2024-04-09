"""
Coder: Niankun Yu @ 2022.12.05  (niankunyu@bao.ac.cn)


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
import FASTfunc_rfi as fast_rfi

def get_calonoff_ind(fluxall):
    """
    Coder: Niankun Yu @ 2022.12.06
    seperate the data of cal on and cal off,
    plot the waterfall figure after separating the cal on and cal off

    Input
    -------
    fluxall: 3d array of the observed data. 
    mjdall: 1d array of the observation time. 

    Output
    -------
    flux_calon: 3d array of the data for noise diode on (cal on)
    flux_caloff: 3d array of the data for noise diode off (cal off)
    mjd_calon: 1d array of the observation time for noise diode on (cal on)
    mjd_caloff: 1d array of the observation time for noise diode off (cal off)
    """
    flux0 = np.sum(fluxall,axis=1)[:,0]
    flux0_smooth = medfilt(flux0, 9)
    delta_flux0 = flux0 - flux0_smooth
    flux0_std = np.std(delta_flux0)
    ind_calon = np.where(delta_flux0 > 2*flux0_std)[0]
    ind_caloff = np.where((delta_flux0 < 1*flux0_std)&(delta_flux0 > -1*flux0_std))[0]
    return ind_calon, ind_caloff

def get_fluxMjd(fluxall, mjdall, ind_calon, ind_caloff):
    """
    Coder: Niankun Yu @ 2022.12.07
    pick up the row of flux and mjd with given index of the first dimension

    Input
    -------
    fluxall: 3d array of the observed data. 
    mjdall: 1d array of the observation time. 
    ind_calon: 1d array, index of row
    ind_caloff: 1d array, index of row

    Output
    -------
    flux_calon: 3d array of the data for noise diode on (cal on)
    flux_caloff: 3d array of the data for noise diode off (cal off)
    mjd_calon: 1d array of the observation time for noise diode on (cal on)
    mjd_caloff: 1d array of the observation time for noise diode off (cal off)
    """
    flux_calon = fluxall[ind_calon, :, :]
    flux_caloff = fluxall[ind_caloff, :, :]
    mjd_calon = mjdall[ind_calon]
    mjd_caloff = mjdall[ind_caloff]
    return flux_calon, flux_caloff, mjd_calon, mjd_caloff

def get_Tcal(beam, cal_path=None, cal_mode=None, obs_mode=None):
    """
    get the temperature calibration at a given frequency
    Coder: Niankun Yu @ 2022.12.07
    Based on Zheng Zheng's code

    Input
    -------
    beam: the beam we want to calibrate
    cal_path: str, the path of the calibration file, 
    cal_mode: str, the mode of calibration
    obs_mode: str, the mode of observation

    Output
    -------
    freq_cal: 1d array of the calibration frequency, (len = 65536, min = 1000, max = 1500)
    T_cal: 1d array of the calibration temperature

    caution, before we do the temperature calibration, we should not cut the frequency array (wide-band)
    """
    if cal_path==None:
        cal_path = "/home/niankunyu/almaquest_2023/pipeline_onoff2022/"
    if cal_mode==None:
        cal_mode = "highcal"
    if obs_mode == None:
        obs_mode = 'W'
    ########## the calibration file from Liu Mengting
    cal_file = cal_path+cal_mode+'_20201014_'+obs_mode+'_tny.npz'
    tmp = np.load(cal_file)
    freq_cal = tmp['freq']
    tcals = tmp['tcal']
    beamN = fast_basic.get_beamN(beam)
    T_cal = tcals[:, :, beamN]
    return freq_cal, T_cal
    
def get_Temp(flux, beam, freq_min, CALon_factor, cal_path=None, cal_mode=None, obs_mode=None):
    """
    convert the data to temperature, by input the calibration temperature, observations of source on (cal on, cal off),
    or source off (cal on, cal off).
    Coder: Niankun Yu @ 2022.12.07

    Input
    -------
    fluxall: 3d array of the observed data. 
    beam: str, the beam we want to calibrate
    freq_min: float, the minimum value of frequency
    CALon_factor: float, the ratio of T_calon and T_sampling, CALon_factor = CALon_time/samptime, which is 0.9662 second for the almaquest sample
    cal_path: str, the path of the calibration file, 
    cal_mode: str, the mode of calibration
    obs_mode: str, the mode of observation
    

    Output
    -------
    freq_cut: 1d array of the frequency, set the minimum value as freq_min
    Temp: 3d array of the temperature, set the minimum value of the frequency dimension as freq_min
    ind_calon: 1d array, index of cal on
    ind_caloff: 1d array, index of cal off

    Caution: the frequency dimension cannot be different from that from wideband observation (len = 65536, min = 1000, max = 1500).
             This is required by the usage of the "get_Tcal" function
    """
    freq_cal, T_cal = get_Tcal(beam, cal_path, cal_mode, obs_mode)
    ########### cut the range of calibration frequency
    if freq_min>1300:
        ind_freq = np.where(((freq_cal>1050)&(freq_cal<1150))|((freq_cal>1300)&(freq_cal<1410)))[0]
    else:
        ind_freq=np.where(((freq_cal>1050)&(freq_cal<1150))|((freq_cal>freq_min)&(freq_cal<1410)))[0]
    freq_cut = freq_cal[ind_freq] 
    T_cal_cut = T_cal[ind_freq,:]
    flux_cut = flux[:,ind_freq,:]
    ind_calon, ind_caloff = get_calonoff_ind(flux_cut)
    # flux_calon, flux_caloff, mjd_calon, mjd_caloff = get_fluxMjd(flux_cut, mjd, ind_calon, ind_caloff)

    len_calon = len(ind_calon)
    ######## initiate the temperature array
    Temp = np.empty([len_calon, flux_cut.shape[1], 2])
    # calculate CAL deflection template
    flux_calon_med = np.nanmedian(flux_cut[ind_calon,:,:],axis=0)
    flux_caloff_med = np.nanmedian(flux_cut[ind_caloff,:,:],axis=0)
    flux_cal_deflection = flux_calon_med - flux_caloff_med
    flux2temp = T_cal_cut/flux_cal_deflection
    for ipol in range(2):
        flux2temp[:,ipol] = gaussian_filter1d(flux2temp[:,ipol],301)
    cal_amp = np.zeros((len_calon,2))
    flux_amp = np.zeros((len_calon,2))
    for iind,iical in enumerate(ind_calon):
        if iical == flux_cut.shape[0]-1:
            cal_amp[iind,:] = np.nan
            continue  # skip this if the CAL is the last record
        tmpwcalflux = flux_cut[iical,:,:]
        if iind < ind_calon.shape[0]-1 :
            tmpnocalflux=np.nanmedian(flux_cut[iical+1:ind_calon[iind+1],:,:],axis=0) 
        else:
            tmpnocalflux=np.nanmedian(flux_cut[ind_calon[iind-1]+1:ind_calon[iind],:,:],axis=0)
        tmpdecal=tmpwcalflux-tmpnocalflux
        flux_amp[iind,:] = np.nanmedian(tmpnocalflux,axis=0)
        for ipol in range(2):
            tmpfit10 = tmpdecal[:,ipol]
            tmpfit20 = flux_cal_deflection[:,ipol]
            tmpfit1 = fast_rfi.flag_rfi(tmpfit10)
            tmpfit2 = fast_rfi.flag_rfi(tmpfit20)
            indgood = np.where((np.isnan(tmpfit1)==0) & (np.isnan(tmpfit2)==0) )[0]
            if indgood.shape[0] > 50:
                tmpp = np.median(tmpfit2[indgood]/tmpfit1[indgood])
                Temp[iind,:,ipol] = tmpnocalflux[:,ipol] * tmpp * flux2temp[:,ipol] * CALon_factor
                cal_amp[iind,ipol] = tmpp
            else:
                Temp[iind,:,ipol] = np.nan
                cal_amp[iind,ipol] = np.nan
    return freq_cut, Temp, ind_calon, ind_caloff
    
def get_Temp_onoff(flux_on, flux_off, beam, freq_min, CALon_factor, cal_path=None, cal_mode=None, obs_mode=None):
    """
    coder: Niankun Yu @ 2022.12.08
    get the temperature and index of source on (calon, caloff) and source off (calon, caloff)

    Input
    -------
    fluxall: 3d array of the observed data. 
    beam: str, the beam we want to calibrate
    freq_min: float, the minimum value of frequency
    CALon_factor: float, the ratio of T_calon and T_sampling, CALon_factor = CALon_time/samptime, which is 0.9662 second for the almaquest sample
    cal_path: str, the path of the calibration file, 
    cal_mode: str, the mode of calibration
    obs_mode: str, the mode of observation
    

    Output
    -------
    freq_cut: 1d array of the frequency, set the minimum value as freq_min
    Temp: 3d array of the temperature, set the minimum value of the frequency dimension as freq_min
    ind_calon: 1d array, index of cal on
    ind_caloff: 1d array, index of cal off
    """
    freq_cut_on, Temp_on, ind_calon_on, ind_caloff_on = get_Temp(flux_on, beam, freq_min, CALon_factor, cal_path, cal_mode, obs_mode)
    freq_cut_off, Temp_off, ind_calon_off, ind_caloff_off = get_Temp(flux_off, beam, freq_min, CALon_factor, cal_path, cal_mode, obs_mode)
    return freq_cut_on, Temp_on, ind_calon_on, ind_caloff_on, Temp_off, ind_calon_off, ind_caloff_off

def fits1_fits2(fits_file1, freq_min, freq_max, CALon_factor, plot_fig = "", write_fits = "", cal_path=None, cal_mode=None, obs_mode=None):
    """
    coder: Niankun Yu @ 2022.12.08
    read the fits file with observation data to fits file with temperature
    """
    freq, flux_on, flux_off, mjd_on, mjd_off = fast_basic.read_fits1(fits_file1)
    str_ind = fits_file1.find('_M')
    beam = fits_file1[str_ind+1:str_ind+4]
    freq_cut_on, Temp_on, ind_calon_on, ind_caloff_on, Temp_off, ind_calon_off, ind_caloff_off = get_Temp_onoff(flux_on, flux_off, beam, freq_min, CALon_factor, cal_path, cal_mode, obs_mode)
    ind_freq = np.where((freq_cut_on>=freq_min)&(freq_cut_on<=freq_max))[0]
    freq_final = freq_cut_on[ind_freq]
    Temp_on_final = Temp_on[:, ind_freq, 0:2]
    Temp_off_final = Temp_off[:, ind_freq, 0:2]
    fits_file2 = fits_file1.replace("_onoff.fits", "_temp.fits")
    if len(plot_fig) > 0:
            pdf_path = "/".join(fits_file1.split("/")[0:-1])+"/"
            srcName_on = fits_file1.split("/")[-1].strip("_onoff.fits")+"_on_temp"
            srcName_off = fits_file1.split("/")[-1].strip("_onoff.fits")+"_off_temp"
            fast_basic.plot_waterfall_2pols(srcName_on, freq_final, Temp_on_final, pdf_path = pdf_path)
            fast_basic.plot_waterfall_2pols(srcName_off, freq_final, Temp_on_final, pdf_path = pdf_path)
    if len(write_fits) > 0:
        # basic_functions.write_fits2(freq_final, Temp_on_final, Temp_off_final, mjd_on, mjd_off, fits_file2)
        fast_basic.write_fits2(freq_final, Temp_on_final, Temp_off_final, mjd_on[ind_calon_on], mjd_off[ind_calon_off], fits_file2)
    return freq_final, Temp_on_final, ind_calon_on, ind_caloff_on, Temp_off_final, ind_calon_off, ind_caloff_off

#%%
################# convert temperature to mJy
def get_KYfile(dataKY, dateKY, source_name):
    """
    get the mjd, ra, dec information from the KY file (馈源舱文件)
    
    Input
    ----------- 
    dataKY: str, the data path of KY file, for example: "/data14/KY/"
    KY_date: str, observation date (the KY file, which may be slightly different from the raw fits file), 
             including year, month, day. For example, 20200930
    source_name: str, the source name

    Output
    -----------
    KY_file: str, the path and name of KY file

    Coder: Niankun Yu, 2022.09.16
    """
    print(dataKY, dateKY, source_name)
    from datetime import date

    # KY_file = dataKY+source_date[0:6]+"/"
    date0 = date(int(dateKY[0:4]), int(dateKY[4:6]), int(dateKY[6:8]))
    date0_ymd = date0.strftime("%Y/%m/%d").replace("/0", "/")
    date0_str = date0_ymd.split("/")
    date0_month = date0_str[-2]
    date0_day = date0_str[-1]
    path_month_day = date0_month+"."+date0_day

    KY_path = dataKY+dateKY[0:6]+"/" + path_month_day+"/"
    KY_file0 = KY_path+source_name+'_*.xlsx'
    KY_files = glob.glob(KY_file0)
    KY_files.sort()
    print(dataKY, dateKY[0:6], path_month_day, source_name)
    print(KY_files)
    if os.path.exists(KY_files[0]):
        pass
    else:
        print("Warning: no KY_file exists, please check again")
    return KY_path, KY_files

def xyz2azel(x0,y0,z0):
    """
    coder: Zheng Zheng & Zheng Yinghui, 2020-2022
    """
    R=np.sqrt(x0**2+y0**2+z0**2)
    z=-z0
    y=-x0
    x=-y0
    az0 = 0.0
    if (np.abs(x) < 1.0e-8):
        if (y>0):
            az0 = 90.0
        else:
            az0 = 270.0
    else:
        tempaz0 = np.rad2deg(np.arctan(y/x))
        if (x>0 and y>0):
            az0 = tempaz0
        if ((x>0 and y<0) or (x>0 and y==0)):
            az0 = tempaz0+360.0
        if (x<0 and y>0):
            az0 = tempaz0+180.0
        if ((x<0 and y<0) or (x<0 and y==0)):
            az0 = tempaz0+180.0
    el0 = np.rad2deg(np.arcsin(z/R))
    return az0, el0

def xyz2radec(x,y,z,mjd,type='deg'):
    '''
    coder: Zheng Zheng & Zheng Yinghui, 2020-2022
        Input:
            x,y,z: float
                Feed position record for the time of your input mjd.
            mjd: float
                observation time in mjd.
            type: string, default is 'deg'
                Should be choose in ['deg', 'str', 'both']
                Type of output.
                If 'deg': output two argments -- ra_deg, dec_deg with the form of float in unit of deg.
                If 'str': output two argments -- ra_str with the form 'hh:mm:ss', dec_str with the form 'dd:mm:ss'.
                If 'both': output four argments -- ra_deg, dec_deg, ra_str, dec_str
    '''
    az, el = xyz2azel(x,y,z)
    return localtoec(az,el,mjd,type=type)

#@jit(nopython=True, parallel=True)
def mjd2radec(mjd,beam,posfile,filetype='processed'):
    '''
    coder: Zheng Zheng & Zheng Yinghui, 2020-2022
        Change observed mjd to RA & DEC for CRAFTS data.
        
        Input:
            mjd: number or list
                Observed mjd.
            beam: number or list, numbers' value range from 1 to 19 
                Beam number from 1 to 19.
            posfile: string
                Path of CRAFTS's position file. 
                If filetype is 'processed', file shoule be fits format contains columns ['mjd','ra_deg','dec_deg'].
                If filetype is 'raw', file should be excel format contains columns ['SysTime', 'SwtDPos_X', 'SwtDPos_Y', 'SwtDPos_Z'] in sheet '测量数据'.
            filetype: string, default is 'processed'
                Type of input position file, must be 'processed' or 'raw'.
        
        Output:
            ra: array
            dec: array
    '''

    mjd, beam = np.array(mjd), np.array(beam)
    # beams = beam_offset()
    
    beams = fast_basic.get_beamPos()
    if filetype == 'raw':
        # print("The posfile is:", posfile)
        ZK = pd.read_excel(posfile, sheet_name=1, header=0, index_col = None)
        # ZK = pd.read_excel(posfile,sheet_name='测量数据')
        if np.min(mjd-BT2mjd(ZK.iloc[0]['SysTime']))>=0 and np.max(mjd-BT2mjd(ZK.iloc[-1]['SysTime']))<=0 :
            ZK.loc[:, 'mjd'] = ZK.loc[:, 'SysTime'].apply(lambda x: BT2mjd(x))
            ZK[['ra_deg','dec_deg']]= ZK.loc[:, ['SwtDPos_X', 'SwtDPos_Y', 'SwtDPos_Z', 'mjd']].apply(lambda x: xyz2radec(x[0], x[1], x[2], x[3], type='deg'), axis=1, result_type='expand')

            ra0 = (ZK.loc[:,'ra_deg'].values[:,np.newaxis] + np.array(beams['beam_offset_ra'])/60./np.cos(ZK.loc[:,'dec_deg'].values[:,np.newaxis]/180*np.pi)).transpose((1,0))
            dec0 = (ZK.loc[:,'dec_deg'].values[:,np.newaxis] + np.array(beams['beam_offset_dec'])/60.).transpose((1,0))
            ra, dec = np.zeros(shape=mjd.shape), np.zeros(shape=mjd.shape)
            for i,tmpmjd in enumerate(mjd):
                f_ra = interp1d(ZK.loc[:,'mjd'].values, ra0[beam[i]-1])
                f_dec = interp1d(ZK.loc[:,'mjd'].values, dec0[beam[i]-1])
                ra[i], dec[i] = f_ra(tmpmjd), f_dec(tmpmjd)
        else:
            raise ValueError('Your observed time outside this file (from ' + str(BT2mjd(ZK.iloc[0]['SysTime'])) + '[' + ZK.iloc[0]['SysTime'] + '] to ' +  str(BT2mjd(ZK.iloc[-1]['SysTime'])) + '[' + ZK.iloc[-1]['SysTime'] + ']).')
    elif filetype == 'processed':
        with fits.open(posfile) as hdu:
            ra0 = (hdu[1].data['ra_deg'][:,np.newaxis] + np.array(beams['beam_offset_ra'])/60./np.cos(hdu[1].data['dec_deg'][:,np.newaxis]/180*np.pi)).transpose((1,0))
            dec0= (hdu[1].data['dec_deg'][:,np.newaxis] + np.array(beams['beam_offset_dec'])/60.).transpose((1,0))
            mjd_crafts = hdu[1].data['mjd']
        if np.min(mjd-mjd_crafts[0])>=0 and np.max(mjd-mjd_crafts[-1])<=0 :
            ra, dec = np.zeros(shape=mjd.shape), np.zeros(shape=mjd.shape)
            beam_set = list(set(beam))
            for ibeam in beam_set:
                indx_ibeam = np.where(beam == ibeam)[0]
                f_ra = interp1d(mjd_crafts, ra0[ibeam-1])
                f_dec = interp1d(mjd_crafts, dec0[ibeam-1])
                ra[indx_ibeam],dec[indx_ibeam] = f_ra(mjd[indx_ibeam]), f_dec(mjd[indx_ibeam])
        else:
            raise ValueError('Your observed time (' + str(np.min(mjd)) + 'to' + str(np.max(mjd)) + ') outside this file (from ' + str(mjd_crafts[0]) + '[' + mjd2BT(mjd_crafts[0]) + '] to ' +  str(mjd_crafts[-1]) + '[' + mjd2BT(mjd_crafts[-1]) + ']).')
    else:
        raise ValueError("Error filetype: 'processed' or 'raw'.")
    return ra, dec

import ephem
import datetime as dt
from astropy.time import Time
from datetime import datetime
from astropy.coordinates import SkyCoord,EarthLocation, AltAz
nowdate = datetime.now().strftime('%Y%m%d')

def BT2mjd(BT):
    """
    coder: Zheng Zheng & Zheng Yinghui, 2020-2022
    """
    bt = datetime.strptime(BT, "%Y-%m-%d %H:%M:%S.%f")
    utc = bt - dt.timedelta(hours=8)
    mjd = Time(str(utc),format='iso',scale='utc').mjd
    return mjd

#@jit(nopython=True, parallel=True)
def localtoec(az,el,mjd, type='deg'):
    """
    coder: Zheng Zheng & Zheng Yinghui, 2020-2022
    """
    from astropy import units
    getlocal = ephem.Observer()
    getlocal.long, getlocal.lat = '106.85645657571428','25.6534387'
    getlocal.pressure = 925
    getlocal.elevation = 1110.0288
    getlocal.temp = 25
    getlocal.epoch = ephem.J2000
    jd=mjd+2400000.5
    date=ephem.julian_date('1899/12/31 12:00:00')
    djd=jd-date
    ct3=ephem.Date(djd)
    getlocal.date = ct3 # UT
    ra,dec = getlocal.radec_of(np.deg2rad(az),np.deg2rad(el))
    radec= SkyCoord(str(ra)+' '+str(dec),unit=(units.hour,units.deg))
    radeg=radec.ra.degree
    decdeg=radec.dec.degree
    if type == 'deg':
        return np.array([radeg,decdeg])
    elif type == 'str':
        return np.array([str(ra),str(dec)])
    elif type == 'both':
        return [radeg,decdeg,str(ra),str(dec)]
    else:
        raise ValueError("Type Error: the input argment 'type' should be 'deg' or 'str' or 'both'.")


def KY2radec0_concatenate(KYpath,driftname,outfile,log_path='./KY2radec_.log',pathmode='path'):
    '''
    coder: Zheng Zheng & Zheng Yinghui, 2020-2022
        Input:
            'KYpath': str or list
                The path where KY recorded file storaged or where the file is.
            'pathmode': str, default = 'path'
                'path': If your input KYpath is where KY file storaged, choose 'path' mode.
                'file': If your input KYpath is where KY file is, choose 'file' mode.
    '''
    if type(KYpath) == str:
        KYpath = [KYpath]
    log = open(log_path,'a+')
    log.writelines('\n\n'+nowdate+'\n')

    if pathmode == 'path':
        KYfile = []
        for tmp_KYpath in KYpath:
            KYfilepat = os.path.join(tmp_KYpath, driftname + '*.xlsx')
            tmp_KYfile = glob.glob(KYfilepat)
            print(len(tmp_KYfile), 'KY file(s) in ', KYfilepat)
            log.writelines('has ' + str(len(tmp_KYfile)) + ' KY file in ' + KYfilepat + '\n')
            KYfile = np.concatenate([KYfile,tmp_KYfile])
    elif pathmode == 'file':
        KYfile = KYpath
        KYfilepat = KYpath

    if len(KYfile) < 1:
        raise ValueError('There is no result of ' + KYfilepat)
    elif len(KYfile) > 1:
        log.writelines('Totally ' + str(len(KYfile)) + ' KY file' + '\n')
        print('Totally: ' + str(len(KYfile)) + ' KY file')
        
        obstime = []
        for x in KYfile:
            obstime.append(x[-17:-9])
        ind_sort = np.argsort(obstime)
        KYfile = np.array(KYfile)
        KYfile_sorted = KYfile[ind_sort]
    else:
        KYfile_sorted = KYfile

    xls_mjd, xls_x, xls_y, xls_z = [],[],[],[]

    time_start, time_end = None, None
    print('KY file list: ', KYfile_sorted)
    for tmp_KYfile in KYfile_sorted:
        print('Reading the KY file: ', tmp_KYfile)
        ZK = pd.read_excel(tmp_KYfile,sheet_name='测量数据')
        if (time_start==None) or (time_end==None):
            xls_mjd = list(ZK['SysTime'])
            xls_x = list(ZK['SwtDPos_X'])
            xls_y = list(ZK['SwtDPos_Y'])
            xls_z = list(ZK['SwtDPos_Z'])
            time_start,time_end = ZK.iloc[0]['SysTime'], ZK.iloc[-1]['SysTime']
        elif (ZK.iloc[0]['SysTime']<=time_end) and (ZK.iloc[-1]['SysTime']>time_end):    #partly overlap (behind)
            ind_add = np.where(ZK['SysTime']>time_end)
            xls_mjd = xls_mjd + list(ZK.iloc[ind_add]['SysTime'])
            xls_x = xls_x + list(ZK.iloc[ind_add]['SwtDPos_X'])
            xls_y = xls_y + list(ZK.iloc[ind_add]['SwtDPos_Y'])
            xls_z = xls_z + list(ZK.iloc[ind_add]['SwtDPos_Z'])
            time_end = ZK.iloc[-1]['SysTime']
        elif (ZK.iloc[-1]['SysTime']<time_end) and (ZK.iloc[0]['SysTime']<=time_start):  #partly overlap (ahead)
            ind_add = np.where(ZK['SysTime']<time_start)
            xls_mjd =  list(ZK.iloc[ind_add]['SysTime']) + xls_mjd
            xls_x =  list(ZK.iloc[ind_add]['SwtDPos_X']) + xls_x
            xls_y = list(ZK.iloc[ind_add]['SwtDPos_Y']) + xls_y
            xls_z = list(ZK.iloc[ind_add]['SwtDPos_Z']) + xls_z        
            time_start = ZK.iloc[0]['SysTime']
        elif (ZK.iloc[-1]['SysTime']<=time_end) and (ZK.iloc[0]['SysTime']>=time_start):   #contain
            # log.writelines(tmp_KYfile + ' was totally overlaped.')
            print('Pass')
        elif (ZK.iloc[0]['SysTime']<=time_start) and (ZK.iloc[-1]['SysTime']>=time_end):  #larger
            xls_mjd = list(ZK['SysTime'])
            xls_x = list(ZK['SwtDPos_X'])
            xls_y = list(ZK['SwtDPos_Y'])
            xls_z = list(ZK['SwtDPos_Z'])
            time_start,time_end = ZK.iloc[0]['SysTime'], ZK.iloc[-1]['SysTime']
        elif ZK.iloc[0]['SysTime']>time_end or ZK.iloc[-1]['SysTime']<time_start:    #gaps
            # raise ValueError('There is gap between '+tmp_KYfile+' and other KY files.')
            # nan_mjd = []
            mjd_nan_edge = [ time_start, time_end, ZK.iloc[0]['SysTime'], ZK.iloc[-1]['SysTime'] ]
            log.writelines('There is gap between '+tmp_KYfile+' and other KY files.\n')
            log.writelines("time_start, time_end, ZK.iloc[0]['SysTime'], ZK.iloc[-1]['SysTime']\n")
            log.writelines(str(mjd_nan_edge)+'\n')
            print('There is gap between '+tmp_KYfile+' and other KY files.')
            print("time_start, time_end, ZK.iloc[0]['SysTime'], ZK.iloc[-1]['SysTime']")
            print(str(mjd_nan_edge))

            ind_edge = np.argsort(mjd_nan_edge)
            mjd_nan_edge.sort()
            mjd_add0 = datetime.strptime(mjd_nan_edge[1], "%Y-%m-%d %H:%M:%S.%f")   # is BT not mjd or utc
            mjd_add1 = datetime.strptime(mjd_nan_edge[2], "%Y-%m-%d %H:%M:%S.%f")
            log.writelines('gap start from: '+ mjd_nan_edge[1] + '\t' + str(BT2mjd(mjd_nan_edge[1])) +'\n')
            log.writelines('gap end at: '+ mjd_nan_edge[2] + '\t' + str(BT2mjd(mjd_nan_edge[2])) +'\n')
            log.writelines('\n\n')
            print('gap start from: '+ mjd_nan_edge[1] + '\t' + str(BT2mjd(mjd_nan_edge[1])))
            print('gap end at: '+ mjd_nan_edge[2] + '\t' + str(BT2mjd(mjd_nan_edge[2])))

            num_nan = int((mjd_add1-mjd_add0)/dt.timedelta(seconds=0.2))
            nan_mjd = [(mjd_add0 + dt.timedelta(seconds=0.2*t)).strftime("%Y-%m-%d %H:%M:%S.%f") for t in range(num_nan)]
            nan_mjd = nan_mjd[1:]
            
            if ind_edge[0] == 0:
                xls_mjd = xls_mjd + nan_mjd + list(ZK['SysTime'])
                xls_x = xls_x + [np.nan]*num_nan + list(ZK['SwtDPos_X'])
                xls_y = xls_y + [np.nan]*num_nan + list(ZK['SwtDPos_Y'])
                xls_z = xls_z + [np.nan]*num_nan + list(ZK['SwtDPos_Z'])
            elif ind_edge[0] == 2:
                xls_mjd = list(ZK['SysTime'])   + nan_mjd + xls_mjd
                xls_x   = list(ZK['SwtDPos_X']) + [np.nan]*num_nan + xls_x
                xls_y   = list(ZK['SwtDPos_Y']) + [np.nan]*num_nan + xls_y
                xls_z   = list(ZK['SwtDPos_Z']) + [np.nan]*num_nan + xls_z
            else:
                raise ValueError('Error for gaps edge caculation, please check by yourself.')
        else:
            raise ValueError(tmp_KYfile + ' other situations')

    xls_mjd, xls_x, xls_y, xls_z = np.array(xls_mjd), np.array(xls_x), np.array(xls_y), np.array(xls_z)

    num = xls_mjd.shape[0]

    mjds = []
    ras = []
    decs = []
    rad = []
    decd = []

    for i in range(int(num)):
        # utc8 = datetime.strptime(xls_mjd[i], "%Y-%m-%d %H:%M:%S.%f")
        # utc = utc8 - dt.timedelta(hours=8)
        # mjd = Time(str(utc),format='iso',scale='utc').mjd
        mjd = BT2mjd(xls_mjd[i])

        if np.isnan(xls_x[i]) or np.isnan(xls_y[i]) or np.isnan(xls_z[i]):
            mjds.append(mjd)
            ras.append(str(np.nan))
            decs.append(str(np.nan))
            rad.append(np.nan)
            decd.append(np.nan)
        else:
            x0 = float(xls_x[i])
            y0 = float(xls_y[i])
            z0 = float(xls_z[i])

            R=np.sqrt(x0**2+y0**2+z0**2)
            z=-z0
            y=-x0
            x=-y0
            az0 = 0.0
            if (np.abs(x) < 1.0e-8):
                if (y>0):
                    az0 = 90.0
                else:
                    az0 = 270.0
            else:
                tempaz0 = np.rad2deg(np.arctan(y/x))
                if (x>0 and y>0):
                    az0 = tempaz0
                if ((x>0 and y<0) or (x>0 and y==0)):
                    az0 = tempaz0+360.0
                if (x<0 and y>0):
                    az0 = tempaz0+180.0
                if ((x<0 and y<0) or (x<0 and y==0)):
                    az0 = tempaz0+180.0
            el0 = np.rad2deg(np.arcsin(z/R))
            tmp_rad,tmp_decd,tmp_ras,tmp_decs=localtoec(az0,el0,mjd,type='both')
            mjds.append(mjd)
            ras.append(tmp_ras)
            decs.append(tmp_decs)
            rad.append(tmp_rad)
            decd.append(tmp_decd)
            #testbody = ephem.Equatorial(ra,dec,epoch=ephem.J2000)
            #testbody_now = ephem.Equatorial(testbody,epoch='2020-08-18')
            #print(i, x,y,z,ra, dec, testbody_now.ra,testbody_now.dec)

    mjds = np.array(mjds)
    ras = np.array(ras)
    decs = np.array(decs)
    rad = np.array(rad)
    decd = np.array(decd)
    print('writing to the output file: ', outfile)
    col1 = fits.Column(name='mjd',format = 'D',array=mjds)
    col2 = fits.Column(name='ra',format = '20A',array=ras)
    col3 = fits.Column(name='dec',format = '20A',array=decs)
    col4 = fits.Column(name='ra_deg',format = 'D',array=rad)
    col5 = fits.Column(name='dec_deg',format = 'D',array=decd)
    coldefs = fits.ColDefs([col1,col2,col3,col4,col5])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto(outfile,overwrite=True)
    print('write to ',outfile)

class coordinate:
      def __init__(place):
         place.name=''
         place.lon=''
         place.lat=''
         place.alt=''

def get_za(str_RA, str_Dec, mjd):
    """
    get zenith angle (ZA) of observing source. 
    The ZA of driftscan obs is obtained from the coordinate and time at the starting moment.

    Input
    ----------
    str_RA: in the format of 'hh mm ss.s'
    str_Dec: in the format of 'dd mm ss.s'
    mjd: float, the mjd of that day

    Output
    ----------
    za: float, the zenith angle of the source at that mjd
 
    Coder: Pei Zuo @ 2020
    """    
    import ephem
    import time
    global convert
    from PyAstronomy import pyasl

    SiteFAST = coordinate()
    SiteFAST.Name = 'Dawodang'
    SiteFAST.Long = str(106.856666872)
    SiteFAST.Lati = str(25.6529518158)
    SiteFAST.elevation = 1110.0288801
    
    InfoSite = ephem.Observer()
    InfoSite.lon = SiteFAST.Long
    InfoSite.lat = SiteFAST.Lati
    InfoSite.temp = 25.0
    InfoSite.pressure = 1.01325e3
    InfoSite.epoch = ephem.J2000

    MJD = mjd
    JD = MJD + 2400000.5
    Date = ephem.julian_date('1899/12/31 12:00:00') 
    DJD = JD - Date    
    Time = ephem.Date(DJD)
    InfoSite.date = Time
    
    Src = ephem.FixedBody()
    # str_Coor = pyasl.coordsDegToSexa(str_RA, str_Dec)
    Src._ra = str_RA
    Src._dec = str_Dec
    Src.compute(InfoSite)
    
    za = (np.pi/2.0 - Src.alt)/np.pi*180.0
    return za

def get_DFabc(file_ita=None):
    """
    coder: Niankun Yu @ 2022.12.08
    build the table 3 of Jiang et al. 2020

    Input
    ----------
    file_ita: the data of Table 3, including its path
    freq_cen: float, the central frequency of the beam
    za: float, the zenith angle in unit of degree
    beam: str, the FAST beam, default "M01"

    Output
    ----------
    """
    if file_ita != None:
        if os.path.exists(file_ita):
            names_ita = ("beam", "abc", "abc_1050", "abc_1050_plus", "abc_1050_error",\
                "abc_1100", "abc_1100_plus", "abc_1100_error",\
                "abc_1150", "abc_1150_plus", "abc_1150_error",\
                "abc_1200", "abc_1200_plus", "abc_1200_error",\
                "abc_1250", "abc_1250_plus", "abc_1250_error",\
                "abc_1300", "abc_1300_plus", "abc_1300_error",\
                "abc_1350", "abc_1350_plus", "abc_1350_error",\
                "abc_1400", "abc_1400_plus", "abc_1400_error",\
                "abc_1450", "abc_1450_plus", "abc_1450_error")
            pd_ita = pd.read_csv(file_ita, skiprows=1, names=names_ita, delim_whitespace = True,\
            dtype = {"abc_1050":float,  "abc_1100":float,  "abc_1150":float,  "abc_1200":float,  "abc_1250":float, \
                "abc_1300":float,  "abc_1350":float,  "abc_1400":float,  "abc_1450":float, } )
            pd_ita = pd_ita[["beam", "abc", "abc_1050", "abc_1050_error",\
                "abc_1100", "abc_1100_error",\
                "abc_1150", "abc_1150_error",\
                "abc_1200", "abc_1200_error",\
                "abc_1250", "abc_1250_error",\
                "abc_1300", "abc_1300_error",\
                "abc_1350", "abc_1350_error",\
                "abc_1400", "abc_1400_error",\
                "abc_1450", "abc_1450_error"]]
            return pd_ita
        else:
            print("Wrong in get_DFabc, because we could not find the file_ita!")
            sys.exit()
    else:
        data_beam = ["M01", "M01", "M01", "M02", "M02", "M02", "M03", "M03", "M03", "M04", "M04", "M04", "M05", "M05", "M05",\
            "M06", "M06", "M06", "M07", "M07", "M07", "M08", "M08", "M08", "M09", "M09", "M09", "M10", "M10", "M10",\
                "M11", "M11", "M11", "M12", "M12", "M12", "M13", "M13", "M13", "M14", "M14", "M14", "M15", "M15", "M15",\
            "M16", "M16", "M16", "M17", "M17", "M17", "M18", "M18", "M18", "M19", "M19", "M19"]
        data_paras = ["a/1e-4", "b/1e-1",  "c/1e-2", "a/1e-4",  "b/1e-1", "c/1e-2",  "a/1e-4", "b/1e-1",  "c/1e-2",\
            "a/1e-4",  "b/1e-1", "c/1e-2",  "a/1e-4", "b/1e-1",  "c/1e-2", "a/1e-4",  "b/1e-1", "c/1e-2",\
            "a/1e-4", "b/1e-1",  "c/1e-2", "a/1e-4",  "b/1e-1", "c/1e-2",  "a/1e-4", "b/1e-1",  "c/1e-2",\
            "a/1e-4",  "b/1e-1", "c/1e-2",  "a/1e-4", "b/1e-1",  "c/1e-2", "a/1e-4",  "b/1e-1", "c/1e-2",\
            "a/1e-4", "b/1e-1",  "c/1e-2", "a/1e-4",  "b/1e-1", "c/1e-2",  "a/1e-4", "b/1e-1",  "c/1e-2",\
            "a/1e-4",  "b/1e-1", "c/1e-2",  "a/1e-4", "b/1e-1",  "c/1e-2", "a/1e-4",  "b/1e-1", "c/1e-2",  "a/1e-4", "b/1e-1",  "c/1e-2"]
        abc_1050 =       np.asarray([ 3.31, 6.19, -1.58, 9.46, 5.97, -1.54, 1.79, 6.41, -1.7,  3.57, 6.24, -1.8,\
                -3.97, 6.45, -1.82, 1.69, 6.16, -1.74, 1.71, 6.1, -1.68, 13.91, 5.76, -1.6,\
                -1.26, 5.96, -1.39, 14.36, 6.01, -1.79, 7.68, 5.9, -1.59, -3., 6.06, -1.56,\
                -1.33, 6.18, -1.49, 0.35, 5.91, -1.56, 1.33, 5.62, -1.42, 9.81, 5.44, -1.58,\
                -0.89, 5.81, -1.54, -1.88, 5.64, -1.37, 11.82, 5.7, -1.76])
        abc_1050_error = np.asarray([2.23, 0.04, 0.02, 3.53, 0.06, 0.03, 4.21, 0.07, 0.03, 4.76, 0.08, 0.03, 2.87, 0.05,\
                0.03, 2.84, 0.04, 0.03, 2.43, 0.04, 0.02, 3.13, 0.05, 0.06, 2.03, 0.03, 0.04, 3.8,\
                0.06, 0.04, 5.51, 0.09, 0.04, 5.82, 0.09, 0.05, 4.57, 0.08, 0.04, 3.81, 0.06, 0.04,\
                2.89, 0.05, 0.04, 3.39, 0.06, 0.04, 3.92, 0.06, 0.02, 2.27, 0.04, 0.02, 2.94, 0.05, 0.03])
        abc_1100 =       np.asarray([ 2.54,  6.32, -1.61,  6.28,  5.97, -1.62,  4.39,  6.24, -1.8,   1.87,  6.15, -1.82,\
                -4.25,  6.48, -1.86,  0.74,  6.22, -1.8,   0.08,  6.04, -1.64, 17.,    5.75, -1.7,\
                -0.43,  5.95, -1.43, 15.37,  5.98, -1.83,  2.98,  5.94, -1.59, -2.08,  5.96, -1.49,\
                0.41,  6.12, -1.59, -0.57,  5.98, -1.67, -0.59,  5.78, -1.53,  3.14,  5.52, -1.52,\
                -4.49,  5.83, -1.52, -5.55,  5.68, -1.34,  6.55,  5.88, -1.75])
        abc_1100_error = np.asarray([1.85, 0.03, 0.03, 3.09, 0.05, 0.03, 3.55, 0.06, 0.03, 4.08, 0.07, 0.03, 2.72, 0.04,\
                0.03, 2.59, 0.04, 0.03, 2.18, 0.04, 0.02, 2.98, 0.05, 0.05, 1.99, 0.03, 0.03, 3.39,\
                0.05, 0.04, 4.73, 0.08, 0.03, 4.75, 0.08, 0.04, 4.05, 0.07, 0.03, 3.77, 0.06, 0.04,\
                2.55, 0.04, 0.03, 3.06, 0.05, 0.03, 3.39, 0.06, 0.02, 2.09, 0.03, 0.02, 2.78, 0.04, 0.03])
        abc_1150 =       np.asarray([0.34, 6.43, -1.37, 9.01, 5.99, -1.46, 2.41, 6.42, -1.56, 3.82, 6.25, -1.65,\
                -4.44, 6.68, -1.72, -0., 6.33, -1.6,  0.77, 6.18, -1.52, 14.04, 5.81, -1.52,\
                -2.68, 5.98, -1.28, 15.65, 6.04, -1.74, 4.16, 6.,  -1.55, 1.33, 5.91, -1.45,\
                2.41, 6.22, -1.53, 4.27, 5.91, -1.65, -3.57, 5.83, -1.34, 0.38, 5.55, -1.46,\
                -3.98, 5.85, -1.41, -5.94, 5.81, -1.34, 5.01, 5.92, -1.59])
        abc_1150_error = np.asarray([1.83, 0.03, 0.04, 3.14, 0.05, 0.03, 4.46, 0.07, 0.03, 3.93, 0.06, 0.02, 2.72, 0.04,\
                0.03, 4.17, 0.07, 0.03, 2.41, 0.04, 0.02, 3.81, 0.06, 0.06, 1.94, 0.03, 0.04, 3.78,\
                0.06, 0.03, 4.57, 0.07, 0.03, 4.29, 0.07, 0.05, 4.07, 0.07, 0.04, 3.61, 0.06, 0.04,\
                3.01, 0.05, 0.04, 4.28, 0.07, 0.03, 3.71, 0.06, 0.02, 2.11, 0.03, 0.02, 3.51, 0.06, 0.03])
        abc_1200 =       np.asarray([ 5.03, 6.22, -1.4, 12.13, 5.81, -1.44, 5.95, 6.21, -1.63, 3.9,  6.05, -1.57,\
                -7.54, 6.42, -1.56, 1.06, 6.01, -1.5,  2.03, 5.9, -1.39, 16.81, 5.62, -1.5,\
                0.12, 5.75, -1.24, 15.2,  5.81, -1.73, 3.38, 5.81, -1.35, 4.98, 5.82, -1.56,\
                -1.35, 6.23, -1.54, 3.01, 5.76, -1.57, 3.71, 5.59, -1.53, 3.85, 5.47, -1.51,\
                -7.57, 5.71, -1.25, -1.53, 5.6, -1.29, 10.73, 5.67, -1.6 ])
        abc_1200_error = np.asarray([2.03, 0.03, 0.04, 3.12, 0.05, 0.02, 4.69, 0.07, 0.02, 4.6, 0.08, 0.06, 3.48, 0.06,\
                0.03, 5.18, 0.08, 0.04, 3.25, 0.05, 0.02, 4.98, 0.08, 0.06, 1.67, 0.03, 0.04, 4.85,\
                0.08, 0.03, 6.44, 0.1, 0.05, 5.1, 0.08, 0.05, 4.89, 0.08, 0.07, 4.84, 0.08, 0.04,\
                5.82, 0.09, 0.03, 3.48, 0.06, 0.03, 4.51, 0.07, 0.03, 2.53, 0.04, 0.02, 3.34, 0.05, 0.04])
        abc_1250 =       np.asarray([ 4.94, 6.22, -1.42, 10.98, 5.81, -1.42, 6.9,  6.17, -1.59, 8.46, 5.98, -1.58,\
                -4.09, 6.39, -1.6,  0.97, 6.04, -1.44, 2.24, 6.02, -1.43, 17.71, 5.54, -1.5,\
                0.96, 5.77, -1.24, 18.06, 5.86, -1.78, 2.45, 5.8, -1.36, 8.5,  5.76, -1.69,\
                2.44, 6.04, -1.54, 7.37, 5.73, -1.64, -1.46, 5.68, -1.42, 6.3,  5.29, -1.41,\
                -3.25, 5.65, -1.33, -2.21, 5.57, -1.2,  9.32, 5.69, -1.57])
        abc_1250_error = np.asarray([2.02, 0.03, 0.03, 3.06, 0.05, 0.02, 3.21, 0.05, 0.02, 4.65, 0.08, 0.02, 4.19, 0.07,\
                0.03, 3.49, 0.06, 0.03, 3.64, 0.06, 0.02, 2.95, 0.05, 0.05, 1.6, 0.03, 0.03, 2.92,\
                0.05, 0.03, 4.86, 0.08, 0.03, 4.02, 0.06, 0.03, 3.87, 0.06, 0.03, 4.78, 0.08, 0.03,\
                2.36, 0.04, 0.03, 2.65, 0.04, 0.02, 2.74, 0.05, 0.02, 3.35, 0.05, 0.02, 2.79, 0.04, 0.03])
        abc_1300 =       np.asarray([ 9.83, 6.08, -1.38, 11.31, 5.65, -1.29, 7.41, 6.07, -1.56, 4.29, 5.92, -1.42,\
                -5.41, 6.26, -1.53, 0.33, 5.92, -1.33, 3.46, 5.86, -1.32, 17.29, 5.46, -1.47,\
                1.53, 5.74, -1.18, 17.97, 5.75, -1.74, 4.27, 5.67, -1.31, -0.98, 5.7, -1.35,\
                2.6,  6.,  -1.57, 2.58, 5.78, -1.68, -2.34, 5.61, -1.32, 5.25, 5.28, -1.36,\
                -0.3,  5.47, -1.22, -0.75, 5.46, -1.13, 8.64, 5.61, -1.5 ])
        abc_1300_error = np.asarray([1.85, 0.03, 0.03, 2.16, 0.04, 0.02, 2.6, 0.04, 0.02, 3.22, 0.05, 0.02, 2.35, 0.04,\
                0.02, 1.82, 0.03, 0.02, 1.82, 0.03, 0.02, 2.75, 0.04, 0.05, 1.53, 0.02, 0.04, 2.67,\
                0.04, 0.03, 3.39, 0.06, 0.03, 3.17, 0.05, 0.04, 3.22, 0.05, 0.02, 2.61, 0.04, 0.02,\
                1.73, 0.03, 0.03, 2.13, 0.04, 0.02, 2.47, 0.04, 0.01, 1.52, 0.02, 0.02, 2.51, 0.04, 0.03])
        abc_1350 =       np.asarray([ 8.5,  6.12, -1.4,  8.71, 5.75, -1.32, 9.99, 5.91, -1.53, 2.75, 5.89, -1.45,\
                -3.33, 6.1, -1.45, 3.11, 5.83, -1.38, 5.46, 5.77, -1.34, 16.95, 5.41, -1.45,\
                1.09, 5.64, -1.09, 19.34, 5.72, -1.72, 5.99, 5.66, -1.35, 2.94, 5.58, -1.4,\
                4.14, 5.94, -1.59, 4.43, 5.64, -1.61, -2.8,  5.53, -1.27, 2.32, 5.19, -1.28,\
                -0.8,  5.48, -1.21, -1.19, 5.38, -1.11, 10.22, 5.48, -1.44])
        abc_1350_error = np.asarray([1.93, 0.03, 0.02, 2.09, 0.03, 0.02, 2.63, 0.04, 0.02, 3.28, 0.05, 0.02, 2.26, 0.04,\
                0.03, 1.8, 0.03, 0.02, 1.6, 0.03, 0.02, 2.29, 0.04, 0.05, 1.36, 0.02, 0.04, 2.49,\
                0.04, 0.03, 3.36, 0.05, 0.02, 3.06, 0.05, 0.03, 3.11, 0.05, 0.03, 2.19, 0.04, 0.03,\
                1.71, 0.03, 0.03, 1.95, 0.03, 0.02, 2.36, 0.04, 0.02, 1.52, 0.02, 0.02, 2.15, 0.03, 0.03])
        abc_1400 =       np.asarray([ 7.87, 6.14, -1.34, 8.78, 5.75, -1.33, 7.66, 5.99, -1.47, 1.85, 5.89, -1.36,\
                -4.51, 6.17, -1.44, -0.03, 5.89, -1.35, 6.7,  5.73, -1.36, 15.92, 5.34, -1.37,\
                1., 5.57, -1.06, 16.72, 5.69, -1.66, 5.03, 5.61, -1.27, 2.49, 5.53, -1.34,\
                5.5,  5.85, -1.53, 5.43, 5.59, -1.6, -3.55, 5.5, -1.21, 0.49, 5.17, -1.21,\
                -1.44, 5.41, -1.16, 0.24, 5.38, -1.11, 9.32, 5.43, -1.36])
        abc_1400_error = np.asarray([1.95, 0.03, 0.02, 2.06, 0.03, 0.02, 2.6, 0.04, 0.02, 3.26, 0.05, 0.02, 2.42, 0.04,\
                0.03, 1.76, 0.03, 0.02, 1.65, 0.03, 0.02, 2.31, 0.04, 0.05, 1.11, 0.02, 0.03, 2.52,\
                0.04, 0.03, 3.28, 0.05, 0.02, 2.71, 0.04, 0.03, 2.95, 0.05, 0.03, 2.22, 0.04, 0.04,\
                1.68, 0.03, 0.03, 1.9, 0.03, 0.02, 2.17, 0.04, 0.01, 1.51, 0.02, 0.02, 2.21, 0.04, 0.03])
        abc_1450 =       np.asarray([10.64, 5.98, -1.26, 13.07, 5.62, -1.21, 10.3,  5.86, -1.35, 5.1,  5.75, -1.24,\
                -4.58, 6.11, -1.3,  1.37, 5.75, -1.22, 6.91, 5.62, -1.22, 17.76, 5.22, -1.28,\
                0.27, 5.53, -1.01, 18.57, 5.6, -1.58, 8.23, 5.45, -1.18, 2.77, 5.48, -1.27,\
                5.83, 5.71, -1.44, 5.53, 5.53, -1.51, -2.2,  5.46, -1.13, 1.14, 5.09, -1.14,\
                -0.64, 5.28, -1.08, 0.93, 5.33, -1.05, 11.03, 5.3, -1.25])
        abc_1450_error = np.asarray([1.9, 0.03, 0.02, 1.98, 0.03, 0.02, 2.72, 0.04, 0.03, 3.2, 0.05, 0.02, 2.53, 0.04,\
                0.03, 1.76, 0.03, 0.02, 1.49, 0.02, 0.02, 2.43, 0.04, 0.05, 0.91, 0.02, 0.03, 2.65,\
                0.04, 0.03, 3.05, 0.05, 0.02, 2.6, 0.04, 0.03, 2.77, 0.05, 0.03, 2.19, 0.04, 0.04,\
                1.64, 0.03, 0.03, 1.83, 0.03, 0.02, 2.08, 0.03, 0.02, 1.53, 0.02, 0.02, 2.24, 0.04, 0.03])
        pd_ita = pd.DataFrame({"beam": data_beam, "abc": data_paras, "abc_1050": abc_1050,  "abc_1050_error": abc_1050_error,\
                "abc_1100": abc_1100, "abc_1100_error": abc_1100_error,\
                "abc_1150": abc_1150, "abc_1150_error": abc_1150_error,\
                "abc_1200": abc_1200, "abc_1200_error": abc_1200_error,\
                "abc_1250": abc_1250, "abc_1250_error": abc_1250_error,\
                "abc_1300": abc_1300, "abc_1300_error": abc_1300_error,\
                "abc_1350": abc_1350, "abc_1350_error": abc_1350_error,\
                "abc_1400": abc_1400, "abc_1400_error": abc_1400_error,\
                "abc_1450": abc_1450, "abc_1450_error": abc_1450_error})
        # pd_ita = pd_ita.astype({'beam': str})
        pd_ita = pd_ita.astype(dtype= {'beam': str, "abc":str, "abc_1050": "float64", "abc_1050_error": "float64",\
            "abc_1100": "float64", "abc_1100_error": "float64", "abc_1150": "float64", "abc_1150_error": "float64",\
            "abc_1200": "float64", "abc_1200_error": "float64", "abc_1250": "float64", "abc_1250_error": "float64",\
            "abc_1300": "float64", "abc_1300_error": "float64", "abc_1350": "float64", "abc_1350_error": "float64",\
            "abc_1400": "float64", "abc_1400_error": "float64", "abc_1450": "float64", "abc_1450_error": "float64"})
        return pd_ita

def get_gain(freq_cen, za, beam=None, file_ita=None):
    """
    get the telescope gain by considering the beam, frequency, and zenith angle

    * https://ui.adsabs.harvard.edu/abs/2020RAA....20...64J/abstract
    * aperture efficiency: Equation 5 and Table 3

    Input
    ----------
    file_ita: the data of Table 3, including its path
    freq_cen: float, the central frequency of the beam
    za: float, the zenith angle in unit of degree
    beam: str, the FAST beam, default "M01"

    Output
    ----------
    gain: the aperature efficiency in unit of K/Jy

    Coder: Niankun Yu @ 2022.09.08
    Notes: without considering the error
    """
    if beam == None:
        beam = "M01"
    ######## the frequency range in the table
    freqs = 1050 + np.arange(9)*50
    dfs = np.abs(freq_cen - freqs)
    ind = np.argmin(dfs)
    pd_ita = get_DFabc(file_ita)
    names_ita = pd_ita.columns
    ####### selct the eaxct beam:
    pd_ita_beam = pd_ita[pd_ita["beam"]== beam].reset_index(drop=True)
    ####### get the values of a, b, c, and d
    abc_unit = pd_ita_beam["abc"]
    a_unit = float(abc_unit[0].split('e')[-1])
    b_unit = float(abc_unit[1].split('e')[-1])
    c_unit = float(abc_unit[2].split('e')[-1])
    abc_freq = pd_ita_beam[names_ita[2*ind+2]]
    a = float(abc_freq[0])*10**(a_unit)
    b = float(abc_freq[1])*10**(b_unit)
    c = float(abc_freq[2])*10**(c_unit)
    d = b+26.4*(a-c)
    if za<26.4:
        ita = a*za+b
    elif za<40:
        ita = c*za+d
    else:
        ita = 0
        print("The declination is out of range!")  
    gain0 = 25.6  ####### 25.6 K/Jy
    gain = gain0*ita
    return gain

def get_radec(beam, mjd_on, mjd_off, dataKY, dateKY, name):
    """
    get the position based on the KY file, the mjd
    coder: Niankun Yu @ 2022.12.08

    Input
    ----------
    file_ita: the data of Table 3, including its path
    freq_cen: float, the central frequency of the beam
    za: float, the zenith angle in unit of degree
    beam: str, the FAST beam, default "M01"

    Output
    ----------

    """
    KY_path, KY_files = get_KYfile(dataKY, dateKY, name)
    beamN = fast_basic.get_beamN(beam)+1
    count_xlsx = 0
    for element in KY_files:
        if ".xlsx" in element:
            count_xlsx = count_xlsx+1
    if count_xlsx == 1:
        # KY_outfile = outpath+KY_files[0].split("/")[-1]
        ra_deg_on, dec_deg_on = mjd2radec(mjd_on, [beamN]*len(mjd_on), posfile=KY_files[0],filetype='raw')
        ra_deg_off, dec_deg_off = mjd2radec(mjd_off, [beamN]*len(mjd_off), posfile=KY_files[0],filetype='raw')
    else:
        print("There are more than two KY file for one source")
        KY2radec0_concatenate(KY_path, name, outfile='driftname.fits.gz',log_path='./KY2radec_.log',pathmode='path')
        ra_deg_on, dec_deg_on = mjd2radec(mjd_on, [beamN]*len(mjd_on),posfile='driftname.fits.gz',filetype='processed')
        ra_deg_off, dec_deg_off = mjd2radec(mjd_off, [beamN]*len(mjd_off),posfile='driftname.fits.gz',filetype='processed')
    return ra_deg_on, dec_deg_on, ra_deg_off, dec_deg_off

def T2mJy(fits_file, dataKY, dateKY, name, freqc, file_ita =None, interp_method = None):
    """
    convert temperature to mJy
    coder: Niankun Yu @ 2022.12.08

    Input
    ----------

    Output
    ----------

    """
    if fits_file.__contains__("M01"):
        beam = "M01"
    elif fits_file.__contains__("M14"):
        beam = "M14"
    else:
        print("So far, we do not process data of other beams.")
        sys.exit()
    # beamN = basic_functions.get_beamN(beam)+1
    freq, temp_on, temp_off, mjd_calon_on, mjd_calon_off = fast_basic.read_fits2(fits_file)
    ra_deg_on, dec_deg_on, ra_deg_off, dec_deg_off = get_radec(beam, mjd_calon_on, mjd_calon_off, dataKY, dateKY, name)
    g_on = []
    g_off = []
    for i in range(len(mjd_calon_on)):
        ra_hms_on, dec_dms_on = fast_basic.radec_deg2hmsdms(ra_deg_on[i], dec_deg_on[i])
        za_on = get_za(ra_hms_on, dec_dms_on, mjd_calon_on[i])
        gain_on = get_gain(freqc, za_on, beam, file_ita)
        g_on_i = 1000./gain_on
        g_on.append(g_on_i)
    for i in range(len(mjd_calon_off)):
        ra_hms_off, dec_dms_off = fast_basic.radec_deg2hmsdms(ra_deg_off[i], dec_deg_off[i])
        za_off = get_za(ra_hms_off, dec_dms_off, mjd_calon_off[i])
        gain_off = get_gain(freqc, za_off, beam, file_ita)
        g_off_i = 1000./gain_off
        g_off.append(g_off_i)
    
    len1_on, len2_on, len3_on = np.shape(temp_on)
    ontemp0_mask = fast_basic.MaskData_Interpolate3d(temp_on, interp_method)
    on_temp = np.mean(ontemp0_mask, axis=0)
    on_flux_mask = np.ma.dot(ontemp0_mask.T, g_on).T
    on_flux = on_flux_mask/len1_on
    on_flux = np.reshape(on_flux, (len(freq), 2))

    len1, len2, len3 = np.shape(temp_off)
    offtemp0_mask = fast_basic.MaskData_Interpolate3d(temp_off, interp_method)
    off_temp = np.mean(offtemp0_mask, axis=0)
    off_flux_mask = np.ma.dot(offtemp0_mask.T, g_off).T
    off_flux = off_flux_mask/len1
    off_flux = np.reshape(off_flux, (len(freq), 2))
    # print(np.shape(offtemp0_mask), np.shape(off_temp), np.shape(off_flux_mask), np.shape(off_flux))
    return freq, on_temp, off_temp, on_flux, off_flux
    
def get_vel(freq, ra, dec, mjd):
    """
    convert frequency to velocity
    coder: Niankun Yu @ 2022.12.13

    Input
    ----------
    freq: 1d array, the frequency array, 
    ra, dec, mjd: 1d array, they have the same length


    Output
    ----------
    vel_radio, vel_optical, velo_cor: 1d array, radio velocity, optical velocity, 
            optical velocity corrected for Doppler broadening.
            They have the same length as the input frequency

    """
    vh_med = fast_basic.get_Vdoppler(ra, dec, mjd)
    vel_radio, vel_optical, velo_cor = fast_basic.get_Freq2Vel(freq, vh_med)
    return vel_radio, vel_optical, velo_cor


def get_mJyVel(fits_file, dataKY, dateKY, name, freqc, plot_fig = "", write_txt = "", file_ita =None, interp_method = None):
    """
    convert the temperature to mJy, by using the telescope gain

    Input
    ---------
    fits_file: str, one of the file in 
        outpath2 + name + '_'+date+'_'+beams[0]+'_*.fits'
    freqc: float, the central frequency of the source
        For example: 1380 MHz
    outpath: str, the path to save the figure and spectrum

    Coder: Niankun Yu @ 2022.10.18
    """
    str_ind = fits_file.find('_M')
    beam = fits_file[str_ind+1:str_ind+4]
    ############# 3d array of temp_on, temp_off
    freq, temp_on, temp_off, mjd_on, mjd_off = fast_basic.read_fits2(fits_file)
    ############# 2d array of on_temp, off_temp, on_flux, off_flux
    freq, on_temp, off_temp, on_flux, off_flux = T2mJy(fits_file, dataKY, dateKY, name, freqc, file_ita, interp_method)
    ra_deg_on, dec_deg_on, ra_deg_off, dec_deg_off = get_radec(beam, mjd_on, mjd_off, dataKY, dateKY, name)
    ############# get the velocity
    vh_med_on = fast_basic.get_Vdoppler(ra_deg_on, dec_deg_on, mjd_on)
    vh_med_off = fast_basic.get_Vdoppler(ra_deg_off, dec_deg_off, mjd_off)
    vh_med = (vh_med_on+vh_med_off)/2.0
    vel_radio, vel_optical, velo_cor = fast_basic.get_Freq2Vel(freq, vh_med)
    pd_cal = fast_basic.write_txtCal(freq, on_temp, off_temp, on_flux, off_flux, vel_radio, vel_optical, velo_cor, plot_fig , write_txt, freqc)
    return pd_cal

