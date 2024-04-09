"""
Make the code public so that people could process the data quickly.
Programmer: Niankun Yu @ 2024.04.09 (niankunyu@bao.ac.cn)
If you use this code to process the FAST data, please read the data process algorithm cite our papers 
(https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3085Z/abstract, https://arxiv.org/abs/2403.19447).
Our observation setting is shown in 7815_12705_FASTobservation.jpg, onoff mode. If you follow this observation setting, 
then you could use our code to process the FAST raw data directly after copying the raw data (_W_*.fits) and KY (.xlsx) files

The work flow is:
(1) read the data, and seperate the source on and source off data (FASTfunc_basic.py, FASTfunc_onoff.py)
(3) get the telescpe gain and calibrate the data (FASTfunc_calibration.py)
(4) remove the ripple and baseline (FAST_func_spectrum.py)
(5) stack the final spectra (FAST_func_spectrum.py, FASTfunc_remove.py)
Here, we take 7815-12705 from FAST project PT2022_0091 as an example.

how to run it:
$ python FAST_onoff_main.py xx ########### where xx is the index of the source
or change the number of "indgal"
then run 
$ python FAST_onoff_main.py
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
import FASTfunc_basic as fast_basic
import FASTfunc_onoff as fast_onoff
import FASTfunc_calibration as fast_cal
import FASTfunc_spectrum as fast_sp
import FASTfunc_remove as fast_remove

#%%
#####################################################################
################################# (1) collect all information we need
#####################################################################
######## get the galaxy index, file path, and rest frequency 
######## thus the code should be run in the terminal as "$ python FAST_onoff_pipeline.py 0", 
######## where 0 in the galaxy number in the list file
try:
    indgal = int(sys.argv[1])
except:
    indgal = 0
    print("The default source order is", str(indgal))
path0 = os.path.dirname(__file__)
######### the file name and path of the source list. 
######### It saves the observational settings, including the information of each object
file_source_list = path0+"/observation_2022.txt"
######### get the galayx name, central frequency, observation date, KY date, and integration time
name, freqc, date, dateKY, inttime, N_cycle = fast_basic.get_src(file_source_list, indgal)
year = date[0:4]
V_, V_c0, V_ = fast_basic.get_Freq2Vel(freqc)
print(name, freqc, V_c0, date, dateKY, inttime)
######### the file name and path of the paths
######### it saves the path of the original FAST raw data and KY files
##################################################################################################################################################################
###################################################### you should edit the file path based on your conditions ####################################################
##################################################################################################################################################################
file_path = path0+"/onoff_2022.par"
print(path0)
######### get the file path, data path 
outpath, datapathprefix, dataKY, projectname = fast_basic.get_path(file_path)
# print(outpath, datapathprefix, dataKY, projectname)
##################################################################################################################################################################
###################################################### our observation setting is shown as 7815_12705_FASTobservation.jpg. #######################################
###################################################### you should edit the observation settings accordingly ######################################################
##################################################################################################################################################################
beams, samptime, switchtime, CALon_time, CALon_factor = fast_basic.get_observationSetting()
print(beams, samptime, switchtime, CALon_time, CALon_factor)
freq_min = freqc - 50
freq_max = freqc + 50
######### save the file into one individual filefolder for each source
outpath_src = outpath+str(indgal)+"_"+"manga"+name+"/"
time1=time.time()
########## the path of calibration file, namely "highcal_20201014_W_tny.npz"
cal_path = path0+"/"
#%%
###########################################################################################
################################# (2) separate the data of source on and source off
###########################################################################################
######## then write the results into fits file
datapath = datapathprefix +  name + '/'+date+'/'
if os.path.exists(datapath) == False:
    datapath = datapathprefix +  name + '/'
print('Processing ',name, 'of project '+ projectname)
print(datapath)

for beam in beams:
    outnam = outpath_src + name + '_' + date + '_' + beam
    if not os.path.exists(outpath_src):
        os.makedirs(outpath_src)
    if os.path.isfile(outnam + '_onoffALL.fits'):
        pass
    else:
        freq, fluxall, mjdall = fast_onoff.read_onoff_files(datapath,beam,outnam)
        freq, flux_on, flux_off, mjd_on, mjd_off = fast_onoff.separate_onoff(freq, fluxall, mjdall, beam,samptime,switchtime,inttime, outnam,outnam)

# ###########################################################################################
# ################################# (3) convert the data to temperature
# ###########################################################################################
file_onoff = outpath_src+name + '_'+date+'*_onoff.fits'
files_onoff = glob.glob(file_onoff)
files_onoff.sort()
# print("The files after separating the source on and source off are:", files_onoff)
for index, files in enumerate(files_onoff):
    print(index+1, '/',len(files_onoff), files)
    fits_file2 = files.replace("_onoff.fits", "_temp.fits")
    fast_cal.fits1_fits2(files, freq_min, freq_max, CALon_factor, plot_fig = files, write_fits = files, cal_path= cal_path)

# # ###########################################################################################
# # ################################# (4) convert the temperature to mJy, using the KY files
# # ###########################################################################################
file_temp = outpath_src+name + '_'+date+'*_temp.fits'
files_temp = glob.glob(file_temp)
files_temp.sort()
print(files_temp)
for index, files in enumerate(files_temp):
    print(index+1, '/',len(files_temp), files)
    plot_fig = files.replace(".fits", ".pdf")
    write_txt = files.replace(".fits", ".txt")
    pd_cal = fast_cal.get_mJyVel(files, dataKY, dateKY, name, freqc, plot_fig, write_txt)

# ###################################################################################################################
# ################################# (5) for each final spectrum of one cycle, remove the ripple and baseline
# ###################################################################################################################
file_txt = outpath_src+name + '_'+date+'*_temp.txt'
files_txt = glob.glob(file_txt)
files_txt.sort()
vel_mask_arr = fast_remove.get_mask(name, year, N_cycle)
print("step1, the mask is:", vel_mask_arr)
for index, files in enumerate(files_txt):
    # print(index+1, '/',len(files_txt), files)
    vel_l = vel_mask_arr[index, 0]
    vel_r = vel_mask_arr[index, 1]
    vel_ls = V_c0-300
    vel_rs = V_c0+300
    deg_arr=[1, 2, 3]
    save_txt = files.replace("_temp.txt", "_sp.txt")
    plot_fig = files.replace("_temp.txt", "_sp.pdf")
    fast_sp.get_finalSP(files, V_c0, vel_l, vel_r, vel_ls, vel_rs, deg_arr, save_txt , plot_fig, deltaV=None, guess_params=None)
    
# ###################################################################################################################
# ################################# (6) stacking the spectra to derive the final spectrum and mask the range
# ################################################################################################################### 
file_sp = outpath_src+name + '_'+date+'*_sp.txt'
files_sp = glob.glob(file_sp)
files_sp.sort()

stringRemove = fast_remove.get_stringRemove(name, year)
files_sp, vel_mask_arr = fast_remove.get_cycle(files_sp, vel_mask_arr, stringRemove)
print("The final stacking file is:", files_sp, vel_mask_arr)
for i in range(len(stringRemove)):
    files_sp = [ x for x in files_sp if stringRemove[i] not in x ]
file_csv = outpath_src+name + '_'+date+ "_final.csv"
plot_fig = outpath_src+name + '_'+date+ "_final.pdf"
rms_arr, weight_arr, dd_final = fast_sp.stacking_sp(files_sp, vel_mask_arr, V_c0, plot_fig = plot_fig , file_csv= file_csv)


