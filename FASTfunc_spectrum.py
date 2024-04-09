"""
coder: Niankun Yu @ 2022.12.13
This file is totally written by myself
remove the ripple and baseline

(1) cut the range of spectra
(2) mask the spectra
(3) remove the ripple (compare the chi square)
(4) remove the baseline (compare the chi square)
(5) stacking the spectrum of each cycle and derive the final spectrum
"""
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import astropy
import io, os, glob
from astropy.io import fits, ascii
from astropy.stats import sigma_clip
from astropy.table import Table
from scipy.optimize import curve_fit, bisect
from scipy.interpolate import interp1d
from scipy import stats
from astropy import constants as const
from scipy.stats import chi2_contingency
import FASTfunc_basic as fast_basic

def cut_sp(dd0, V_c0, deltaV=None):
    """
    coder: Niankun Yu @ 2022.11
    For a given dataframe, we only take the range centered at V_c0+-deltaV

    Input
    --------
    dd0: dataframe, basically results from read_ascii

    Output
    --------
    dd0_cut: dataframe, dd0[(dd0["velocity"]>V_c0-deltaV) & (dd0["velocity"]<V_c0+deltaV)]
    """
    if deltaV == None:
        deltaV = 1500
    dd_cut = dd0[(dd0["velocity"]>=V_c0-deltaV) & (dd0["velocity"]<=V_c0+deltaV)].reset_index(drop=True)
    if len(dd_cut) ==0:
        os.sys.exit("Your dataframe cutting is wrong, check the central velocity V_c0")
    else:
        return dd_cut

def mask_sp(dd0, vel_l, vel_r):
    """
    coder: Niankun Yu @ 2022.12.13
    mask the spectrum based on the range of vel_l, vel_r

    Input
    -----------
    dd0: pandas dataframe, such as pd.DataFrame({"velocity":xx, "flux":yy})
    vel_l, vel_r: velocity array, we will mask from vel_l[i] to vel_r[i]

    Output
    -----------
    dd_mask: the masked spectrum (dropping the masked region)
    """
    if (np.any(vel_l) == None) or (np.any(vel_r) == None):
        return dd0
    length = len(vel_l)
    length1 = len(vel_r)
    assert length == length1
    ######## the index we want to drop
    dropIdx = []
    for i in range(length):
        dfdrop = dd0[(dd0["velocity"]>vel_l[i]) & (dd0["velocity"]<vel_r[i])]
        idx = dfdrop.index.values
        dropIdx.extend(idx)
    dd1 = dd0.drop(dd0.index[dropIdx])
    dd_mask = dd1.reset_index(drop=True)
    return dd_mask

def mask_signal(dd0, vel_ls, vel_rs):
    """
    mask the signal range of the spectrum
    """
    dfdrop = dd0[(dd0["velocity"]>vel_ls) & (dd0["velocity"]<vel_rs)]
    idx = dfdrop.index.values
    dd1 = dd0.drop(dd0.index[idx])
    dd0_noSignal = dd1.reset_index(drop=True)
    return dd0_noSignal

# def get_deg(vel0_fit, flux0_fit, deg_arr = [1, 2, 3]):
#     """
#     use the standard deviation to give the best value of deg
#     coder: Niankun Yu @ 2022.12.15

#     Input
#     -----------
#     vel0_fit, flux0_fit: velocity and flux array after masking the RFI and signal

#     """
#     from scipy.stats import chi2_contingency
#     std = []
#     for i in range(len(deg_arr)):
#         deg_i = deg_arr[i]
#         try:
#             pfit_i = np.poly1d(np.polyfit(vel0_fit, flux0_fit, deg_i))
#         except:
#             pfit_i = np.poly1d(np.asarray([0]*deg_i))
#         flux0_baseline = pfit_i(vel0_fit)
#         flux = flux0_fit - flux0_baseline
#         std_i = np.std(flux)
#         std.append(std_i)
#     index = np.argmin(std)
#     deg_best = deg_arr[index]
#     print("The best polynomial degree of the baseline is:", deg_best)
#     return deg_best

def get_deg(vel0_fit, flux0_fit, deg_arr = [1, 2, 3]):
    """
    use the standard deviation to give the best value of deg
    coder: Niankun Yu @ 2022.12.15

    Input
    -----------
    vel0_fit, flux0_fit: velocity and flux array after masking the RFI and signal

    """
    bic = []
    for i in range(len(deg_arr)):
        deg_i = deg_arr[i]
        try:
            pfit_i = np.poly1d(np.polyfit(vel0_fit, flux0_fit, deg_i))
        except:
            pfit_i = np.poly1d(np.asarray([0]*deg_i))
        flux0_baseline = pfit_i(vel0_fit)
        bic_i = fast_basic.get_bic(flux0_fit, deg_i+1, flux0_baseline)
        bic.append(bic_i)
    index = np.argmin(bic)
    deg_best = deg_arr[index]
    print("The best polynomial degree of the baseline is:", deg_best)
    return deg_best

def sp_baseline(dd0, vel_ls, vel_rs, deg_arr=[1, 2, 3], dd0_mask = None):
    """
    coder: Niankun Yu @ 2022.12.13
    remove the baseline, 

    Input
    -----------
    dd0: pandas dataframe, such as pd.DataFrame({"velocity":xx, "flux":yy})
    deg: degree of polynominal fitting to the baseline
    dd_mask: the masked spectrum (dropping the masked region)
    vel_ls, vel_rs: [vel_ls, vel_rs] roughly define the signal range

    Output
    -----------
    dd_baseline: pandas dataframe after removing the baseline

    """
    vel0 = dd0.velocity
    flux0 = dd0.flux
    if dd0_mask is not None:
        ########## remove the signal range
        dd0_noSignal = mask_signal(dd0_mask, vel_ls, vel_rs)
    else:
        dd0_noSignal = mask_signal(dd0, vel_ls, vel_rs)
    vel0_fit = dd0_noSignal.velocity
    flux0_fit  = dd0_noSignal.flux
    deg_best = get_deg(vel0_fit, flux0_fit, deg_arr)
    try:
        pfit = np.poly1d(np.polyfit(vel0_fit, flux0_fit, deg_best))
    except:
        pfit = np.poly1d(np.asarray([0]*deg_best))
    flux0_baseline = pfit(vel0)
    flux = flux0 - flux0_baseline
    dd_baseline = dd0.copy()
    dd_baseline["flux"] = flux
    return dd_baseline, flux0_baseline

def sin_ripple(x, guess_params):
    """
    coder: Niankun Yu @ 2022.12.14
    return a function as a format of sin function

    """
    guess_amp, guess_vel, guess_phase, guess_mean = guess_params
    return guess_amp*np.sin(guess_vel*x+guess_phase)+guess_mean

def fit_ripple(xx, yy, guess_params = None):
    """
    coder: Niankun Yu @ 2022.12.14
    fit the spectra with the Gaussian function
    """
    from scipy.optimize import leastsq
    if guess_params == None:
        guess_mean = np.nanmedian(yy)
        ########### fit in the velocity space
        guess_vel = 1.0/40
        guess_phase = 0
        guess_amp = 10
        guess_params = [guess_amp, guess_vel, guess_phase, guess_mean]
    else:
        guess_amp, guess_vel, guess_phase, guess_mean = guess_params
    best_func = lambda guess_params: sin_ripple(xx, guess_params)-yy
    est_amp, est_vel, est_phase, est_mean = leastsq(best_func, [guess_params])[0]
    est_params = [est_amp, est_vel, est_phase, est_mean]
    return est_params

def remove_ripple(xx, yy, guess_params = None):
    """
    coder: Niankun Yu @ 2022.12.14
    remove the fitted ripple if the chisquare becomes smaller

    """
    est_params = fit_ripple(xx, yy, guess_params)
    yy_fit = sin_ripple(xx, est_params)
    yy_rp = yy - yy_fit
    ########## compare the chisqaure, to determined the final results
    std_0 = np.std(yy)
    std_1 = np.std(yy_rp)
    print("check remove_ripple", std_0, std_1)
    ########### now we compare the BIC
    bic_0 = fast_basic.get_bic(yy, 1)
    bic_1 = fast_basic.get_bic(yy_rp, 4+1)
    print("check remove_ripple", bic_0, bic_1)
    # bic_i = fast_basic.get_bic(flux0_fit, deg_i+1, flux0_baseline)
    if bic_0>bic_1:
    # if std_0>std_1:
        return yy_fit, yy_rp, est_params
    else:
        yy_fit = [0]*len(yy)
        yy_rp = yy
        est_params = [0, 0, 0, 0]
        return yy_fit, yy_rp, est_params

def sp_sigma(dd0, center=None, threshold=None, iters=None):
    """
    dd0=pd.DataFrame({"velocity":V, "flux":F}), it should be the spectrum after removing RFI channels
    center: the center we choose to get minor spectra from flux density<center, default=0
    threshold: threshold for sigma-clipping, default 3
    iters: iteration times, default 3
    
    return: mean and sigma of the Gaussian fitting (noise level), mirrored flux density

    Coder: Niankun Yu @ 2020
    Note: the dd0 should be a clean spectra, without contaminated by RFI
    """
    if center is None:
        center = 0
    if threshold is None:
        threshold=2
    if iters is None:
        iters=3
    flux = dd0.flux
    flux1 = flux[flux<=center]
    flux2 = flux1 +(center - flux1)*2
    flux_mirror = np.concatenate((flux1, flux2), axis=0)
    fluxNew = sigma_clip(flux_mirror, sigma=threshold, maxiters=iters)
    if sum(fluxNew) ==0:  ##### for mock spectra, the Gaussian fitting may fail because all values are 0 at signal-free part
        f_mean=0
        sigma=0
    else:
        f_mean, sigma = stats.norm.fit(fluxNew)  # get mean and standard deviation
    return f_mean, sigma, flux_mirror

def plot_sp(plot_fig, dd0, dd0_cut, dd1_cut, dd_final, flux_ripple1, flux_baseline1, V_c0, vel_l, vel_r):
    """
    coder: Niankun Yu @ 2022.12.15
    plot the spectrum, panel a -- the raw spectrum; panel b -- the cutted spectrum and its mask; 
    panel c -- the fitted ripple and baseline; panel d -- the final spectrum
    
    """

    plt.close()
    mpl.rcParams['figure.subplot.top']=0.98 
    mpl.rcParams['figure.subplot.bottom']=0.1
    mpl.rcParams['figure.subplot.left']=0.05
    mpl.rcParams['figure.subplot.right']=0.98
    mpl.rcParams['figure.subplot.hspace']=0.2
    mpl.rcParams['figure.subplot.wspace']=0.2
    fig = plt.figure(figsize=[12, 12])
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    ax1.step(dd0["velocity"], dd0["flux"], linewidth=1 , color="black", zorder=1, label = "Raw")

    vel = dd0_cut["velocity"]
    flux = dd0_cut["flux"]
    dd0_signal = dd0_cut[(dd0_cut["velocity"]>=V_c0-300) & (dd0_cut["velocity"]<=V_c0+300)]
    flux_signal = dd0_signal["flux"]
    y_min1 = np.nanmin(flux_signal)*1.2
    y_max1 = np.nanmax(flux_signal)*1.2
    dd_signal = dd_final[(dd_final["velocity"]>=V_c0-300) & (dd_final["velocity"]<=V_c0+300)]
    y_min2 = np.nanmin(dd_signal["flux"])*1.2
    y_max2 = np.nanmax(dd_signal["flux"])*1.2
    y_min = min(y_min1, y_min2)
    y_max = max(y_max1, y_max2)
    x_min = np.min(vel)
    x_max = np.max(vel)
    ax2.step(vel, flux, linewidth=1 , color="black", zorder=1, label = "Raw")
    try:
        for i in range(len(vel_l)):
            ax2.fill_between(np.linspace(vel_l[i], vel_r[i], 100), y_min, y_max, hatch = 'X', color="grey")
            ax3.fill_between(np.linspace(vel_l[i], vel_r[i], 100), y_min, y_max, hatch = 'X', color="grey")
            ax4.fill_between(np.linspace(vel_l[i], vel_r[i], 100), y_min, y_max, hatch = 'X', color="grey")
    except:
        print("No input mask!")


    ax3.step(vel, flux, linewidth=1 , color="black", zorder=1, label = "Raw")
    ax3.step(vel, flux_ripple1, linewidth=3 , color="green", zorder=1, label = "Ripple")
    ax3.step(dd1_cut["velocity"], dd1_cut["flux"], linewidth=1 , color="blue", zorder=1, label = r"Raw$-$Ripple")
    ax3.step(vel, flux_baseline1, linewidth=3, color="magenta", zorder=1, label = "Baseline")
    ax3.step(dd_final["velocity"], dd_final["flux"], linewidth=1 , color="red", zorder=1, label = r"Raw$-$Ripple$-$Baseline")
    ax3.hlines(0, x_min, x_max, color="black", linewidth=1, zorder=0)
    ax3.legend(loc= "upper left")

    ax4.step(dd_final["velocity"], dd_final["flux"], linewidth=1 , color="black", zorder=1, label = r"Final")

    ax_list = [ax1, ax2, ax3, ax4]
    for i in range(len(ax_list)):
        p = ax_list[i]
        p.minorticks_on()
        p.tick_params(axis='both', which='major', length=10, width=3., direction="in", labelsize=18)
        p.tick_params(axis="both", which="minor", length=5, width=2, direction="in")
        p.set_ylabel(r"Flux (mJy)", fontsize=22)
        p.set_xlabel(r"Velocity (km s$^{-1}$)", fontsize=22)
        if p != ax1:
            p.axis([V_c0-1500, V_c0+1500, y_min, y_max])

    plt.savefig(plot_fig, bbox_inches="tight")
    plt.close()

def get_finalSP(file_txt, V_c0, vel_l, vel_r, vel_ls, vel_rs, deg_arr=[1, 2, 3], save_txt = "", plot_fig= "", deltaV=None, guess_params=None):
    """
    get the final spectrum (remove the ripple and remove the baseline)
    coder: Niankun Yu @ 2022.12.15
    """
    dd0 = fast_basic.read_txtCal(file_txt)
    ########### remove the median value of each spectrum
    dd0["flux"] = dd0["flux"] - np.nanmedian(dd0["flux"])
    dd0_cut = cut_sp(dd0, V_c0, deltaV)
    dd0_mask = mask_sp(dd0_cut, vel_l, vel_r)
    dd0_masks = mask_signal(dd0_mask, vel_ls, vel_rs)
    flux_ripple, flux_rp, est_params = remove_ripple(dd0_masks["velocity"], dd0_masks["flux"], guess_params)
    dd1_cut = dd0_cut.copy()
    vel1 = dd1_cut["velocity"]
    flux1 = dd1_cut["flux"]
    flux_ripple1 = sin_ripple(vel1, est_params)
    flux_NOripple1 = flux1 - flux_ripple1
    dd1_cut["flux"] = flux_NOripple1
    dd1_mask = mask_sp(dd1_cut, vel_l, vel_r)
    dd_final, flux_baseline1 = sp_baseline(dd1_cut, vel_ls, vel_rs, deg_arr, dd1_mask)
    if len(save_txt)>0:
        dd_final.to_csv(save_txt, sep = ",")
    if len(plot_fig)>0:
        plot_sp(plot_fig, dd0, dd0_cut, dd1_cut, dd_final, flux_ripple1, flux_baseline1, V_c0, vel_l, vel_r)
    return dd_final

def get_finalTXT(file_txt, V_c0, vel_l, vel_r, vel_ls, vel_rs, deg_arr=[1, 2, 3], save_txt = "", plot_fig= "", deltaV=None, guess_params=None):
    """
    get the final spectrum (do not remove the ripple and remove the baseline)
    coder: Niankun Yu @ 2024.01.18
    for MaNGA7975-6104
    """
    dd0 = fast_basic.read_txtCal(file_txt)
    ########### remove the median value of each spectrum
    dd0["flux"] = dd0["flux"] - np.nanmedian(dd0["flux"])
    dd0_cut = cut_sp(dd0, V_c0, deltaV)
    dd0_mask = mask_sp(dd0_cut, vel_l, vel_r)
    dd0_masks = mask_signal(dd0_mask, vel_ls, vel_rs)
    # flux_ripple, flux_rp, est_params = remove_ripple(dd0_masks["velocity"], dd0_masks["flux"], guess_params)
    # dd1_cut = dd0_cut.copy()
    # vel1 = dd1_cut["velocity"]
    # flux1 = dd1_cut["flux"]
    # flux_ripple1 = sin_ripple(vel1, est_params)
    # flux_NOripple1 = flux1 - flux_ripple1
    # dd1_cut["flux"] = flux_NOripple1
    # dd1_mask = mask_sp(dd1_cut, vel_l, vel_r)
    # dd_final, flux_baseline1 = sp_baseline(dd1_cut, vel_ls, vel_rs, deg_arr, dd1_mask)
    if len(save_txt)>0:
        dd0_cut.to_csv(save_txt, sep = ",")
    if len(plot_fig)>0:
        plot_sp(plot_fig, dd0, dd0_cut, dd0_cut, dd0_cut, [0]*len(dd0_cut), [0]*len(dd0_cut), V_c0, vel_l, vel_r)
    return dd0_cut

########################### stack the spectra by its rms
def read_sp(file_sp):
    """
    coder: Niankun Yu @ 2022.12.19
    read the sp file for each cycle, order the dataframe by velocity
    """
    name_sp = ("num", "frequency", "velocity", "flux", "weight")
    df_sp0 = pd.read_csv(file_sp, names = name_sp, skiprows = 1, sep = ",") 
    df_sp = df_sp0.sort_values(by=['velocity'], ascending=True)
    # print("read_sp: the dataframe is", df_sp)
    return df_sp

def get_weights(rms_arr):
    """
    coder: Niankun Yu @ 2022.12.19
    get the weights for each spectrum, based on the rms of the spectrum
    """
    weight0 = 0
    for i in range(len(rms_arr)):
        if rms_arr[i]!= np.nan:
            weight0 = weight0+1.0/(rms_arr[i]**2)
    
    weight_arr = []
    for i in range(len(rms_arr)):
        if rms_arr[i]!= np.nan:
            weight_i = 1.0/(rms_arr[i]**2)*1.0/weight0
        else:
            weight_i = 0 
        weight_arr.append(weight_i)
    return weight_arr

def resample_sp(vel1, flux1, vel2):
    """
    coder: Niankun Yu @ 2022.12.19
    resample the spectrum (vel1, flux1) by using the velocity array vel2
    """
    flux2 = np.interp(vel2, vel1, flux1)
    return flux2

def stacking_plot(plot_fig, files_sp, vel_mask_arr, V_c0, dd_final):
    """
    coder: Niankun Yu @ 2022.12.20
    plot the spectra after stacking ()
    """
    vel_final = dd_final["velocity"]
    flux_final = dd_final["flux"]
    dd_signal = dd_final[(dd_final["velocity"]>V_c0-300) & (dd_final["velocity"]<V_c0+300)]
    flux_signal = dd_signal["flux"]
    y_min = 1.2*np.nanmin(flux_signal)
    y_max = 1.4*np.nanmax(flux_signal)
    plt.close()
    mpl.rcParams['figure.subplot.top']=0.98 
    mpl.rcParams['figure.subplot.bottom']=0.1
    mpl.rcParams['figure.subplot.left']=0.05
    mpl.rcParams['figure.subplot.right']=0.98
    mpl.rcParams['figure.subplot.hspace']=0.2
    mpl.rcParams['figure.subplot.wspace']=0.2
    fig = plt.figure(figsize=[12, 6])
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for i in range(len(files_sp)):
        dd_spi = read_sp(files_sp[i])
        cycle_i = fast_basic.get_cycle(files_sp[i])
        vel_l = vel_mask_arr[i, 0]
        vel_r = vel_mask_arr[i, 1]
        vel_spi = dd_spi["velocity"]
        flux_spi = dd_spi["flux"]
        flux_spi2 = np.interp(vel_final, vel_spi, flux_spi)
        ax1.step(vel_final, flux_spi2, linewidth =1, label = cycle_i, zorder=0)
        if np.any(vel_l) != None:
            for j in range(len(vel_l)):
                ax1.fill_between(np.linspace(vel_l[j], vel_r[j], 100), y_min, y_max, hatch = 'X', color="grey", zorder=1)
                ax2.fill_between(np.linspace(vel_l[j], vel_r[j], 100), y_min, y_max, hatch = 'X', color="grey", zorder=1)
    ax_list = [ax1, ax2]
    for i in range(len(ax_list)):
        p = ax_list[i]
        p.step(vel_final, flux_final, linewidth =2, color= "red", zorder=3, label = "Final")
        p.hlines(0, V_c0-1500, V_c0+1500, color="black", linewidth=1, zorder=0)
        p.vlines(V_c0, y_min, y_max, color="blue", linewidth=3, zorder=3)
        p.minorticks_on()
        p.tick_params(axis='both', which='major', length=10, width=3., direction="in", labelsize=18)
        p.tick_params(axis="both", which="minor", length=5, width=2, direction="in")
        p.set_ylabel(r"Flux (mJy)", fontsize=22)
        p.set_xlabel(r"Velocity (km s$^{-1}$)", fontsize=22)
    ax1.axis([V_c0-1500, V_c0+1500, y_min, y_max])
    ax2.axis([V_c0-500, V_c0+500, y_min, y_max])
    ax1.legend(loc= "upper left", fontsize=14, ncol=2)
    plt.savefig(plot_fig, bbox_inches="tight")
    plt.close()

def stacking_sp(files_sp, vel_mask_arr, V_c0, plot_fig = "", file_csv = ""):
    """
    coder: Niankun Yu @ 2022.12.19
    stacking the given spectra to derive the final spectra and mask range
    """
    rms_arr = []
    for i in range(len(files_sp)):
        dd_spi = read_sp(files_sp[i])
        vel_l = vel_mask_arr[i, 0]
        vel_r = vel_mask_arr[i, 1]
        # print("The velocity mask is:", vel_l, vel_r)
        vel_ls = V_c0-300
        vel_rs = V_c0+300
        ddi_mask = mask_sp(dd_spi, vel_l, vel_r)
        ddi_masks = mask_signal(ddi_mask, vel_ls, vel_rs)
        f_mean, rms_i, flux_mirror = sp_sigma(ddi_masks)
        rms_arr.append(rms_i)
        ############ get the final velocity array
        if i == 0:
            len_min = len(dd_spi)
            vel_final = dd_spi["velocity"]
        elif len(dd_spi) < len_min:
            len_min = len(dd_spi)
            vel_final = dd_spi["velocity"]
    rms_arr = np.asarray(rms_arr)
    weight_arr = get_weights(rms_arr)
    flux_final = np.zeros(len_min)
    for i in range(len(files_sp)):
        dd_spi = read_sp(files_sp[i])
        vel_spi = dd_spi["velocity"]
        flux_spi = dd_spi["flux"]
        ############## the x array should be ascending
        flux_spi2 = np.interp(vel_final, vel_spi, flux_spi)
        flux_final = flux_final+weight_arr[i]*flux_spi2
    dd_final = pd.DataFrame({"velocity": vel_final, "flux": flux_final}, dtype= float)
    ############## plot the final figure:
    if len(plot_fig)>0:
        stacking_plot(plot_fig, files_sp, vel_mask_arr, V_c0, dd_final)
    ############## write the final spectra into csv final
    if len(file_csv)>0:
        astro_tab = Table.from_pandas(dd_final)
        ascii.write(astro_tab, file_csv, overwrite=True)
    return rms_arr, weight_arr, dd_final






