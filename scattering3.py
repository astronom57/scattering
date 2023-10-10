#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:28:44 2023


Fitting data for the RadioAstron scattering project
New approach with model containing only Gaussians and refractive noise being calculated numerically

@author: lisakov
"""
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import platform
import asyncio

import ehtim as eh
import ehtim.scattering as so
import pathos.multiprocessing as mp


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 140)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.markers import MarkerStyle
import mplcursors


import seaborn as sns

from scipy.optimize import curve_fit

import sys
sys.path.append('..')
from tb import tb, tb_err


def add_weights(dataframe, bins=11, doplot=True):
    """Bin data in UV plane. For this a rectangulag grid is overlaid onto the UV plot and number of points is counted in each cell. 
    This binning is visualized as as 2d histogram. The weight is a number of points in a cell
    """
    df = dataframe.copy()
    df_minus = dataframe.copy()
    df_minus['x'] = -df_minus['x'] 
    df_minus['y'] = -df_minus['y'] 
    df = pd.concat([df, df_minus])
    df.reset_index(drop=True, inplace=True)
    
    H, xedges, yedges = np.histogram2d(df.x.values, df.y.values, bins=bins)
    
    if doplot:
        fig, ax = plot_uv(df, interactive=False, random=False)
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X*1e-6, Y*1e-6, H.T, zorder=0, cmap='binary', edgecolors='k')
    
    for i in df.index:
        x = df.loc[i, 'x']
        y = df.loc[i, 'y']
        for j in range(xedges.size-1):
            for k in range(yedges.size-1):
                if (xedges[j] <= x < xedges[j+1]) and (yedges[k] <= y < yedges[k+1]):
                    # print(f'found a point idx = {i}. j={j}, k={k}. H = {H[j, k]}')
                    df.loc[i, 'weight'] =  1 / H[j, k]
    return df
    
    
def selfcal(dataframe, exper, band, ampl=1., hwhm=100e6, dofit=True, 
            trusted=['EF', 'RA', 'GB', 'AR', 'HH', 'MC', 'NT', 'YS'],
            corrections=None):
    """Use a snapshot dataframe (one experiment) to perform amplitude self-cal to a single Gaussian.
    Plot all together.
    
    Args:
        dataframe: dataframe which can be considered a snapshot (simultaneous data with a single global amplitude correction per telescope).
        exper: experiment code
        band: band (C, L)
        ampl: Gaussian model amplitude [Jy]
        hwhm: Gaussian model half-width-at-half-max [lambda]
        trusted: list of trusted telescopes to build a model
        corrections: hash of amplitude corrections {telescope}{polar} = correction
    """
    def g(x, a, b):
        return a * np.exp(-x**2/ (2 * b**2))
    
    sigma = 2* hwhm / np.sqrt(8 * np.log(2))
    
    df = dataframe.copy()
    df = df.loc[(df.exper==exper) & (df.band==band), :]
    
    print(df)
    
    ax = plot_rad(df)
    
    
    # fit trusted
    df_fit = df.loc[(df.sta1.isin(trusted)) & (df.sta2.isin(trusted)), : ] # leave only trusted antennas to fit a model to 
    x = df_fit['base']
    y = df_fit['ampl']
    popt, pcov = curve_fit(g, x, y, p0=[ampl, sigma])
    
    x = np.linspace(*ax.get_xlim(), 300)
    y = g(x, *popt)
    ax.plot(x, y, '-')
    
    # make corrections
    if corrections is not None:
        for t in corrections.keys():
            dt = df.loc[(df.sta1 == t) | (df.sta2 == t), :]
            for p in corrections[t].keys():
                dt.loc[dt.polar == p, 'ampl'] = dt.loc[dt.polar == p, 'ampl'] * corrections[t][p]
                print(f'Applied correction factor {corrections[t][p]:.2f} to tel={t} polar={p}')
                print(dt)
            
            ax.plot(dt.base, dt.ampl, 'x', color='k', label='Corrected amplitudes')
    
    ax.legend()
    ax.get_figure().suptitle(exper)
    
    return df


def rl_ratio(dataframe, telescope='IR16'):
    """Check RR/LL ratio for a given telescope
    """
    if telescope in ('RA', 'WB'):
        return
    
    df = dataframe.copy()
    df = df.loc[(df.sta1==telescope) | (df.sta2==telescope), :]
    dp = pd.pivot(df, index=['exper', 'sta1', 'sta2'], columns='polar', values='ampl')
    
    dp.dropna(axis=0, inplace=True)
    dp['ratio'] = dp.LL / dp.RR
    dp.reset_index(inplace=True)
    print(dp)
    
    fig, ax = plt.subplots(1,1)
    sns.scatterplot(dp, x='exper', y='ratio', ax=ax)
    ax.tick_params(axis='x', rotation=60)
    fig.tight_layout()
    fig.suptitle(f'LL/RR ratio for telescope {telescope}')
    
    return dp


def vplot(dataframe, telescope='GB'):
    """Plot amplitude on all baselines with telescope as a function of time.
    Ignore upper limits.
    """
    df = dataframe.copy()
    df = df.loc[df.uplim == 0, :]
    df = df.loc[(df.sta1==telescope) | (df.sta2==telescope), :]
    df.loc[df.index, 'sta'] = df.loc[df.index, 'sta1'] + df.loc[df.index, 'sta2']
    df.sta = df.sta.str.replace(telescope, '')
    # dp = pd.pivot(df, values='ampl', index='time', columns=['sta', 'polar'])
    
    N = df.sta.unique().size
    
    fig, ax = plt.subplots(N,1 , sharex=True, figsize=[8, 20])
    
    for i, s in enumerate(df.sta.unique()):
        dfi = df.loc[df.sta==s, :]
        sns.scatterplot(dfi, x='time', y='base_ml', hue='ampl', ax=ax[i])
        for j, point in dfi.iterrows():
            ax[i].text(point['time'], point['base_ml'], str(point['exper']))
        ax[i].set_title(f'{telescope} - {s}')
        
    fig.tight_layout()
    return


def read_noise(filename):
    """Read refractive noise generated by the script by Michael Johnson aka refractive_noise_3mm.py.
    It produces noise in EW and NS directions separately. Just average them? Or take max? or EW only?.
    Data are read from a text file in a format:
        uv_distance    sigma_EW    sigma_NS
     
    ??? The noise is relative ???
    
    Args:
        filename: file to read from
    Returns:
        Pandas dataframe: r, sigma
    
    """
    df = pd.read_csv(filename, sep='\s', names=['uv_dist', 'sigma_ew', 'sigma_ns'], comment='#', engine='python')
    
    df.loc[:, 'sigma'] = df.sigma_ew
    df.drop(columns=['sigma_ew', 'sigma_ns'], inplace=True)
    return df

def read_data(source, band, version):
    """Reads FF data into a dataframe
    Args:
        source: 
            source name
        band:
            band (L or C)
        version:
            antab file version (1 or 2)
    Returns:
        pandas dataframe with the data
    """
    band = band.upper()
    file = f'/home/lisakov/data2/scattering/{source}_{band}_antab{version}.txt'
    df = pd.read_csv(file, sep='\s+')
    return df


def read_catalog(source, band):
    """Read data from the catalogue file.
    
    Args:
        source: source to read 
        band: band to read. Default= None == read all bands
        
    Returns:
        pandas dataframe. 
    """
    if platform.node() == 'loki':
        # file = '/home/lisakov/data2/scattering/RA_catalog_rags28+raks18el_2021-04-16.txt'
        file = '/home/lisakov/data2/scattering/RA_catalog_rags28_2023-09-22.txt'
    elif platform.node() == 'smth else':
        # file = '/home/lisakov/data/scattering/RA_catalog_rags28+raks18el_2021-04-16.txt'
        file = '/home/lisakov/data/scattering/RA_catalog_rags28_2023-09-22.txt'
    else:
        # file = '/home/lisakov/data/scattering/RA_catalog_rags28+raks18el_2021-04-16.txt'
        file = '/home/lisakov/data/scattering/RA_catalog_rags28_2023-09-22.txt'
        
    df = pd.read_csv(file, sep='\s+', comment='#', parse_dates={'datetime':['date', 'time']})
    df.rename(columns={'datetime': 'time'}, inplace=True)
    df = df.loc[df.source == source]
    if band is not None:
        band = band.upper()
        df = df.loc[df.band == band]
    
    df.loc[:, 'base'] = df.base_ml * 1e6  # baseline in lambdas
    return df


def read_nondet(source, band):
    """Read info on non-detection.
    
    Args:
        source: source to read 
        band: band to read. Default= None == read all bands
        
    Returns:
        pandas dataframe
    """
    if platform.node() == 'loki':
        file = '/home/lisakov/data2/scattering/RA_rags28_nondet_uplims_2021-05-12.txt'
    elif platform.node() == 'smth else':
        file = '/home/lisakov/data/scattering/RA_rags28_nondet_uplims_2021-05-12.txt'
    else:
        file = '/home/lisakov/data/scattering/RA_rags28_nondet_uplims_2021-05-12.txt'

    df = pd.read_csv(file, sep='\s+', comment='#', parse_dates=['start_time'])
    df = df.loc[df.b1950 == source]
    if band is not None:
        band = band.upper()
        df = df.loc[df.band == band]
        
    df.loc[:, 'base'] = df.base_ml * 1e6  # baseline in lambdas
    return df
    

def get_data(source, band, polar=None, uplim=True, wavelength=18, quality_control=True):
    """Wraps read_catalogue and read_nondet together. Unifies columns names etc.
    
    Args:
        source: source to read 
        band: band to read. Default= None == read all bands
        polar: polarisation. One of [RR,LL,RL,LR, parallel=RR+LL, cross=RL+LR, all=RR+LL+RL+LR]
        uplim: import upper limits too
        wavelength: wavelength in [cm] to show Earth diameter on the radplots
        quality_control: default True. Remove bad data by invoking data_qa()
        
    Returns:
        pandas dataframe with both detections and upper limits (with proper flags for the latter)
    
    """
    pols = []
    
    if polar.lower() == 'parallel':
        pols = ['RR', 'LL']
    elif polar.lower() == 'cross':
        pols = ['RL', 'LR']
    elif polar.lower() == 'all':
        pols = ['RR', 'LL', 'RL', 'LR']
    else:
        pols = [polar]
    
    df = read_catalog(source, band)
    df['uplim'] = 0
    columns_catalog = ['source', 'exper', 'time', 'band', 'polar', 'sta1', 'sta2', 'snr', 'base_ml', 'pa', 'ampl', 'ampl_err', 'uplim']
    df_short = df[columns_catalog]


    if uplim is True:
        dfu = read_nondet(source, band)
    
        dfu.drop(columns=['ampl'], inplace=True)
        dfu.rename(columns={'b1950':'source', 'exper_name':'exper', 'start_time':'time', 'sta':'sta2', 'upper_lim':'ampl'}, inplace=True)
        dfu['sta1'] = 'RADIO-AS'
        dfu['ampl_err'] = 0.0
        
        dfu['uplim'] = 1
        
        columns_nondet =  ['source', 'exper', 'time', 'band', 'polar', 'sta1', 'sta2', 'snr', 'base_ml', 'pa', 'ampl', 'ampl_err', 'uplim']
        dfu_short = dfu[columns_nondet]
    
        df = pd.concat([df_short, dfu_short])
        
    else:
        df = df_short
        
    df.loc[:, 'base'] = df.base_ml * 1e6 # megalambda to lambda
    
    
    
    if polar is not None:
        df = df.loc[df.polar.isin(pols)]
    
    
    # # change telescope names to short ones
    df['sta1'] = df['sta1'].map({'RADIO-AS':'RA', 'EFLSBERG':'EF', 'MEDICINA':'MC', 'TORUN':'TR', 'GBT-VLBA':'GB',
                                  'HARTRAO':'HH', 'ARECIBO':'AR', 'SVETLOE':'SV', 'WSTRB-07':'WB', 'IRBENE32':'IR', 
                                  'MOPRA':'MP', 'IRBENE16':'IR16', 'NOTO':'NT', 'YEBES40M':'YS', 'WARK30M':'WA', 
                                  'TIDBIN64': 'TD', 'BADARY':'BD', 'ZELENCHK':'ZC' })
    
    df['sta2'] = df['sta2'].map({'RADIO-AS':'RA', 'EFLSBERG':'EF', 'MEDICINA':'MC', 'TORUN':'TR', 'GBT-VLBA':'GB',
                                  'HARTRAO':'HH', 'ARECIBO':'AR', 'SVETLOE':'SV', 'WSTRB-07':'WB', 'IRBENE32':'IR', 
                                  'MOPRA':'MP', 'IRBENE16':'IR16', 'NOTO':'NT', 'YEBES40M':'YS', 'WARK30M':'WA', 
                                  'TIDBIN64': 'TD', 'BADARY':'BD', 'ZELENCHK':'ZC' })
    
    df['x'] = df.base * np.sin(np.deg2rad(df.pa))
    df['y'] = df.base * np.cos(np.deg2rad(df.pa))
    df['x_ml'] = df.base_ml * np.sin(np.deg2rad(df.pa))
    df['y_ml'] = df.base_ml * np.cos(np.deg2rad(df.pa))

    
    # tackle bad data
    if quality_control is True:
        df = data_qa(df)
    
    df.reset_index(drop=True, inplace=True)
    df['weight'] = 1
    
    return df


def data_qa(dataframe):
    """Perform actions to correct data issues:
        Corrections are found manually by comparing fluxes at different baselines with close length and PA, 
        comparing amplitude in two polarisations, sidereal baselines, 'ampl self cal' etc. 
        Possible actions:
            flag: flag this data point
            XN: multiply amplitude by N
            check: do nothing, check later
        
        Args:
            dataframe: dataframe to modify
        Returns:
            processed dataframe
        """
    df = dataframe.copy()
    
    dq = pd.read_csv('/home/lisakov/data2/scattering/data_qa.csv', comment='#', engine='python', sep=',') # file with data QA assessment
    df = df.merge(dq.loc[:, ['exper', 'band', 'polar', 'sta1', 'sta2', 'action']], on=['exper', 'band', 'polar', 'sta1', 'sta2'])
    # flag data
    df = df.loc[df.action != 'flag', :]
    # correct amplitudes
    X_index = df.where(df.action.str.startswith('X')).dropna().index
    df.loc[X_index, 'ampl'] = df.loc[X_index, 'ampl'] * df.loc[X_index, 'action'].str.lstrip('X').astype(float)
    df.drop(columns=['action'], inplace=True)
    return df
 

def plot_uv(dataframe, colors=['tab:blue', 'lightgrey'], wavelength=18, hue:bool=True, random:bool=True, interactive:bool=True):
    """Plot uv-coverage
    
    Args:
        dataframe: pandas dataframe with columns base_ml, ampl, ampl_err, uplim
        colors: list of colors. [detections, non-detections]
        hue: colorize plot based on amplitudes
        random: add some scatter to points so they do not coinside completely
        interactive: connect events to show datapoint values
    """
    
    df = dataframe.copy()
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    x = df.base_ml * np.sin(np.deg2rad(df.pa))
    y = df.base_ml * np.cos(np.deg2rad(df.pa))
    x = np.append(x, -x)
    y = np.append(y, -y)
    
    df = pd.concat([df,df])
    
    df['x'] = x
    df['y'] = y
    
    if random is True: # add small random displacement to the points so they can be visually separated in the plot
        df['x'] = df['x'] * (1 + np.random.uniform(-0.02, 0.02, size=df.index.size))
        df['y'] = df['y'] * (1 + np.random.uniform(-0.02, 0.02, size=df.index.size))
    
    
    if hue==True:
        sc = sns.scatterplot(df, x='x' ,y='y', style='uplim', markers={1:'$\circ$', 0:'o'}, hue='ampl', ax=ax, zorder=20)
    else:
        sc = sns.scatterplot(df, x='x' ,y='y', style='uplim', markers={1:'$\circ$', 0:'o'}, ax=ax, zorder=20)
    # sc = ax.scatter(x, y)
    
    df['label'] = df['sta1'] + ' - ' + df['sta2'] +  '\n' + df['exper'] + '\n' + df['polar'] +' ' + ((df['ampl'] * 1000 ).astype(int) / 1000 ).astype(str) + ' Jy'
    labels = df['label'].values
    
    if interactive:
        mplcursors.cursor(ax, multiple=False, highlight=True, hover=False, annotation_kwargs={'visible':False})
        mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(labels[sel.index]))

    uv_1ED = 12800e-1/wavelength # 1 Earth Diameter in UV plane in [Mlambda]
    m = mpatches.Circle(((0,0)), uv_1ED, fc="none", ec='k', ls='--')
    ax.add_artist(m)

    ax.set_xlabel(r'u [M$\lambda$]')
    ax.set_ylabel(r'v [M$\lambda$]')
    ax.set_aspect(1)
    
    return fig, ax


def plot_rad(dataframe, colors=['tab:blue', 'lightgrey'], wavelength=18, connect=False, colorize=False, errors=False):
    """make a radplot with errors and upper limits.
    
    Args:
        dataframe: pandas dataframe with columns base_ml, ampl, ampl_err, uplim
        colors: list of colors. [detections, non-detections]
        wavelength: in [cm] to properly show where the Earth ends 
        connect: connect points for the same baseline
        errors: plot errorbars as 1/weight if true
    """
    
    df = dataframe.copy()
    df.loc[:, 'sta12'] = df['sta1'] + ' - ' + df['sta2']

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    
    xcol = 'base'
    ycol = 'ampl'
    ycol_err = 'ampl_err'
    
    if colorize is True:
        sns.scatterplot(df.assign(upper_limit=df.uplim.map({0: "Detection", 1: "Upper limit"})), 
                        x=xcol, y=ycol,  hue='sta12', style='upper_limit', markers={"Detection":'o', "Upper limit":'$\circ$'})
    else:
        # sns.scatterplot(df, x=xcol, y=ycol,  style='uplim', markers={1:'$\circ$', 0:'o'})
        sns.scatterplot(df.assign(upper_limit=df.uplim.map({0: "Detection", 1: "Upper limit"})), 
                        x=xcol, y=ycol,  style='upper_limit', markers={"Detection":'o', "Upper limit":'$\circ$'})

        
        if errors:
            ax.errorbar(df.base, df.ampl, 0.01 / df.weight**0.5, fmt='o')
    
    ax.set_yscale('log')
    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('Visibility amplitude [Jy]')
    
    # interactive labeling
    df['label'] = df['sta1'] + ' - ' + df['sta2'] +  '\n' + df['exper'] + '\n' + df['polar'] +' ' + ((df['ampl'] * 1000 ).astype(int) / 1000 ).astype(str) + ' Jy'

    labels = df['label'].values
    mplcursors.cursor(ax, multiple=False, highlight=True, hover=False, annotation_kwargs={'visible':False})
    mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(labels[sel.index]))

    uv_1ED = 12800e5/wavelength # 1 Earth Diameter in UV plane in [lambda]
    ax.axvline(uv_1ED, 0, 1, ls='--', c='k')


    if connect is True:
        for s in df.sta12.unique():
            dfs = df.loc[(df.sta12==s) & (df.uplim==0), :] # no need to connect upperlimits
            ax.plot(dfs[xcol], dfs[ycol], '-')
            
    return ax
    
       
class Model1D():
    """1D model for radplot as a collection of Gaussians
    """
    
    def __init__(self, wavelength=18, theta_scatt_1ghz=1.0, theta_scatt=None):
        """Make new instance.
        
        Args:
            wavelength: wavelength in [cm]
            theta_scatt_1ghz: size of scattered point source at 1 ghz in [mas]. It is assumed that 
                            this size scales as -2 power of frequency. For the actual 
                            wavelength it is calculated as theta_scatt = theta_scatt_1ghz * (1e9 / freq[Hz])**2
            theta_scatt: overrides calculated theta_scatt. In [mas]. No size in the image plane can be smaller
        
        Args:
            wavelength: wavelength in [cm]
        """
        self.NTHREADS = 10 # number of threads for multithreading
        self.C = np.sqrt(8*np.log(2)) # constant
        self.MAS2RAD = 1 / 206265000 # mas to radians
        self.LAM2MLAM = 1e-6 # lambda to megalambda
        
        
        self.source = 'SOURCE'
        self.wavelength = wavelength  # wavelength in [cm]
        if 5 < self.wavelength < 7:
            self.band = 'C'
        elif 17 < self.wavelength < 19:
            self.band = 'L'
        
        self.frequency = eh.C / self.wavelength * 100 # [Hz]
        self.ncomp = 0
        
        
        
        # parameters of scattering
        self.D = 1 # in [kpc]
        if theta_scatt is not None:
            self.theta_scatt = theta_scatt # override any calculations of theta_scatt
        else:
            self.theta_scatt = theta_scatt_1ghz * (1e9 / self.frequency)**2
            
        # limit on visibility extent from theta_scatt
        try:
            self.max_sigma_V = 1 / (2 * np.pi * self.theta_scatt * self.MAS2RAD)
        except:
            self.max_sigma_V = np.inf
        
        
        
        self.p = pd.DataFrame(columns=['name', 'flux', 'sigma_V', 'sigma_I', 'fwhm_V', 'fwhm_I']) # dataframe with Gaussians. 
        # fwhm = self.C * sigma                [Jy],    [lambda]    [mas]    [lambda]   [mas]
        self.p2= pd.DataFrame(columns=['name', 'flux', # dataframe with 2D Gaussians. 
                                       'sigma_max_V', 'sigma_max_I', 'sigma_min_V', 'sigma_min_I',  # sigmas in both visibility and image domains
                                       'fwhm_max_V', 'fwhm_max_I', 'fwhm_min_V', 'fwhm_min_I',  # FWHMs in both visibility and image domains
                                       'phi_V', 'phi_I']) # angle between major axis and zero in vis and img planes
        self.pn= pd.DataFrame(columns=['name', 'flux', # dataframe with 2D Gaussians. 
                                       'sigma_major_V', 'sigma_major_I', 'axratio',  # sigmas major in both visibility and image domains and axratio
                                       'fwhm_major_V', 'fwhm_major_I',  # FWHMs in both visibility and image domains. axratio is the same 
                                       'phi_V', 'phi_I']) # angle between major axis and zero in vis and img planes
        
        
        self.func = lambda x: 0
        self.func2 = lambda u,v: 0
        self.funcn = lambda u,v: 0
        
        # fittable part of the model. To the moment is the same as the full model
        self.func_fit = None # compound function
        self.func2_fit = None # compound function
        self.funcn_fit = None # compound function
        self.par_fit = [] # values of all parameters in the right order, e.g. to supply into the fitting routine
        self.par2_fit = [] # values of all parameters in the right order, e.g. to supply into the fitting routine
        self.parn_fit = [] # values of all parameters in the right order, e.g. to supply into the fitting routine
        
        return
    
    def g1(self, ampl, sigma_V):
        return lambda x, a, s: a*np.exp(-x**2/(2*s**2))
    
    def g2(self, ampl1, sigma_V1, ampl2, sigma_V2):
        return lambda x, a1, s1, a2, s2 : a1*np.exp(-x**2/(2*s1**2)) + a2*np.exp(-x**2/(2*s2**2))
    
    def g3(self, ampl1, sigma_V1, ampl2, sigma_V2, ampl3, sigma_V3):
        return lambda x, a1, s1, a2, s2, a3, s3 : a1*np.exp(-x**2/(2*s1**2)) + a2*np.exp(-x**2/(2*s2**2)) + a3*np.exp(-x**2/(2*s3**2))

    def init_1g(self, flux, fwhm_I):
        """Initiate a model consisting of 1 circular Gaussian.
        
        Args:
            flux: component flux in [Jy]
            fwhm_I: component major axis in [mas]
        """
        self.ncomp = 1
        sigma_I = fwhm_I / self.C # [mas]
        sigma_V = 1 / (2 * np.pi * sigma_I * self.MAS2RAD) # [lambda]
        fwhm_V = self.C * sigma_V #[lambda]
        
        self.p = pd.concat([self.p, pd.DataFrame(data=[['g1', flux, sigma_V, sigma_I, fwhm_V, fwhm_I]], columns=['name', 'flux', 'sigma_V', 'sigma_I', 'fwhm_V', 'fwhm_I'])])
        self.func = self.g1(flux, sigma_V)
        self.func_fit = self.g1(flux, sigma_V)
        self.par_fit = [flux, sigma_V]
        return
    

    def init_2g(self, flux1, fwhm_I1, flux2, fwhm_I2):
        """Initiate a model consisting of 2 circular Gaussians.
        
        Args:
            flux: component flux in [Jy]
            fwhm_I: component major axis in [mas]
        """
        self.ncomp = 2
        sigma_I1 = fwhm_I1 / self.C # [mas]
        sigma_V1 = 1 / (2 * np.pi * sigma_I1 * self.MAS2RAD) # [lambda]
        fwhm_V1 = self.C * sigma_V1 #[lambda]
        sigma_I2 = fwhm_I2 / self.C # [mas]
        sigma_V2 = 1 / (2 * np.pi * sigma_I2 * self.MAS2RAD) # [lambda]
        fwhm_V2 = self.C * sigma_V2 #[lambda]
        
        self.p = pd.concat([self.p, pd.DataFrame(data=[['g1', flux1, sigma_V1, sigma_I1, fwhm_V1, fwhm_I1], ['g2', flux2, sigma_V2, sigma_I2, fwhm_V2, fwhm_I2]], 
                                                 columns=['name', 'flux', 'sigma_V', 'sigma_I', 'fwhm_V', 'fwhm_I'])])
        
        self.func = self.g2(flux1, sigma_V1, flux2, sigma_V2)
        self.func_fit = self.g2(flux1, sigma_V1, flux2, sigma_V2)
        self.par_fit = [flux1, sigma_V1, flux2, sigma_V2]
        return    


    def init_3g(self, flux1, fwhm_I1, flux2, fwhm_I2, flux3, fwhm_I3):
        """Initiate a model consisting of 2 circular Gaussians.
        
        Args:
            flux: component flux in [Jy]
            fwhm_I: component major axis in [mas]
        """
        self.ncomp = 3
        sigma_I1 = fwhm_I1 / self.C # [mas]
        sigma_V1 = 1 / (2 * np.pi * sigma_I1 * self.MAS2RAD) # [lambda]
        fwhm_V1 = self.C * sigma_V1 #[lambda]
        sigma_I2 = fwhm_I2 / self.C # [mas]
        sigma_V2 = 1 / (2 * np.pi * sigma_I2 * self.MAS2RAD) # [lambda]
        fwhm_V2 = self.C * sigma_V2 #[lambda]
        sigma_I3 = fwhm_I3 / self.C # [mas]
        sigma_V3 = 1 / (2 * np.pi * sigma_I3 * self.MAS2RAD) # [lambda]
        fwhm_V3 = self.C * sigma_V3 #[lambda]
        
        self.p = pd.concat([self.p, pd.DataFrame(data=[['g1', flux1, sigma_V1, sigma_I1, fwhm_V1, fwhm_I1], 
                                                       ['g2', flux2, sigma_V2, sigma_I2, fwhm_V2, fwhm_I2], 
                                                       ['g3', flux3, sigma_V3, sigma_I3, fwhm_V3, fwhm_I3]], 
                                                 columns=['name', 'flux', 'sigma_V', 'sigma_I', 'fwhm_V', 'fwhm_I'])])
        
        self.func = self.g3(flux1, sigma_V1, flux2, sigma_V2, flux3, sigma_V3)
        self.func_fit = self.g3(flux1, sigma_V1, flux2, sigma_V2, flux3, sigma_V3)
        self.par_fit = [flux1, sigma_V1, flux2, sigma_V2, flux3, sigma_V3]
        return    
                
 
    def g1_2d(self, flux, sigma_max_V, sigma_min_V, phi_V):
        """2D Gaussian in UV plane. The dimensiones are sigma_max, sigma_min. Sigma_max is oriented with an angle phi to somewhere."""
        # p= [ampl, sigma_max, sigma_min, phi]. Units are [Jy, lambda, lambda, degrees]
        return lambda u,v, a, sx,sy,t:  a*np.exp( -((((np.cos(np.deg2rad(t)))**2)/(2*sx**2) + ((np.sin(np.deg2rad(t)))**2)/(2*sy**2))*u**2 +
                                               2*(-(np.sin(2*np.deg2rad(t)))/(4*sx**2) + (np.sin(2*np.deg2rad(t)))/(4*sy**2))*u*v + 
                                               (((np.sin(np.deg2rad(t)))**2)/(2*sx**2) + ((np.cos(np.deg2rad(t)))**2)/(2*sy**2))*v**2))
        # return lambda u,v, *p:  p[0]*np.exp( -((((np.cos(np.deg2rad(p[3])))**2)/(2*p[1]**2) + ((np.sin(np.deg2rad(p[3])))**2)/(2*p[2]**2))*u**2 +
        #                                        2*(-(np.sin(2*np.deg2rad(p[3])))/(4*p[1]**2) + (np.sin(2*np.deg2rad(p[3])))/(4*p[2]**2))*u*v + 
        #                                        (((np.sin(np.deg2rad(p[3])))**2)/(2*p[1]**2) + ((np.cos(np.deg2rad(p[3])))**2)/(2*p[2]**2))*v**2))
        
    def g2_2d(self):
        """2D Gaussian in UV plane. The dimensiones are sigma_max, sigma_min. Sigma_max is oriented with an angle phi to somewhere."""
        # p= [ampl, sigma_max, sigma_min, phi]. Units are [Jy, lambda, lambda, degrees]
        return lambda u,v, a1,sx1,sy1,t1, a2,sx2,sy2,t2:  a1*np.exp( -((((np.cos(np.deg2rad(t1)))**2)/(2*sx1**2) + ((np.sin(np.deg2rad(t1)))**2)/(2*sy1**2))*u**2 +
                                               2*(-(np.sin(2*np.deg2rad(t1)))/(4*sx1**2) + (np.sin(2*np.deg2rad(t1)))/(4*sy1**2))*u*v + 
                                               (((np.sin(np.deg2rad(t1)))**2)/(2*sx1**2) + ((np.cos(np.deg2rad(t1)))**2)/(2*sy1**2))*v**2)) + \
                                               a2*np.exp( -((((np.cos(np.deg2rad(t2)))**2)/(2*sx2**2) + ((np.sin(np.deg2rad(t2)))**2)/(2*sy2**2))*u**2 +
                                               2*(-(np.sin(2*np.deg2rad(t2)))/(4*sx2**2) + (np.sin(2*np.deg2rad(t2)))/(4*sy2**2))*u*v + 
                                               (((np.sin(np.deg2rad(t2)))**2)/(2*sx2**2) + ((np.cos(np.deg2rad(t2)))**2)/(2*sy2**2))*v**2))
    def g3_2d(self):
        """2D Gaussian in UV plane. The dimensiones are sigma_max, sigma_min. Sigma_max is oriented with an angle phi to somewhere."""
        # p= [ampl, sigma_max, sigma_min, phi]x3. Units are [Jy, lambda, lambda, degrees]
        return lambda u,v, a1,sx1,sy1,t1, a2,sx2,sy2,t2,  a3,sx3,sy3,t3, :  a1*np.exp( -((((np.cos(np.deg2rad(t1)))**2)/(2*sx1**2) + ((np.sin(np.deg2rad(t1)))**2)/(2*sy1**2))*u**2 +
                                               2*(-(np.sin(2*np.deg2rad(t1)))/(4*sx1**2) + (np.sin(2*np.deg2rad(t1)))/(4*sy1**2))*u*v + 
                                               (((np.sin(np.deg2rad(t1)))**2)/(2*sx1**2) + ((np.cos(np.deg2rad(t1)))**2)/(2*sy1**2))*v**2)) + \
                                               a2*np.exp( -((((np.cos(np.deg2rad(t2)))**2)/(2*sx2**2) + ((np.sin(np.deg2rad(t2)))**2)/(2*sy2**2))*u**2 +
                                               2*(-(np.sin(2*np.deg2rad(t2)))/(4*sx2**2) + (np.sin(2*np.deg2rad(t2)))/(4*sy2**2))*u*v + 
                                               (((np.sin(np.deg2rad(t2)))**2)/(2*sx2**2) + ((np.cos(np.deg2rad(t2)))**2)/(2*sy2**2))*v**2)) + \
                                               a3*np.exp( -((((np.cos(np.deg2rad(t3)))**2)/(2*sx3**2) + ((np.sin(np.deg2rad(t3)))**2)/(2*sy3**2))*u**2 +
                                               2*(-(np.sin(2*np.deg2rad(t3)))/(4*sx3**2) + (np.sin(2*np.deg2rad(t3)))/(4*sy3**2))*u*v + 
                                               (((np.sin(np.deg2rad(t3)))**2)/(2*sx3**2) + ((np.cos(np.deg2rad(t3)))**2)/(2*sy3**2))*v**2))                                               
        
    @staticmethod
    def g1_2d_fit(uv, flux, sigma_max_V, sigma_min_V, phi_V):
        (u, v) = uv
        a = flux
        sx = sigma_max_V
        sy = sigma_min_V
        t = phi_V
        return a*np.exp( -((((np.cos(np.deg2rad(t)))**2)/(2*sx**2) + ((np.sin(np.deg2rad(t)))**2)/(2*sy**2))*u**2 +
                                               2*(-(np.sin(2*np.deg2rad(t)))/(4*sx**2) + (np.sin(2*np.deg2rad(t)))/(4*sy**2))*u*v + 
                                               (((np.sin(np.deg2rad(t)))**2)/(2*sx**2) + ((np.cos(np.deg2rad(t)))**2)/(2*sy**2))*v**2))


    def init_1g_2d_uv(self, flux, fwhm_max_V, fwhm_min_V, phi):
        """Test case. Initiate a model consisting of one 2D elliptical Gaussian.
        
        Args:
            flux: component flux in [Jy]
            fwhm_max: component major axis in UV plane [lambdas]
            """
        self.ncomp = 1
        sigma_max_V = fwhm_max_V / self.C # [lambdas]
        sigma_min_V = fwhm_min_V / self.C # [lambdas]
        
        # NO CONVERSION TO UV BECAUSE IT IS ALREADY SET IN UV PLANE
        sigma_max_I = 0
        sigma_min_I = 0
        fwhm_max_I = 0
        fwhm_min_I = 0
        phi_I = 0
        
        phi_V = phi
        
        self.p2 = pd.concat([self.p2, pd.DataFrame(data=[['g1', flux, 
                                                          sigma_max_V, sigma_max_I, sigma_min_V, sigma_min_I, 
                                                          fwhm_max_V, fwhm_max_I, fwhm_min_V, fwhm_min_I,
                                                          phi_V, phi_I]], 
                                                   columns=['name', 'flux', # dataframe with 2D Gaussians. 
                                                            'sigma_max_V', 'sigma_max_I', 'sigma_min_V', 'sigma_min_I',  # sigmas in both visibility and image domains
                                                            'fwhm_max_V', 'fwhm_max_I', 'fwhm_min_V', 'fwhm_min_I',  # FWHMs in both visibility and image domains
                                                            'phi_V', 'phi_I'])])
        
        self.func2 = self.g1_2d(flux, sigma_max_V, sigma_min_V, phi_V)
        self.func2_fit = self.g1_2d(flux, sigma_max_V, sigma_min_V, phi_V)
        self.par2_fit = [flux, sigma_max_V, sigma_min_V, phi_V]
        return
        
        

    def init_1g_2d(self, flux, fwhm_max_I, fwhm_min_I, phi_I):
        """Initiate a model consisting of one 2D elliptical Gaussian in image plane and make all conversions to the Visibility plane.
        
        Args:
            flux: component flux in [Jy]
            fwhm_max_I: component major axis in IMAGE plane [mas]
            fwhm_min_I: component minor axis in IMAGE plane [mas]
            phi_I: component major axis direction [degrees]
            """
        self.ncomp = 1
        
        sigma_max_I = fwhm_max_I / self.C # [mas]
        sigma_min_I = fwhm_min_I / self.C # [mas]
        
        sigma_max_V = 1 / (2 * np.pi * sigma_max_I * self.MAS2RAD) # [lambda]
        sigma_min_V = 1 / (2 * np.pi * sigma_min_I * self.MAS2RAD) # [lambda]
        fwhm_max_V = self.C * sigma_max_V
        fwhm_min_V = self.C * sigma_min_V
        phi_V = phi_I + 90
        
        
        self.p2 = pd.concat([self.p2, pd.DataFrame(data=[['g1', flux, 
                                                          sigma_max_V, sigma_max_I, sigma_min_V, sigma_min_I, 
                                                          fwhm_max_V, fwhm_max_I, fwhm_min_V, fwhm_min_I,
                                                          phi_V, phi_I]], 
                                                   columns=['name', 'flux', # dataframe with 2D Gaussians. 
                                                            'sigma_max_V', 'sigma_max_I', 'sigma_min_V', 'sigma_min_I',  # sigmas in both visibility and image domains
                                                            'fwhm_max_V', 'fwhm_max_I', 'fwhm_min_V', 'fwhm_min_I',  # FWHMs in both visibility and image domains
                                                            'phi_V', 'phi_I'])])
        
        self.func2 = self.g1_2d(flux, sigma_max_V, sigma_min_V, phi_V)
        self.func2_fit = self.g1_2d(flux, sigma_max_V, sigma_min_V, phi_V)
        self.par2_fit = [flux, sigma_max_V, sigma_min_V, phi_V]
        return
        
    def init_2g_2d(self, flux1, fwhm_max_I1, fwhm_min_I1, phi_I1,
                         flux2, fwhm_max_I2, fwhm_min_I2, phi_I2):
        """Initiate a model consisting of one 2D elliptical Gaussian in image plane and make all conversions to the Visibility plane.
        
        Args:
            flux: component flux in [Jy]
            fwhm_max_I: component major axis in IMAGE plane [mas]
            fwhm_min_I: component minor axis in IMAGE plane [mas]
            phi_I: component major axis direction [degrees]
            """
        self.ncomp = 2
        
        sigma_max_I1 = fwhm_max_I1 / self.C # [mas]
        sigma_min_I1 = fwhm_min_I1 / self.C # [mas]
        sigma_max_I2 = fwhm_max_I2 / self.C # [mas]
        sigma_min_I2 = fwhm_min_I2 / self.C # [mas]
        
        sigma_max_V1 = 1 / (2 * np.pi * sigma_max_I1 * self.MAS2RAD) # [lambda]
        sigma_min_V1 = 1 / (2 * np.pi * sigma_min_I1 * self.MAS2RAD) # [lambda]
        sigma_max_V2 = 1 / (2 * np.pi * sigma_max_I2 * self.MAS2RAD) # [lambda]
        sigma_min_V2 = 1 / (2 * np.pi * sigma_min_I2 * self.MAS2RAD) # [lambda]
        fwhm_max_V1 = self.C * sigma_max_V1
        fwhm_min_V1 = self.C * sigma_min_V1
        phi_V1 = phi_I1 + 90
        fwhm_max_V2 = self.C * sigma_max_V2
        fwhm_min_V2 = self.C * sigma_min_V2
        phi_V2 = phi_I2 + 90
        
        self.p2 = pd.concat([self.p2, pd.DataFrame(data=[['g1', flux1, 
                                                          sigma_max_V1, sigma_max_I1, sigma_min_V1, sigma_min_I1,
                                                          fwhm_max_V1, fwhm_max_I1, fwhm_min_V1, fwhm_min_I1,
                                                          phi_V1, phi_I1],
                                                         ['g2', flux2,
                                                          sigma_max_V2, sigma_max_I2, sigma_min_V2, sigma_min_I2,
                                                          fwhm_max_V2, fwhm_max_I2, fwhm_min_V2, fwhm_min_I2,
                                                          phi_V2, phi_I2]],
                                                   columns=['name', 'flux', # dataframe with 2D Gaussians. 
                                                            'sigma_max_V', 'sigma_max_I', 'sigma_min_V', 'sigma_min_I',  # sigmas in both visibility and image domains
                                                            'fwhm_max_V', 'fwhm_max_I', 'fwhm_min_V', 'fwhm_min_I',  # FWHMs in both visibility and image domains
                                                            'phi_V', 'phi_I'])])
        
        self.func2 = self.g2_2d()
        self.func2_fit = self.g2_2d()
        self.par2_fit = [flux1, sigma_max_V1, sigma_min_V1, phi_V1, flux1, sigma_max_V1, sigma_min_V1, phi_V1]
        return
        
    def init_3g_2d(self, flux1, fwhm_max_I1, fwhm_min_I1, phi_I1,
                         flux2, fwhm_max_I2, fwhm_min_I2, phi_I2,
                         flux3, fwhm_max_I3, fwhm_min_I3, phi_I3,):
        """Initiate a model consisting of one 2D elliptical Gaussian in image plane and make all conversions to the Visibility plane.
        
        Args:
            flux: component flux in [Jy]
            fwhm_max_I: component major axis in IMAGE plane [mas]
            fwhm_min_I: component minor axis in IMAGE plane [mas]
            phi_I: component major axis direction [degrees]
            """
        self.ncomp = 3
        flux = np.array([flux1, flux2, flux3])
        fwhm_max_I = np.array([fwhm_max_I1, fwhm_max_I2, fwhm_max_I3])
        fwhm_min_I = np.array([fwhm_min_I1, fwhm_min_I2, fwhm_min_I3])
        phi_I = np.array([phi_I1, phi_I2, phi_I3])
        sigma_max_I, sigma_min_I, sigma_max_V, sigma_min_V, fwhm_max_V, fwhm_min_V, phi_V = [], [], [], [], [], [], []
        
        
        for i in range(self.ncomp):
            sigma_max_I.append(fwhm_max_I[i] / self.C) # [mas]
            sigma_min_I.append(fwhm_min_I[i] / self.C) # [mas]
            
            sigma_max_V.append(1 / (2 * np.pi * sigma_max_I[i] * self.MAS2RAD)) # [lambda]
            sigma_min_V.append(1 / (2 * np.pi * sigma_min_I[i] * self.MAS2RAD)) # [lambda]
            fwhm_max_V.append(self.C * sigma_max_V[i])
            fwhm_min_V.append(self.C * sigma_min_V[i])
            phi_V.append(phi_I[i] + 90)
            
            self.p2 = pd.concat([self.p2, pd.DataFrame(data=[[f'g{i+1}', flux[i], 
                                                              sigma_max_V[i], sigma_max_I[i], sigma_min_V[i], sigma_min_I[i], 
                                                              fwhm_max_V[i], fwhm_max_I[i], fwhm_min_V[i], fwhm_min_I[i],
                                                              phi_V[i], phi_I[i]]], 
                                                       columns=['name', 'flux', # dataframe with 2D Gaussians. 
                                                                'sigma_max_V', 'sigma_max_I', 'sigma_min_V', 'sigma_min_I',  # sigmas in both visibility and image domains
                                                                'fwhm_max_V', 'fwhm_max_I', 'fwhm_min_V', 'fwhm_min_I',  # FWHMs in both visibility and image domains
                                                                'phi_V', 'phi_I'],
                                                       index=[i])])
        
            self.par2_fit.extend([flux[i], sigma_max_V[i], sigma_min_V[i], phi_V[i]])

        self.func2 = self.g3_2d()
        self.func2_fit = self.g3_2d()
        # self.par2_fit = [flux1, sigma_max_V1, sigma_min_V1, phi_V1, flux1, sigma_max_V1, sigma_min_V1, phi_V1]
        
        return



    def update_param_I2V(self, ref_cols=['fwhm_major_I', 'phi_I']):
        """Calculate parameters stored in self.pn from image plane to visibility plane. True values are store in ref_cols.
        
        Args:
            ref_cols: columns with reference values used to update ohter columns. Could be one of 
            [fwhm_major_I, phi_I] -> used when initializing a gaussian model
            [sigma_major_I, phi_I]
        """
        if 'fwhm_major_I' in ref_cols:
            self.pn.loc[:, 'sigma_major_I'] = self.pn.loc[:, 'fwhm_major_I'] / self.C # [mas]
        elif 'sigma_major_I' in ref_cols:
            self.pn.loc[:, 'fwhm_major_I'] = self.pn.loc[:, 'sigma_major_I'] * self.C # [mas]
        
        self.pn.loc[:, 'sigma_major_V'] = 1 / (2 * np.pi * self.pn.loc[:, 'sigma_major_I'] * self.pn.loc[:, 'axratio'] * self.MAS2RAD) # minor in Image plane translates to major in Vis plane. # [lambdas]
        self.pn.loc[:, 'fwhm_major_V'] = self.pn.loc[:, 'sigma_major_V'] * self.C # [lambdas]
        self.pn.loc[:, 'phi_V'] = self.pn.loc[:, 'phi_I'] + 90
        
        return

    def update_param_V2I(self, ref_cols=['sigma_major_V', 'phi_V']):
        """Calculate parameters stored in self.pn from Vis plane to Image plane. True values are store in ref_cols.
        
        Args:
            ref_cols: columns with reference values used to update ohter columns. Could be one of 
            [fwhm_major_V, phi_V] 
            [sigma_major_V, phi_V] -> used when updating parameters after fitting
        """
        
        if 'fwhm_major_V' in ref_cols:
            self.pn.loc[:, 'sigma_major_V'] = self.pn.loc[:, 'fwhm_major_V'] / self.C # [lambdas]
        elif 'sigma_major_V' in ref_cols:
            self.pn.loc[:, 'fwhm_major_V'] = self.pn.loc[:, 'sigma_major_V'] * self.C # [lambdas]
        
        self.pn.loc[:, 'sigma_major_I'] = 1 / (2 * np.pi * self.pn.loc[:, 'sigma_major_V'] * self.pn.loc[:, 'axratio'] * self.MAS2RAD) # minor in Vis plane translates to major in Image plane. # [mas]
        self.pn.loc[:, 'fwhm_major_I'] = self.pn.loc[:, 'sigma_major_I'] * self.C # [mas]
        self.pn.loc[:, 'phi_I'] = self.pn.loc[:, 'phi_V'] - 90
        
        return
        

        
    def make_noise(self, max_uv=2e9, samples:int=10, multithread=True,
                   theta_maj_mas_ref=1,
                   r_in=1e8,
                   r_out=150e11,
                   outfile=None):
        """ Take parameters of the fitted Gaussian(s) and calculate expected refractive noise.
        
        Args:
            max_uv: sample UV distances up to this value in [lambda]
            samples: how many realizations of noise phase screen to use
            theta_maj_mas_ref: in [mas] scattered image size of a point source aka theta_scatt
            r_in: inner scale in [cm]. Defaults to 100 km = 1e7 cm
            r_out: outer scale in [cm]. Defaults to 1 AU = 1.5e13 cm
            
        """
        
        # take the component with the highest flux. Get its measured size fwhm_I == theta_img**2 = theta_src**2 + theta_scatt**2
        # In the assumption that this angular size is determined by scattering, theta_scatt == theta_img
        # Another possibility to estimate theta_scatt is from NE2001. 
        # Tanya says it is ~0.1 mas
        # In NE2001, typical theta_scatt at 1 GHz is 1 mas. Then theta_scatt at 1.67 GHz (18 cm) is ~ (1 / 1.67) ** k
        # For 2209+236, k= 0.7 (Tanya) => theta_scatt = 0.7
        
        flux = self.p.flux.max() # maximum flux of a component [Jy]
        size = self.p.loc[self.p.flux == self.p.flux.max(), 'fwhm_I'] # size in the image plane of this component [mas]
        
        # Create a scattering model. Any parameters that aren't specify will default to current estimates for Sgr A*
        sm = so.ScatteringModel(POS_ANG=0., 
                                scatt_alpha = 5.0/3,
                                # scatt_alpha=1.5,
                                observer_screen_distance = 1 * 3.086e21, # 1 kpc in [cm]
                                source_screen_distance = 1e6 * 3.086e21, # 1 Gpc in [cm]
                                # theta_maj_mas_ref = self.p.fwhm_I.min(), # should be in [mas]
                                # theta_min_mas_ref = self.p.fwhm_I.min(), # should be in [mas]
                                theta_maj_mas_ref = theta_maj_mas_ref, # should be in [mas]
                                theta_min_mas_ref = theta_maj_mas_ref, # should be in [mas]
                                wavelength_reference_cm = 18.0,  # in [cm]
                                r_in = r_in, # in [cm]. 100 km
                                r_out = r_out # in [cm]. 1AU = 150e6 km
                                )
        
        # Define the observing wavelength in cm
        lambda_cm = self.wavelength
        # Make the unscattered image object 
        npix = 512
        fov = 2e4 * eh.RADPERUAS # fov = 20 mas
        
        # outfile
        if outfile is None:
            outfile = 'renormalized_refractive_noise_' + str(lambda_cm) + '.txt'
        
        
        print('Defining initial source')
        
        # Define the intrinsic source size
        im = eh.image.Image(np.zeros((npix,npix)), fov/npix, 0, 0, 0, rf=eh.C/(lambda_cm*0.01), source="DUMMY") # angular values should be in [rad]
        for i in self.p.index: # loop over all Gaussian components in the model
            theta_src_FWHM_uas = self.p.loc[i, 'fwhm_I'] * self.MAS2RAD  # self.p contains major in [mas]. 1e3 converts it to muas. Finally, the value is in [rad]
            print(f'theta_src_FWHM_uas = {theta_src_FWHM_uas:.2g} [rad]')
            im = im.add_gauss(self.p.loc[i, 'flux'], (theta_src_FWHM_uas, theta_src_FWHM_uas, 0, 0, 0))
        
        print(f'Defined the source consisting of {i+1} Gaussians')
        
        # Create a list of scattered images 
        if multithread is False:
            im_list = np.array([sm.Scatter(im) for j in range(int(samples))])
        else: # use multithreading to speed up
            n_threads = self.NTHREADS
            print(f"Running {n_threads} threads in parallel to generate {samples} random scattering screens")
            pool = mp.Pool(processes=n_threads)
            def f(j): 
                print('.', end='')
                return sm.Scatter(im, DisplayImage=False)
            im_list = pool.map(f, range(samples))
            pool.close()
            print('')
        
        print('Initialized screen images')
        
        # Normalize the images
        for j in range(len(im_list)):
            im_list[j].imvec /= im_list[j].total_flux()
        
        uv_EW = np.array([[u, 0] for u in np.arange(0e9,max_uv,0.01e9)]) # in [lambda]
        uv_NS = np.array([[0, u] for u in np.arange(0e9,max_uv,0.01e9)])
        vis_EW = [_.sample_uv(uv_EW, ttype='fast')[0] for _ in im_list]
        vis_NS = [_.sample_uv(uv_NS, ttype='fast')[0] for _ in im_list]
        sigma_EW = np.std(vis_EW,axis=0)
        sigma_NS = np.std(vis_NS,axis=0)
        
        print(f'flux = {flux} [Jy]')
        sigma_EW = sigma_EW * flux # refractive noise is relative to the total flux
        sigma_NS = sigma_NS * flux # and hence need to be normalized
        
        # Export the refractive noise table (u,v,sigma)
        # np.savetxt('renormalized_refractive_noise_' + str(lambda_cm) + '.txt', np.vstack([uv_EW[:,0],sigma_EW,sigma_NS]).transpose())
        np.savetxt(outfile, np.vstack([uv_EW[:,0],sigma_EW,sigma_NS]).transpose())
        return 

    def fit(self, xy, weights=None, bounds=None, gigalambda:bool=False):
        """Fit variable part of the model.
        
        Args:
            xy: 2d array of x and y data values. X - baselines in lambdas, Y - amplitude in Jy
            weights: 1d array with weights. Larger weights mean lower sigma and higher impact on the model
            bounds: 2d array of [[lower_bounds], [upper_bounds]] to supply to curve_fit routine
            gigalambda: when fitting, convert all baselines to gigalambdas (divide by 1e9). With this y_axis and x_axis will have comparable amplitudes
            
        """
        x, y = xy # now in [lambdas]
        nfit = self.ncomp
        namefit = self.p['name'].values
        print(f'Fitting {nfit} component')
        # print(f'Components fitted: {namefit}')
        
        p0 = self.par_fit
        if len(p0) == 0:
            # self.compose_model_fit()
            self.compose_model_fit()
            p0 = self.par_fit
        assert len(p0) > 0
        p0 = np.array(p0)
        # print(f'p0 = {p0}')
        
        # make bounds for easier fit
        if bounds is None:
            l_bounds = [0, 0] * nfit
            u_bounds = [2, 1e12] * nfit
            
            # lower limit on size in image plane is theta_scatt. It transforms in an upper limit of the extent of a Gaussian in the vis plane
            if self.theta_scatt > 0:
                u_bounds[1::2] = [self.max_sigma_V * 1.5] * self.ncomp # 1.5 factor makes it not that strict
            
            bounds = np.array((l_bounds, u_bounds))
        
        print(f'bounds = {bounds}')
        print(f'p0   = {p0}')
        
        if gigalambda: # convert all sizes to gigalambdas
            l_bounds[1::2] = [t / 1e9 for t in l_bounds[1::2]]
            u_bounds[1::2] = [t / 1e9 for t in u_bounds[1::2]]
            p0[1::2] = p0[1::2] / 1e9
            x = x / 1e9
        
        
        if weights is None:
            popt, pcov = curve_fit(self.func_fit, x, y, p0=p0, bounds=bounds)
        else:
            popt, pcov = curve_fit(self.func_fit, x, y, sigma=0.01/weights, absolute_sigma=True, p0=p0, bounds=bounds)
           
            
        if gigalambda: # convert all sizes back to lambdas
            popt[1::2] = [t * 1e9 for t in popt[1::2]]
            pcov[1::2] = [t * 1e9 for t in pcov[1::2]]
            pcov[:, 1::2] = [t * 1e9 for t in pcov[:, 1::2]]
            pass
            
            
        # print(popt)
        np.set_printoptions(precision=4, suppress=True)
        # print(f'popt = {np.round(np.array(popt), 3)}')
        # print(f'{np.round(np.array([p0, popt]).T, 3)}')
        
        self.pcov_fit = np.array(pcov)
        self.par_fit = np.array(popt)
        
        print('FITTING COMPLETED')
        
        
        # self.p.loc[:, 'flux'] = popt[::2]
        self.p['flux'] = popt[::2]
        self.p.loc[:, 'sigma_V'] = popt[1::2] # [mega lambda] -> requires a factor of 1e6 in the following line
        self.p.loc[:, 'sigma_I'] = 1 / (2 * np.pi * self.p.loc[:, 'sigma_V'] ) / self.MAS2RAD   #[mas]???
        self.p['fwhm_I'] = self.C * self.p['sigma_I']
        self.p['fwhm_V'] = self.C * self.p['sigma_V']
        
        
        return 
    
    def fit_2d(self, uva, bounds=None):
        """uva stands for U-V-amplitude
        """
        u, v, a = uva
        nfit = self.ncomp
        namefit = self.p['name'].values
        print(f'2D fitting {nfit} component')
        # print(f'Components fitted: {namefit}')
        
        p0 = self.par2_fit
        assert len(p0) > 0
        p0 = np.array(p0)
        
        # make bounds for easier fit
        if bounds is None:
            l_bounds = [0, 1e6, 1e6, 0] * nfit
            u_bounds = [2, 1e12, 1e12, 360] * nfit
            bounds = np.array((l_bounds, u_bounds))
        
        # print(f'bounds = {bounds}')
        # print(f'p0   = {p0}')
        
        def g1_2d_fit(uv, flux, sigma_max_V, sigma_min_V, phi_V):
            (u, v) = uv
            a = flux
            sx = sigma_max_V
            sy = sigma_min_V
            t = phi_V
            return a*np.exp( -((((np.cos(np.deg2rad(t)))**2)/(2*sx**2) + ((np.sin(np.deg2rad(t)))**2)/(2*sy**2))*u**2 +
                                                   2*(-(np.sin(2*np.deg2rad(t)))/(4*sx**2) + (np.sin(2*np.deg2rad(t)))/(4*sy**2))*u*v + 
                                                   (((np.sin(np.deg2rad(t)))**2)/(2*sx**2) + ((np.cos(np.deg2rad(t)))**2)/(2*sy**2))*v**2))
        def g2_2d_fit(uv, flux1, sigma_max_V1, sigma_min_V1, phi_V1, flux2, sigma_max_V2, sigma_min_V2, phi_V2):
            (u, v) = uv
            return  flux1*np.exp( -((((np.cos(np.deg2rad(phi_V1)))**2)/(2*sigma_max_V1**2) + ((np.sin(np.deg2rad(phi_V1)))**2)/(2*sigma_min_V1**2))*u**2 +
                    2*(-(np.sin(2*np.deg2rad(phi_V1)))/(4*sigma_max_V1**2) + (np.sin(2*np.deg2rad(phi_V1)))/(4*sigma_min_V1**2))*u*v + 
                    (((np.sin(np.deg2rad(phi_V1)))**2)/(2*sigma_max_V1**2) + ((np.cos(np.deg2rad(phi_V1)))**2)/(2*sigma_min_V1**2))*v**2)) + \
                    flux2*np.exp( -((((np.cos(np.deg2rad(phi_V2)))**2)/(2*sigma_max_V2**2) + ((np.sin(np.deg2rad(phi_V2)))**2)/(2*sigma_min_V2**2))*u**2 +
                    2*(-(np.sin(2*np.deg2rad(phi_V2)))/(4*sigma_max_V2**2) + (np.sin(2*np.deg2rad(phi_V2)))/(4*sigma_min_V2**2))*u*v + 
                    (((np.sin(np.deg2rad(phi_V2)))**2)/(2*sigma_max_V2**2) + ((np.cos(np.deg2rad(phi_V2)))**2)/(2*sigma_min_V2**2))*v**2))
                    
        def g3_2d_fit(uv, flux1, sigma_max_V1, sigma_min_V1, phi_V1, flux2, sigma_max_V2, sigma_min_V2, phi_V2,  flux3, sigma_max_V3, sigma_min_V3, phi_V3):
            (u, v) = uv
            return  flux1*np.exp( -((((np.cos(np.deg2rad(phi_V1)))**2)/(2*sigma_max_V1**2) + ((np.sin(np.deg2rad(phi_V1)))**2)/(2*sigma_min_V1**2))*u**2 +
                    2*(-(np.sin(2*np.deg2rad(phi_V1)))/(4*sigma_max_V1**2) + (np.sin(2*np.deg2rad(phi_V1)))/(4*sigma_min_V1**2))*u*v + 
                    (((np.sin(np.deg2rad(phi_V1)))**2)/(2*sigma_max_V1**2) + ((np.cos(np.deg2rad(phi_V1)))**2)/(2*sigma_min_V1**2))*v**2)) + \
                    flux2*np.exp( -((((np.cos(np.deg2rad(phi_V2)))**2)/(2*sigma_max_V2**2) + ((np.sin(np.deg2rad(phi_V2)))**2)/(2*sigma_min_V2**2))*u**2 +
                    2*(-(np.sin(2*np.deg2rad(phi_V2)))/(4*sigma_max_V2**2) + (np.sin(2*np.deg2rad(phi_V2)))/(4*sigma_min_V2**2))*u*v + 
                    (((np.sin(np.deg2rad(phi_V2)))**2)/(2*sigma_max_V2**2) + ((np.cos(np.deg2rad(phi_V2)))**2)/(2*sigma_min_V2**2))*v**2)) + \
                    flux3*np.exp( -((((np.cos(np.deg2rad(phi_V3)))**2)/(2*sigma_max_V3**2) + ((np.sin(np.deg2rad(phi_V3)))**2)/(2*sigma_min_V3**2))*u**2 +
                    2*(-(np.sin(2*np.deg2rad(phi_V3)))/(4*sigma_max_V3**2) + (np.sin(2*np.deg2rad(phi_V3)))/(4*sigma_min_V3**2))*u*v + 
                    (((np.sin(np.deg2rad(phi_V3)))**2)/(2*sigma_max_V3**2) + ((np.cos(np.deg2rad(phi_V3)))**2)/(2*sigma_min_V3**2))*v**2))
        
        
        def gn_2d_fit(uv, *p):
            ngauss = int((len(p) - 1) / 3)
            result = 0
            # print(f'p = {p}   len(p) = {len(p)}, ngauss = {ngauss}')
            
            for i in range(ngauss):
                result = result + g1_2d_fit(uv, p[3*i], p[3*i+1], p[3*i+2], p[3*i+3])
            return result
                
        
        
        func = None
        if nfit == 1:
            func = g1_2d_fit
        elif nfit == 2:
            func = g2_2d_fit
        elif nfit == 3:
            func = g3_2d_fit
            

        popt, pcov = curve_fit(func, (u, v), a, p0=p0, bounds=bounds, )
        # print(popt)
        
        self.par2_fit = popt
        
        # print(popt[::4])
        # print(self.p2.loc[:, 'flux'])
        
        self.p2.loc[:, 'flux'] = popt[::4]
        self.p2.loc[:, 'fwhm_max_I'] = self.C / (2 * np.pi * popt[1::4] * self.MAS2RAD)
        self.p2.loc[:, 'fwhm_min_I'] = self.C / (2 * np.pi * popt[2::4] * self.MAS2RAD)
        self.p2.loc[:, 'phi_I'] = popt[3::4] + 90

        # swap min and max if needed. And rotate phi by 90 deg then
        
        self.p2.loc[:, 'tmp_min'] = self.p2.loc[:, ['fwhm_max_I', 'fwhm_min_I']].min(axis=1)
        # print(self.p2)

        # print(self.p2['tmp_min'])
        
        self.p2['fwhm_max_I'] = self.p2.loc[:, ['fwhm_max_I', 'fwhm_min_I']].max(axis=1)
        self.p2['fwhm_min_I'] = self.p2['tmp_min']
        self.p2.drop(columns=['tmp_min'], inplace=True)
        
        # print(self.p2)
        
        return

    
    def report_tb(self):
        """Report Brightness Temperature [K] calculated from parameters of Gaussians in the model.
        """
        print('Brightness temperature from circular Gaussians')
        for i in self.p.index:
            flux = self.p.loc[i, 'flux'] # [Jy]
            size = self.p.loc[i, 'fwhm_I'] # [mas]
            nu = self.frequency * 1e-9 # [GHz]
            t = tb(flux, size, nu)
            print(f'Component {self.p.loc[i, "name"]}: T_b = {t:.2g}')
        
        return
    
    def report_tb2(self):
        """Report Brightness Temperature [K] calculated from parameters of Gaussians in the model.
        """
        print('Brightness temperature from elliptical Gaussians')
        for i in self.p2.index:
            flux = self.p2.loc[i, 'flux'] # [Jy]
            size_max = self.p2.loc[i, 'fwhm_max_I'] # [mas]
            size_min = self.p2.loc[i, 'fwhm_min_I'] # [mas]
            nu = self.frequency * 1e-9 # [GHz]
            size = np.sqrt(size_max * size_min)
            t = tb(flux, size, nu)
            print(f'Component {self.p2.loc[i, "name"]}: T_b = {t:.2g}')
        return 
    
    def report_tbn(self):
        """Report Brightness Temperature [K] calculated from parameters of Gaussians in the model.
        """
        print('Brightness temperature from elliptical Gaussians')
        for i in self.pn.index:
            flux = self.pn.loc[i, 'flux'] # [Jy]
            size_max = self.pn.loc[i, 'fwhm_major_I'] # [mas]
            size_min = self.pn.loc[i, 'fwhm_major_I'] * self.pn.loc[i, 'axratio'] # [mas]
            nu = self.frequency * 1e-9 # [GHz]
            size = np.sqrt(size_max * size_min)
            t = tb(flux, size, nu)
            print(f'Component {self.pn.loc[i, "name"]}: T_b = {t:.2g}')
        return  
    

    def radplot(self, dataframe, errors:bool=False):
        """Make a radplot
        
        Args:
            errors: plot errorbars
        """
        
        ax = plot_rad(dataframe, errors, wavelength=self.wavelength)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        xfunc = np.arange(0, xlim[1], 1e6)
        # yfunc = self.project1d()(xfunc)
        yfunc = self.func_fit(xfunc, *self.par_fit)
        ax.plot(xfunc, yfunc, color='tab:orange', zorder=10000)
        # ax.fill_between(xfunc, yfunc_bmaj, yfunc_bmin, color='tab:orange', alpha=0.5)
        
        ax.set_ylim(ylim)
        self.ax = ax
        
        # colors = ['tab:green', 'tab:brown', 'tab:purple']
        # for i in range(self.ncomp):
        #     yg = self.g1(self.par_fit[2*i], self.par_fit[2*i+1])(xfunc,self.par_fit[2*i], self.par_fit[2*i+1])
        #     ax.plot(xfunc, yg, ls='--', color=colors[i])

        return ax
    
    def radplot2(self, dataframe):
        """Make a radplot"""
        
        ax = plot_rad(dataframe)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        xfunc = np.arange(0, xlim[1], 1e6)
        
        # print(f'par2_fit: {self.par2_fit}')
        
        yfunc_major = self.g1(0,0)(xfunc, self.par2_fit[0], self.par2_fit[1])
        yfunc_minor = self.g1(0,0)(xfunc, self.par2_fit[0], self.par2_fit[2])
        
        for i in range(1, self.ncomp):
            yfunc_major = yfunc_major +  self.g1(0,0)(xfunc, self.par2_fit[4*i], self.par2_fit[4*i+1])
            yfunc_minor = yfunc_minor +  self.g1(0,0)(xfunc, self.par2_fit[4*i], self.par2_fit[4*i+2])
        
        ax.plot(xfunc, yfunc_major, color='tab:purple', zorder=10000)
        ax.plot(xfunc, yfunc_minor, color='tab:purple', zorder=10000)
        # ax.fill_between(xfunc, yfunc_bmaj, yfunc_bmin, color='tab:orange', alpha=0.5)
        
        ax.set_ylim(ylim)
        self.ax = ax
        
        if len(self.par_fit) > 0: # also plot 1D
            yfunc = self.func_fit(xfunc, *self.par_fit)
            ax.plot(xfunc, yfunc, color='tab:orange', zorder=10000)
            
        
        
        # colors = ['tab:green', 'tab:brown', 'tab:purple']
        # for i in range(self.ncomp):
        #     yg = self.g1(self.par_fit[2*i], self.par_fit[2*i+1])(xfunc,self.par_fit[2*i], self.par_fit[2*i+1])
        #     ax.plot(xfunc, yg, ls='--', color=colors[i])

        return ax


    def make_asymptotic_noise(self, df, dofit:bool=True):
        """Make refractive noise using asymptotic expression for long baselines
        Args:
            df: dataframe to fit
            dofit: perform fitting
        """        
        flux = self.p.flux.max()
        fwhm_I = self.p.loc[self.p.flux == self.p.flux.max(), 'fwhm_I'].values[0]
        index = df.loc[df.base > 0.25e9, :].index
        x = np.array(df.loc[index, 'base'].values)  # baselines of real measurements 
        y_residual = df.loc[index, 'ampl'] - self.func(x, *self.par_fit)
        
        self.ax.plot(x, y_residual, 'x', color='red')
        xfit = np.arange(1e7, 1.4e9, 1e7)

        if dofit is True:
            
            if True: # two parameters only
                func58 =  lambda x, *p: flux* 0.0071 * (x / 555.555555e6)**(-5/6) * (p[0] / 0.3)**(5/6) * (fwhm_I / 1)**(-2) * (p[1] / 1)**(-1/6)
                p0 = [0.36, 1] # theta_scatt, D
                print(f'p0 = {p0}')
                popt, pcov = curve_fit(func58, x, y_residual, p0=p0, bounds=[[0.0, 1] , [1, 2]])
                print(f'Asymptotic ref noise fit: theta_scatt = {popt[0]:.2f} mas, D = {popt[1]:.1f} kpc')
                self.ax.plot(xfit, func58(xfit, *popt), '-.', label='Asymptotic refractive noise')
                
            
            if False:
                func57 =  lambda x, *p: p[0] * 0.0071 * (x / 555.555555e6)**(-5/6) * (p[1] / 0.3)**(5/6) * (p[2] / 1)**(-2) * (p[3] / 1)**(-1/6)    
                p0 = []
                p0.append( self.p.flux.max())
                p0.append(0.36) # scattering size
                p0.append(1 *  self.p.loc[self.p.flux == self.p.flux.max(), 'fwhm_I'].values[0])
                p0.append(1)
                # p0 = [flux, theta_scatt, theta_img, D ]
                print(f'p0 = {p0}')
                popt, pcov = curve_fit(func57, x, y_residual, p0=p0, bounds=[[0.5, 0.0, 0.0, 1] , [1, 10, 10, 2]])
                print(f'Asymptotic ref noise fit: flux = {popt[0]:.2f} Jy, theta_scatt = {popt[1]:.2f} mas, theta_img = {popt[2]:.2f} mas, D = {popt[3]:.1f} kpc')
                self.ax.plot(xfit, func57(xfit, *popt), '-.', label='Asymptotic refractive noise')
        else:
            func57 =  lambda x, *p: p[0] * 0.0071 * (x / 555.555555e6)**(-5/6) * (p[1] / 0.3)**(5/6) * (p[2] / 1)**(-2) * (p[3] / 1)**(-1/6)    
            p0 = []
            p0.append( self.p.flux.max())
            p0.append(0.36) # scattering size
            p0.append(1 *  self.p.loc[self.p.flux == self.p.flux.max(), 'fwhm_I'].values[0])
            p0.append(1)
            popt = p0
            print(f'Asymptotic ref noise fit: flux = {popt[0]:.2f} Jy, theta_scatt = {popt[1]:.2f} mas, theta_img = {popt[2]:.2f} mas, D = {popt[3]:.1f} kpc')
            self.ax.plot(xfit, func57(xfit, *popt), '-.', label='Asymptotic refractive noise (not fitted)')

        return
    
    def init_Ng_2d(self, flux, fwhm_major_I, axratio, phi_I):
        """Initiate a model consisting of N 2D elliptical Gaussian in image plane and make all conversions to the Visibility plane.
        
        Args:
            flux: array of component fluxes in [Jy]
            fwhm_mjor_I: array of component major axes in IMAGE plane [mas]
            axratio: array of minor/major axes ratio. axratio is in [0,1]
            phi_I: array of component major axis directions [degrees]
            """
        assert len(flux)==len(fwhm_major_I)==len(axratio)==len(phi_I)
        self.ncomp = len(flux)
        flux = np.array(flux)
        fwhm_major_I = np.array(fwhm_major_I)
        axratio = np.array(axratio)
        phi_I = np.array(phi_I)
        
        
        for i in range(self.ncomp):
            self.pn.loc[i, ['name', 'flux', 'fwhm_major_I', 'axratio', 'phi_I']] = [f'g{i+1}', flux[i], fwhm_major_I[i], axratio[i], phi_I[i]]
            self.update_param_I2V()
            self.parn_fit.extend(self.pn.loc[i, ['flux', 'sigma_major_V', 'axratio', 'phi_V']].values)
        
        def g1_2d(uv, *p):
            """2D Gaussian in UV plane. The dimensiones are sigma_max, sigma_min. Sigma_max is oriented with an angle phi to somewhere."""
            # p= [ampl, sigma_max, axratio, phi]. Units are [Jy, lambda, 1, degrees]
            return p[0] * np.exp( -( ((np.cos(np.deg2rad(p[3])))**2 / (2*p[1]**2) + (np.sin(np.deg2rad(p[3])))**2 / (2*(p[1]*p[2])**2)) * uv[0]**2 +
                               2*(-(np.sin(2*np.deg2rad(p[3]))) / (4*p[1]**2) + np.sin(2*np.deg2rad(p[3])) / (4*(p[1]*p[2])**2)) * uv[0] * uv[1] + 
                                        ((np.sin(np.deg2rad(p[3])))**2 / (2*p[1]**2) + (np.cos(np.deg2rad(p[3])))**2 / (2*(p[1]*p[2])**2)) * uv[1]**2) )
        
        self.funcn = lambda uv, *P: sum([g1_2d(uv, *P[4*i:4*i+4]) for i in range(int(len(P)/4))])
        self.funcn_fit = lambda uv, *P: sum([g1_2d(uv, *P[4*i:4*i+4]) for i in range(int(len(P)/4))])
        
        return
        
    def fitn_2d(self, uva, bounds=None, weights=None):
        """uva stands for U-V-amplitude
        """
        u, v, a = uva
        print(f'2D fitting {self.ncomp} component in fitn_2d()')
        
        p0 = self.parn_fit
        assert len(p0) > 0
        p0 = np.array(p0)
        # l_bounds = []
        # u_bounds = []
        
        # make bounds for easier fit
        if bounds is None:
            l_bounds = [0, 1e6, 0.7, 0] * self.ncomp
            u_bounds = [2, 1e15, 1, 360] * self.ncomp
            bounds = np.array((l_bounds, u_bounds))
        
        print(f'bounds = {bounds}', end='\n\n')
        # print(f'p0   = {p0}', end='\n\n')
        
        # adjust p0 so it is never outside of bounds
        for i, p in enumerate(p0):
            if p < bounds[0][i]:
                p0[i] = bounds[0][i]
            elif p > bounds[1][i]:
                p0[i] = bounds[1][i]
        
        
        
        if weights is not None:
            popt, pcov = curve_fit(self.funcn_fit, np.array((u, v)), a, p0=p0, bounds=bounds, sigma=weights )  # passing (u, v) as np.array was crucial here
        else:
            popt, pcov = curve_fit(self.funcn_fit, np.array((u, v)), a, p0=p0, bounds=bounds, sigma=(np.log(np.abs(u+v))) )  # passing (u, v) as np.array was crucial here
        # print(popt)
        
        # update model parameters
        self.parn_fit = popt
        for i in range(self.ncomp):
            self.pn.loc[i, ['flux', 'sigma_major_V', 'axratio', 'phi_V']] = popt[4*i : 4*i + 4]
        
        self.update_param_V2I()
        
        return
    
        
    def radplotn(self, dataframe):
        """Make a radplot"""
        
        ax = plot_rad(dataframe)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        xfunc = np.arange(0, xlim[1], 1e6)
        
        yfunc_major = self.g1(0,0)(xfunc, self.parn_fit[0], self.parn_fit[1] )
        yfunc_minor = self.g1(0,0)(xfunc, self.parn_fit[0], self.parn_fit[1] * self.parn_fit[2])
        
        for i in range(1, self.ncomp):
            yfunc_major = yfunc_major +  self.g1(0,0)(xfunc, self.parn_fit[4*i], self.parn_fit[4*i+1])
            yfunc_minor = yfunc_minor +  self.g1(0,0)(xfunc, self.parn_fit[4*i], self.parn_fit[4*i+1] * self.parn_fit[4*i+2])
        
        ax.plot(xfunc, yfunc_major, color='tab:purple', zorder=10000)
        ax.plot(xfunc, yfunc_minor, color='tab:purple', zorder=10000)
        # ax.fill_between(xfunc, yfunc_bmaj, yfunc_bmin, color='tab:orange', alpha=0.5)
        
        ax.set_ylim(ylim)
        self.ax = ax
        
        if len(self.par_fit) > 0: # also plot 1D
            yfunc = self.func_fit(xfunc, *self.par_fit)
            ax.plot(xfunc, yfunc, color='tab:orange', zorder=10000)
            
        
        
        # colors = ['tab:green', 'tab:brown', 'tab:purple']
        # for i in range(self.ncomp):
        #     yg = self.g1(self.par_fit[2*i], self.par_fit[2*i+1])(xfunc,self.par_fit[2*i], self.par_fit[2*i+1])
        #     ax.plot(xfunc, yg, ls='--', color=colors[i])

        return ax          
        
      
def proceed_2D(noise=False, **kw):
    plot_init = kw.get('plot_init', True)
    # source and data parameters
    bins = kw.get('bins', 51)
    source = kw.get('source', '2209+236')
    band = kw.get('band', 'L')
    polar = kw.get('polar', 'parallel')
    theta_scatt = kw.get('theta_scatt', 0.0)
    
    # initial model parameters and bounds
    flux = kw.get('flux', [1, 0.1])
    flux_lower = kw.get('flux_lower', [0.01, 0.01])
    flux_upper = kw.get('flux_upper', [2., 2.])
    
    fwhm = kw.get('fwhm', [1, 2.])
    fwhm_lower = kw.get('fwhm_lower', [0.01, 0.01])
    fwhm_upper = kw.get('fwhm_upper', [10, 10.])
    
    fwhm_V_lower = 1 / (2 * np.pi * np.array(fwhm_upper) / 206265000 )
    fwhm_V_upper = 1 / (2 * np.pi * np.array(fwhm_lower) / 206265000 )
    print(fwhm_V_upper)
    
    
    axratio = kw.get('axratio', [1., 1.])
    axratio_lower = kw.get('axratio_lower', [0.01, 0.01])
    axratio_upper = kw.get('axratio_upper', [1.0, 1.0])
    
    phi = kw.get('phi', [0., 0.])
    phi_lower = kw.get('phi_lower', [0.01, 0.01])
    phi_upper = kw.get('phi_upper', [359.99, 359.99])
    
    # scattering parameters
    max_uv = kw.get('max_uv', 1e9)
    samples = kw.get('samples', 10)
    r_in = kw.get('r_in', 5e10)
    r_out = kw.get('r_out', 5000e10)
    
    if band == 'L':
        wavelength = 18
    elif band == 'C':
        wavelength = 6
    
    outfile = f'renormalized_refractive_noise_{wavelength}cm_{source}.txt'
    
    # arrange bounds
    l_bounds = np.array([flux_lower, fwhm_V_lower, axratio_lower, phi_lower]).T.reshape(1,-1).squeeze()
    u_bounds = np.array([flux_upper, fwhm_V_upper, axratio_upper, phi_upper]).T.reshape(1,-1).squeeze()
    
    
    df = get_data(source, band, polar)
    df = add_weights(df, bins=bins, doplot=False)
    m = Model1D(wavelength=wavelength, theta_scatt=theta_scatt)
    
    # ND
    m.init_Ng_2d(flux, fwhm, axratio, phi)  
    # m.init_Ng_2d([0.59, 0.23, 0.04], [1.22, 9.5, 0.09], [0.7, 0.8, 0.9], [270., 270., 10.])
    if plot_init:
        axn = m.radplotn(df)
    
    m.fitn_2d([df.x.values, df.y.values, df.ampl.values], weights=1/df.weight, bounds=[l_bounds, u_bounds]) # base - in [lambdas]
    axn = m.radplotn(df)
    m.report_tbn()
    print('Parameters of circular components in the image plane')
    print(m.pn.loc[:, ['flux', 'fwhm_major_I', 'axratio', 'phi_I']])
    
    if noise:
        m.make_noise(max_uv=max_uv, samples=samples,
                             theta_maj_mas_ref=m.theta_scatt,
                             r_in=r_in,
                             r_out=r_out,
                             outfile=outfile)
        noise = read_noise(outfile)
        axn.plot(noise.uv_dist, noise.sigma, '--', color='tab:green', label=f'ref noise, theta={m.theta_scatt:.2f} [mas]')
        # m.make_asymptotic_noise(df, dofit=False) # also plot asymptotic noise to compare with the numerically calculated one
    axn.legend()
    
    return m


def report(filename, m):
    """Write useful info on the fit of teh model m into the file filename
    Args:
        filename: filename to write to
        m: model object
    """
    with open(filename, 'w') as F:
        
        print(f'source = {m.source}, band = {m.band}', file=F)
        print(f'Fitted model: {m.p.index.size} Gaussians', file=F)
        print('Parameters of circular components in the image plane', file=F)
        print(m.p.loc[:, ['flux', 'fwhm_I']], file=F)
        print(f'Total flux of the model is {m.p.flux.sum():.3f} Jy', file=F)
    

        errors = np.sqrt(np.diag(m.pcov_fit))

        print('Errors:', file=F)
        flux_errors = errors[::2]
        size_errors_uv = errors[1::2]
        size_errors_image = []
        for i, se_uv in enumerate(size_errors_uv):
            # for each component, calculate size +- size_error in UV plane and then convert whese sizes to the image plane
            s_uv = m.p.iloc[i]['sigma_V']
            smin_uv = s_uv - se_uv
            smax_uv = s_uv + se_uv
            # to image plane
            smax_im =  1 / ( 2 * np.pi * smin_uv * m.MAS2RAD) # sigma
            smin_im =  1 / ( 2 * np.pi * smax_uv * m.MAS2RAD)
            
            # convert to FWHM_image
            fwhm_image_min = smin_im * m.C
            fwhm_image_max = smax_im * m.C
            
            fwhm_image_error = (fwhm_image_max - fwhm_image_min) / 2 
            size_errors_image.append(fwhm_image_error)
            
            print(f'Comp {i+1}: Size = {m.p.iloc[i]["fwhm_I"]:.3f} +- {fwhm_image_error:.3f} mas, Flux = {m.p.iloc[i]["flux"]:.3f} +- {flux_errors[i]:.3f} Jy', file=F)
        
        print('Brightness temperature from circular Gaussians', file=F)
        for i in m.p.index:
            flux = m.p.loc[i, 'flux'] # [Jy]
            size = m.p.loc[i, 'fwhm_I'] # [mas]
            nu = m.frequency * 1e-9 # [GHz]
            t = tb(flux, size, nu)
            t_err = tb_err(flux, size, nu, flux_errors[i], size_errors_image[i])
            print(f'Component {m.p.loc[i, "name"]}: observed T_b = {t:.2g} +- {t_err:.2g} K', file=F)

        
        
        print(f'\n\n\nDiagonal of pcov is {errors}', file=F)
        print(f'The matrix itself is \n{m.pcov_fit}', file=F)


    
    return



def do(source, wavelength:int=18, polar:str='parallel', timerange=None, bins:int=51,
       r_in=1e9, r_out=2e20, samples:int=100, nthreads:int=10,
       function:str='1g', param=[1,1],
       overwrite_noise:bool=False,
       scale_to_noise:bool=True
       ):
    """Do all the job. Read data, fit it, add noise, write report
    
    Args:
        function: function name [1g, 2g, 3g]
        param: array of function parameters
        scale_to_noise: change y_lim to show noise
    """
    if 5 < wavelength < 7:
        band = 'C'
    if 17 < wavelength < 19:
        band = 'L'

    outfile = f'renormalized_refractive_noise_{wavelength}cm_{source}_{function.upper()}_rin_{r_in:g}_rout_{r_out:g}.txt'
    reportfile = f'model_report_{wavelength}cm_{source}_{function.upper()}N.txt'
    plotfile = f'{source}_{band}_{function.upper()}N.png'
    
        
    df = get_data(source, band, polar, uplim=True)
    det = df.loc[df.uplim == 0, :]   # use only detections for the fit
    if bins > 0:
        det = add_weights(det, bins=bins)
    m = Model1D(wavelength=wavelength, theta_scatt=0.00)
    m.source = source
    m.NTHREADS = nthreads
    
    
    
    if function == '1g':
        m.init_1g(*param)
    if function == '2g':
        m.init_2g(*param)
    if function == '3g':
        m.init_3g(*param)
        
    m.fit([det.base.values, det.ampl.values], weights=det.weight.values)
    ax = m.radplot(df)
    
    report(reportfile, m)

    if m.theta_scatt == 0.0:
        m.theta_scatt = m.p.fwhm_I.min()
    
    if True and overwrite_noise is True:
        m.make_noise(max_uv=3.0e9, samples=samples,  # with samples < 30 the
                             theta_maj_mas_ref=m.theta_scatt,
                             r_in=r_in,
                             r_out=r_out,
                             outfile=outfile)
    
    try:
        noise = read_noise(outfile)
    except:
        raise FileNotFoundError(f'Noise file {outfile} is not found or cannot be read')
        
    ax.plot(noise.uv_dist, noise.sigma, '--', color='tab:green', label=rf'Refractive noise, $\theta_{{scatt}}$={m.theta_scatt:.2f} mas')
    ylimits = np.array(ax.get_ylim())
    
    xmax = df.base.max()
    xmax_noise = noise.loc[noise.uv_dist <= xmax, 'uv_dist'].max()
    ymin = noise.loc[noise.uv_dist == xmax_noise, 'sigma']
    ylimits[0] = ymin
    ax.set_ylim(ylimits)
    
    ax.legend()
    ax.loglog()
    ax.get_figure().suptitle(f'{source}, {m.band}-band, {m.p.index.size}G+N ')
    ax.get_figure().savefig(plotfile)

    return


if __name__ == "__main__":
    
    # for t in np.arange(0, 0.4, 0.1):
        # m = proceed_2D(theta_scatt=t, fwhm_lower=[t,t],  flux=[0.5, 1.2], fwhm=[1., 0.01], bins=101, source='2209+236', band='L', polar='parallel', plot_init=False)


    if True:
        
        # r_in= 1e11 is too much. I mixed up, Jupiter size is 7e9, Earth size is 1.3e9
        
        
        do(source='2209+236', wavelength=6, function='2g', param=[0.77, 0.5, 0.15, 0.08], samples=3000, nthreads=45, overwrite_noise=True)
        
        do(source='2209+236', wavelength=18, function='1g', param=[0.8, 1], bins=0, samples=3000, overwrite_noise=True)
        do(source='2209+236', wavelength=18, function='2g', param=[0.77, 18, 0.15, 0.08], bins=0, samples=3000, overwrite_noise=True)
        do(source='2209+236', wavelength=18, function='3g', param=[0.6, 0.86, 0.3, 10, 0.1, 0.1], samples=3000, overwrite_noise=True)


        do(source='0657+172', wavelength=6, function='2g', param=[0.77, 0.5, 0.15, 0.08], samples=3000, overwrite_noise=True)

        do(source='0657+172', wavelength=18, polar='RR', function='1g', param=[1, 1], samples=3000, overwrite_noise=True)
        do(source='0657+172', wavelength=18, polar='RR', function='2g', param=[0.77, 0.5, 0.15, 0.08], bins=0, samples=3000, overwrite_noise=True) # redo fitting
        do(source='0657+172', wavelength=18, polar='RR', function='3g', param=[0.30847, 2.6178, 0.3, 0.6984, 5.0000e-02, 0.2589], bins=0, samples=3000, overwrite_noise=True) # redo fitting




        

        
        for rin_power in [7,8,9,10]:
            # do(source='2209+236', wavelength=18, function='2g', param=[0.77, 18, 0.15, 0.08], bins=0, samples=100, r_in=10**rin_power, overwrite_noise=True) 
            # Result: refractive noise raises with increasing rin_power, but never crosses the lower detection. 10^13 seems too much. 
            
            
            # do(source='0657+172', wavelength=18, polar='RR', function='2g', param=[0.77, 0.5, 0.15, 0.08], bins=0, samples=100, r_in=10**rin_power, overwrite_noise=True)
            # Result: modelling was off. Needs to be redone. But refractive noise raises with increasing rin_power. 10^13 seems too much. At 10^12 some detections might be due to ref noise. 
    
            pass









    if False:
        
        overwrite_noise = False
        
        wavelength = 6
        r_in = 1e11
        r_out = 1e20
        samples = 100
        bins = 51
        
        # 2209+236, C-band. Works okay. 
        
        source = '2209+236'
        outfile = f'renormalized_refractive_noise_{wavelength}cm_{source}_rin_{r_in:g}_rout_{r_out:g}.txt'
        reportfile = f'model_report_{wavelength}cm_{source}.txt'
            
        
        df = get_data(source, 'C', 'parallel', uplim=True)
        det = df.loc[df.uplim == 0, :]   # use only detections for the fit
        det = add_weights(det, bins=bins)
        m = Model1D(wavelength=wavelength, theta_scatt=0.00)
        m.source = source
        
        m.init_2g(0.77, 0.5, 0.15, 0.08)
        
        m.fit([det.base.values, det.ampl.values], weights=det.weight.values)
        ax = m.radplot(df)
        
        report(reportfile, m)
        
        print('Parameters of circular components in the image plane')
        print(m.p.loc[:, ['flux', 'fwhm_I']])
        
        print(f'Total flux of the model is {m.p.flux.sum():.3f} Jy')
    
        m.report_tb()
        print(f'Diagonal of pcov is {np.sqrt(np.diag(m.pcov_fit))}')
        print(f'The matrix itself is \n{m.pcov_fit}')
        
        if m.theta_scatt == 0.0:
            m.theta_scatt = m.p.fwhm_I.min()
        
        if True and overwrite_noise is True:
            m.make_noise(max_uv=3.0e9, samples=samples,  # with samples < 30 the
                                 theta_maj_mas_ref=m.theta_scatt,
                                 r_in=r_in,
                                 r_out=r_out,
                                 outfile=outfile)
        
        try:
            noise = read_noise(outfile)
        except:
            raise FileNotFoundError(f'Noise file {outfile} is not found or cannot be read')
            
        ax.plot(noise.uv_dist, noise.sigma, '--', color='tab:green', label=f'ref noise, theta={m.theta_scatt:.2f} [mas]')
        ax.legend()
        ax.get_figure().suptitle(f'{source}, {m.band}-band, {m.p.index.size}G+N ')




    if False:
        
        overwrite_noise = False
        
        wavelength = 18
        r_in = 1e11
        r_out = 1e20
        samples = 100
        
        # 2209+236, L-band. Works well
        
        source = '2209+236'
        outfile = f'renormalized_refractive_noise_{wavelength}cm_{source}_rin_{r_in:g}_rout_{r_out:g}.txt'
        reportfile = f'model_report_{wavelength}cm_{source}.txt'

        df = get_data(source, 'L', 'parallel', uplim=True)
        det = df.loc[df.uplim == 0, :]   # use only detections for the fit
        det = add_weights(det, bins=51)
        m = Model1D(wavelength=18, theta_scatt=0.00)
        
        
        # # 1G
        m.init_1g(0.8, 1)
        # title = f'{source}, L-band, 1G+N'
        
        # # 2G
        # m.init_2g(0.77, 18, 0.15, 0.08)  # good fit (by quality), but gives one large component (10 mas)
        # title = f'{source}, L-band, 2G+N'
        
        # # 3G
        # m.init_3g(0.6, 0.86, 0.3, 10, 0.1, 0.1)  # good fit. No room for scattering
        
        
        
        title = f'{source}, L-band, {m.p.index.size}G+N'
        
        
        
        
        m.fit([det.base.values, det.ampl.values], weights=det.weight.values, gigalambda=True)
        ax = m.radplot(df)
        ax.loglog()
        
        print('Parameters of circular components in the image plane')
        print(m.p.loc[:, ['flux', 'fwhm_I']])
        
        print(f'Total flux of the model is {m.p.flux.sum():.3f} Jy')
    
        m.report_tb()
        print(f'Diagonal of pcov is {np.sqrt(np.diag(m.pcov_fit))}')
        print(f'The matrix itself is \n{m.pcov_fit}')
        
        if m.theta_scatt == 0.0:
            m.theta_scatt = m.p.fwhm_I.min()
            # m.theta_scatt = 10
            
        
        if False:
            m.make_noise(max_uv=1.4e9, samples=100,
                                 theta_maj_mas_ref=m.theta_scatt,
                                 r_in=1e11,
                                 r_out=2e20,
                                 outfile=outfile)
            noise = read_noise(outfile)
            ax.plot(noise.uv_dist, noise.sigma, '--', color='tab:green', label=f'ref noise, theta={m.theta_scatt:.2f} [mas]')
            # m.make_asymptotic_noise(df, dofit=False) # also plot asymptotic noise to compare with the numerically calculated one

            # r_in = 1e9
            # r_in = 1e10
            
        ax.legend()
        # ax.set_ylim(5e-4, 2)
        ax.get_figure().suptitle(title)



    



    if False:
        # 0657+172, C-band. Looks okay. Check expected fluxes and sizes
        
        source = '0657+172'
        outfile = f'renormalized_refractive_noise_6cm_{source}.txt'
        
        df = get_data(source, 'C', 'parallel', uplim=True)
        det = df.loc[df.uplim == 0, :]   # use only detections for the fit
        det = add_weights(det, bins=51)
        m = Model1D(wavelength=6, theta_scatt=0.00)
        
        m.init_2g(0.77, 0.5, 0.15, 0.08)
        
        m.fit([det.base.values, det.ampl.values], weights=det.weight.values)
        ax = m.radplot(df)
        
        print('Parameters of circular components in the image plane')
        print(m.p.loc[:, ['flux', 'fwhm_I']])
        
        print(f'Total flux of the model is {m.p.flux.sum():.3f} Jy')
    
        m.report_tb()
        print(f'Diagonal of pcov is {np.sqrt(np.diag(m.pcov_fit))}')
        print(f'The matrix itself is \n{m.pcov_fit}')
        
        if m.theta_scatt == 0.0:
            m.theta_scatt = m.p.fwhm_I.min()
        
        if True:
            m.make_noise(max_uv=3.0e9, samples=100,
                                 theta_maj_mas_ref=m.theta_scatt,
                                 r_in=1e11,
                                 r_out=2e20,
                                 outfile=outfile)
            noise = read_noise(outfile)
            ax.plot(noise.uv_dist, noise.sigma, '--', color='tab:green', label=f'ref noise, theta={m.theta_scatt:.2f} [mas]')
        ax.legend()
        ax.loglog()
        ax.get_figure().suptitle(f'{source}, C-band, 2G+N ')

    
    
    
    
    
    if False:
        # 0657+172, L-band. Looks Good
        
        source = '0657+172'
        outfile = f'renormalized_refractive_noise_6cm_{source}.txt'
        
        df = get_data(source, 'L', 'parallel', uplim=True)
        
        # check with trusted telescopes only
        # telescopes = ['RA', 'HH', 'EF', 'AR', 'GB', 'WB']
        # df = df.loc[(df.sta1.isin(telescopes)) & (df.sta2.isin(telescopes)), :]
        
        # select only RR
        df = df.loc[df.polar == 'RR', :]    # this btw has a lot of sense. Because there are no LL correlation in space => adding LL overweights ground baselines
        
        
        
        
        det = df.loc[df.uplim == 0, :]   # use only detections for the fit
        # det = add_weights(det, bins=51)
        m = Model1D(wavelength=18, theta_scatt=0.00)
        
        # m.init_2g(0.77, 0.5, 0.15, 0.08) # fits well except for the longest baselines. m.par_fit = array([5.5847e-01, 2.3154e+07, 3.4895e-01, 1.3361e+08])
        # m.init_2g(0.49, 2.3, 0.29, 0.40) 
        
        
        # m.init_3g(0.55, 3.3, 0.35, 0.58, 0.15, 1.3) # fits well except for the longest baselines, where the third gaussian is too small
        # m.par_fit = array([5.5847e-01, 2.3154e+07, 3.0000e-01, 1.1000e+08, 5.0000e-02, 3.0141e+08])  # it gives the fit I want (more or less. HAndtuned parameters)
        # their corresponding sizes in mas are array([1.4178, 0.2984, 0.1089])
        #
        m.init_3g(0.30847, 2.6178, 0.3, 0.6984, 5.0000e-02, 0.2589)   # initial conditions look very promising for the fitting
        
        m.fit([det.base.values, det.ampl.values], weights=det.weight.values)
        # m.fit([det.base.values, det.ampl.values], weights=det.weight.values, bounds=[[0,0,0,0,0,0], [1, 1e9, 1, 1e9, 1,5e8]])
        ax = m.radplot(df)
        
        print('Parameters of circular components in the image plane')
        print(m.p.loc[:, ['flux', 'fwhm_I']])
        
        print(f'Total flux of the model is {m.p.flux.sum():.3f} Jy')
    
        m.report_tb()
        print(f'Diagonal of pcov is {np.sqrt(np.diag(m.pcov_fit))}')
        print(f'The matrix itself is \n{m.pcov_fit}')
        
        if m.theta_scatt == 0.0:
            m.theta_scatt = m.p.fwhm_I.min()
        
        if True:
            m.make_noise(max_uv=1.4e9, samples=100,
                                 theta_maj_mas_ref=m.theta_scatt,
                                 r_in=1e11,
                                 r_out=2e20,
                                 outfile=outfile)
            noise = read_noise(outfile)
            ax.plot(noise.uv_dist, noise.sigma, '--', color='tab:green', label=f'ref noise, theta={m.theta_scatt:.2f} [mas]')
        ax.legend()
        ax.loglog()
    
    
    
    
    
    
    
    if False:
        
        band = 'L'
        if band == 'C':
            wavelength = 6
        elif band == 'L':
            wavelength = 18
      
        # df = get_data('2209+236', band, 'parallel', uplim=False, quality_control=True)
        df = get_data('0657+172', band, 'parallel', uplim=True, quality_control=True)
        
        # fig, ax = plot_uv(df.loc[df.sta1 == 'RA', :], wavelength=6)
        
        exper2check = ['rags28ba', 'rags28bc', 'rags28be', 'rags28bi', 'rags28bo', 'rags28bs', 'rags28bu']
        # df = df.loc[df.exper.isin(exper2check), :]
        
        fig, ax = plot_uv(df, wavelength=wavelength)
        axi = plot_rad(df, wavelength=wavelength, connect=False)
        axi.loglog()
        
        if False:
            for t in set(list(df.sta1.values) + list(df.sta2.values)):
                dp = rl_ratio(df.loc[df.uplim==0, :], telescope=t)
        
        # dp = rl_ratio(df, telescope='TR')
        # dp = rl_ratio(df, telescope='EF')
        
        # TODO: check RR/LL ratio for all telescopes to see global trends/miscalibration
        # IR16: <LL/RR> = 1.31 # TODO correc/ EF-IR16@rags28bl/be = MC-NT@rags28bt  =>> LL@IR16 is correct. RR should be scaled up by 1.31
        # IR16 this correction applies to all C-band observatinos of 0657+172 except for rags28bt/bu
        # 
        # do = selfcal(df, 'rags28bo', 'L',  trusted=['EF', 'RA', 'GB', 'AR', 'HH', 'MC', 'NT', 'YS', 'TR'], corrections={'WB':{'LL':20, 'RR':3.5}}) # done
        # do = selfcal(df, 'rags28bu', 'L',  trusted=['EF', 'RA', 'GB', 'AR', 'HH', 'MC', 'NT', 'YS'], corrections={'WB':{'LL':1, 'RR':1}}) # done
        # do = selfcal(df, 'rags28ac', band,  trusted=['EF', 'RA', 'GB', 'AR', 'HH', 'MC', 'NT', 'YS', 'TR'], corrections={'WB':{'LL':1, 'RR':1}})
        
        # do = selfcal(df, 'rags28ax', 'L',  trusted=['EF', 'RA', 'GB', 'AR', 'HH', 'MC', 'NT', 'YS'], corrections={'Tr':{'LL':1, 'RR':1}})
        # do = selfcal(df, 'rags28az', 'L',  trusted=['EF', 'RA', 'GB', 'AR', 'HH', 'MC', 'NT', 'YS'], corrections={'Tr':{'LL':1, 'RR':1}})
        
        
        # vplot(df)
        # vplot(df, telescope='TR')
        
    
    
    