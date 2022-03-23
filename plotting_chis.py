#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:17:05 2022

@author: felipeabedrapo
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

from df_functions import find_intersection_points
from df_functions import round_list_to_decimal

def plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel = 'Energy (GeV)'):
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize = fontsize-5)
    ax.legend(fontsize = fontsize)



#%% LOADING DFS

from reading_files import create_all_dfs_for_angantyr_model

energies_param = 'sigma_D'

dfs1 = create_all_dfs_for_angantyr_model(angantyr_model = 1, seeds=False, energies=True, param=energies_param)
dfs2 = create_all_dfs_for_angantyr_model(angantyr_model = 2, seeds=False, energies=True, param=energies_param)

columns = dfs1[0].columns

basic_energies = [int(df['energy'][0]) for df in dfs1]
log_energies = [np.log(energy) for energy in basic_energies]


#%% CHI-SQ AT FIXED ENERGY


model = 1

with open( f"Pickles/Angantyr{model}/{energies_param}_mean_values", "rb" ) as f:
	avgs = pickle.load(f)
with open( f"Pickles/Angantyr{model}/{energies_param}_errors", "rb" ) as f:
	stdevs = pickle.load(f)
    
if model == 1:
    dfs = dfs1
if model == 2:
    dfs = dfs2


sigma_mins = []
param_ranges = []
for num in range(len(dfs)):

    fig, ax = plt.subplots(figsize=[20, 12])
    xscale = 'linear'

    obs_param = 'sigma_D'

    df = dfs[num]
    df = df.sort_values(by=obs_param, ascending=True)
    
    ax.axvline(x = avgs[num], label='Mean Sigma (Seeds)', color='green')
    ax.axvline(x = avgs[num] + stdevs[num], label='1 stdev', color='limegreen' )
    ax.axvline(x = avgs[num] - stdevs[num],  color='limegreen' )
    ax.axvline(x = avgs[num] + 2*stdevs[num], label='2 stdevs', color='orange' )
    ax.axvline(x = avgs[num] - 2*stdevs[num],  color='orange' )

    
    energy = df['energy'][0]
    alpha = df['alpha']
    sigma_d = df['sigma_D']
    k_0 = df['k_0']
    chi = df['chi-sq/Ndf']
    
    xdata = df[obs_param]
    
    ax2 = ax.twinx()
    

    # Error bars    
    p1, p2, hline = find_intersection_points(xdata, list(chi), hline=None, ax=ax)
    param_range = [p1[0], p2[0]]
    param_range = round_list_to_decimal(param_range, decimals=2)
    param_ranges.append(param_range)
    
    # Plotting
    ax.plot(xdata, chi, marker = 'o', color = 'steelblue', label = f'Angantyr {model}')
    ax.axhline(y=hline, color='red', label = 'Uncertainty line')

    
    # Minimum Sigma
    sorted_chi_df = df.sort_values(by='chi-sq/Ndf', ascending=True)
    sorted_chi_df = sorted_chi_df.reset_index()
    min_sigma = sorted_chi_df['sigma_D'][0]
    sigma_mins.append(min_sigma)
    
    
    
    # Plotting Format
    title = f'Energy = {energy} GeV'
    ylabel = 'Chi-sq / Ndf'
    plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel = obs_param)
    
    plt.show(fig)
    plt.close(fig)
    

#%%

fig, ax = plt.subplots(figsize=[20, 12])
color = 'steelblue'
ecolor = 'blue'


above_errors = [_range[1] - sigma for sigma, _range in zip(sigma_mins, param_ranges)]
below_errors = [sigma - _range[0] for sigma, _range in zip(sigma_mins, param_ranges)]

line = ax.errorbar(basic_energies, sigma_mins, [below_errors, above_errors], fmt='-o', color=color, ecolor=ecolor, capsize=8, label='chi method')
line2 = ax.errorbar(basic_energies, avgs, stdevs, fmt='-o', color='orange', ecolor='red', label='stdevs', capsize=8)


title = f'Angantyr {model}'
xscale = 'log'
ylabel = obs_param
plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel = 'Energy (GeV)')

