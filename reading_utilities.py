#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:54:06 2022

@author: felipeabedrapo
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress




################ FUNCTIONS ON DATA ################################


def find_indexes_for_data(data):
    for i, line in enumerate(data):
        if 'Resulting cross sections (target value)' in line:
            indexes = [i + k for k in range(1, 15)]
            return indexes
        
def filter_df_by_column_max(df, column_name, _max):
    print(f'Old length: {len(df)}')
    print(f'Filtering by maximum {_max} in {column_name}')
    df = df[df[column_name] <= _max]
    print(f'New length: {len(df)}')
    print('---------------------------------')
    df.reset_index(drop = True, inplace = True)
    return df


def get_all_files_in_directory(directory):
    from os import walk
    filenames = next(walk(directory), (None, None, []))[2]
    files = [filename for filename in sorted(filenames)]
    return sorted(files)


def extract_info_from_given_params_file(filename, directory = 'fitted_params', seed = None):
    energy = filename.split('.')[0]
    with open (f"./{directory}{filename}", "r") as myfile:
        data = myfile.read().splitlines()
        
    # Not reading bad files
    file_size = os.path.getsize(f'{directory}{filename}')
    if file_size < (15 * 10**3):
        print(f'File {filename} has size {file_size}. Skipping...')
        return None
    
    energy = int(energy)
    indexes = find_indexes_for_data(data)
    row = [energy]  
    for i, index in enumerate(indexes):
        line = data[index][30:]
        if 'AXB diffractive:' in data[index] or 'elastic b-slope:' in data[index]:
            continue
        info = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(info) == 3:
            sim, pythia_data, extra = info
            row.append(float(sim)); row.append(float(pythia_data))
            
        if len(info) == 2:
            sim, pythia_data = info
            row.append(float(sim)); row.append(float(pythia_data))
            
        if len(info) == 1:
            row.append(float(info[0]))

    row.append(seed) 
    
    return row
    
    
def extract_info_from_seed_file(seed, filename, directory):
    energy = filename.split('.')[0]
    # print(f'Extracting info from file {filename}...')
    with open (f"./{directory}{filename}", "r") as myfile:
        data = myfile.read().splitlines()
        
    file_size = os.path.getsize(f'{directory}{filename}')
    if file_size < (10 * 10**3):
        print(f'File {filename} has size {file_size}. Skipping...')
        return None
            
    energy = int(energy); seed = int(seed)
    indexes = find_indexes_for_data(data)
    row = [energy]  
    for i, index in enumerate(indexes):
        line = data[index][30:]
        if 'AXB diffractive:' in data[index] or 'elastic b-slope:' in data[index]:
            continue
        info = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(info) == 3:
            sim, pythia_data, extra = info
            row.append(float(sim)); row.append(float(pythia_data))
            
        if len(info) == 2:
            sim, pythia_data = info
            row.append(float(sim)); row.append(float(pythia_data))
            
        if len(info) == 1:
            row.append(float(info[0]))

    row.append(seed)
    
    return row


def add_rows_to_df(df, rows, e_min=0, e_max=100000):
    for row in rows:
        if row == None: continue 
        energy = int(row[0])
        if energy < e_min or energy > e_max: continue
        row_series = pd.Series(row, index = df.columns)
        df = df.append(row_series, ignore_index=True)
        
    return df
        
        
        
def create_df_from_directory(directory, e_min=0, e_max=100000, max_chi = 1000):
    columns = ['Energy (GeV)', 'sim_total', 'data_total' ]
    columns.extend(['sim_non_diff', 'data_non_diff', 'sim_double_diff', 'data_double_diff'])
    columns.extend(['sim_wounded_target', 'data_wounded_target', 'sim_wounded_projectile', 'data_wounded_projectile', 'sim_elastic', 'data_elastic', 'chi-sq/Ndf', 'sigma_D', 'k_0', 'alpha', 'seed'])
    df = pd.DataFrame(columns = columns)
    
    filenames = get_all_files_in_directory(directory=directory)
    
    rows = [extract_info_from_given_params_file(filename, directory=directory) for filename in filenames]
    df = add_rows_to_df(df, rows, e_min=e_min, e_max=e_max)
    
    df = filter_df_by_column_max(df, 'chi-sq/Ndf', _max=max_chi)
    df.sort_values(['Energy (GeV)'], inplace=True)
    
    return df
    
    
        
def create_df_from_seed(seed, e_min=0, e_max=100000, max_chi = 10000):
    columns = ['Energy (GeV)', 'sim_tot_cross_sec', 'data_tot_cross_sec' ]
    columns.extend(['sim_non_diff_cross_sec', 'data_non_diff_cross_sec', 'sim_double_diff_cross_sec', 'data_double_diff_cross_sec'])
    columns.extend(['sim_wounded_target', 'data_wounded_target', 'sim_wounded_projectile', 'data_wounded_projectile', 'sim_elastic', 'data_elastic', 'chi-sq/Ndf', 'sigma_D', 'k_0', 'alpha', 'seed'])
    df = pd.DataFrame(columns = columns)
    
    directory = f'./seed_{seed}/'
    filenames = get_all_files_in_directory(directory=directory)

    rows = [extract_info_from_seed_file(seed, filename, directory) for filename in filenames]
    df = add_rows_to_df(df, rows, e_min=e_min, e_max=e_max)
    
    df = filter_df_by_column_max(df, 'chi-sq/Ndf', _max=max_chi)
    df.sort_values(['Energy (GeV)'], inplace=True)
    
    return df
    

def plot_columns_df(df, x_column, y_column, ax=None, color='steelblue', linestyle = ''):
    if ax == None:
        fig, ax = plt.subplots(figsize= [20, 12.6])
    fontsize = 20
    ax.set_xlabel(x_column, fontsize=fontsize)
    ax.set_ylabel(y_column, fontsize=fontsize)
    ax.plot(df[x_column], df[y_column], marker='o', linestyle=linestyle, color = color, label = f'Seed: {int(df["seed"][0])}')    
    
    
def find_chi_for_energy_in_df(energy, df):
    chi = df.loc[(df['Energy (GeV)'] == energy)]['chi-sq/Ndf']
    try:
        return float(chi)
    except:
        print(f'Row not found for energy {energy} in seed: {df["seed"][0]}')
        return 12345
    

def get_min_chi_row_at_energy(dfs, energy):
    chis = [find_chi_for_energy_in_df(energy, df) for df in dfs]
    index = chis.index(min(chis))
    return dfs[index].loc[ dfs[index]['Energy (GeV)'] == energy]


def plot_column_with_min_chi(column, dfs, energies, seeds, minE=0, ax=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    x_data = [energy for energy in energies if (energy > minE)]
    y_data = [float(get_min_chi_row_at_energy(dfs, energy)[column]) for energy in x_data]

    fontsize = 20
    ax.set_xlabel('Energy (GeV)', fontsize=fontsize)
    ax.set_ylabel(column, fontsize=fontsize)
    seed_range = f'[{seeds[0]}...{seeds[-1]}]'
    title = f'Plotting minimum Chi-Sq from {seed_range} {len(seeds)} seeds'
    ax.set_title(title, fontsize=fontsize)
    ax.plot(x_data, y_data, marker='o', linestyle='-')
    
    return x_data, y_data


def find_mean_stdev_at_energy(param, energy, seeds, dfs):
    param_values = []
    for df in dfs:
        seed = df['seed'][0]
        if seed not in seeds: continue
        row = df.loc[df['Energy (GeV)'] == energy]
        try:
            value = float(row[param])
        except TypeError:
            print(f'No value found for enery {energy} seed {seed}')
        param_values.append(value)
        
    mean = np.mean(param_values)
    stdev = np.std(param_values)
        
    return round(mean, 4), round(stdev, 4)
        
    

################ FITTING FUNCTIONS ################################


def linear_fit(x_data, y_data, ax=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    slope, intercept, r, p, se = linregress(x_data, y_data)
    y_fit = np.array(x_data) * slope + intercept
    fit_line = ax.plot(x_data, y_fit, marker='o')
    ax.legend((fit_line), [f'y = {slope:.2e}x + {round(intercept, 3)}; r={round(r,3)}'], fontsize=20, loc='upper left')
    
    return y_fit



def get_linear_fit_values_at_energy(energy_eval, energies, dfs):
    param_values = []
    for param in ['sigma_D', 'k_0', 'alpha']:
        y_data = [float(get_min_chi_row_at_energy(dfs, energy)[param]) for energy in energies]
        x_data = energies
        slope, intercept, r, p, se = linregress(x_data, y_data)
        y_fit = np.array(x_data) * slope + intercept
        energy_index = energies.index(energy_eval)
        param_values.append(y_fit[energy_index])
    sigma_D, k_0, alpha = param_values
    print(sigma_D, k_0, alpha)
    return round(sigma_D, 3), round(k_0, 3), round(alpha, 3)


def open_dfs(seeds, directory='saved_dfs'):
    filenames = get_all_files_in_directory(directory)
    dfs = []
    for filename in filenames:
        if '.DS_Store' in filename: continue
        seed = int(filename.split('.')[0].split('_')[1])
        if seed in seeds:
            print(f'Opening {filename}...')
            df = pd.read_csv(f'{directory}/{filename}') 
            dfs.append(df)
            
    return dfs





##################### PLOTTING ##########################################



def plot_mean_values_from_dfs(param, dfs, ax=None, energies=None, fontsize=20, xscale='log', color='steelblue', ecolor='blue'):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    if energies == None: energies = dfs[0]['Energy (GeV)']
    seeds = [int(df['seed'][0]) for df in dfs]
    mean_values = []; stdevs = []
    for energy in energies:
        mean, stdev = find_mean_stdev_at_energy(param, energy, seeds, dfs)
        mean_values.append(mean)
        stdevs.append(stdev)
            
    ax.errorbar(energies, mean_values, yerr=stdevs, fmt='-o', color=color, ecolor=ecolor, label=param)
    ax.set_xscale(xscale)
    ax.set_xlabel('Energy (GeV)', fontsize=fontsize)
    ax.set_ylabel(param, fontsize=fontsize)
    seed_range = f'[{seeds[0]},...,{seeds[-1]}]'    
    title = f'Plotting minimum Chi-Sq from {seed_range} {len(seeds)} seeds'
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize= fontsize-5)
    ax.legend(fontsize = fontsize)
    
    
    return mean_values, stdevs, energies


def plot_sim_vs_data_cross_sec(cross_sec, dfs, ax=None, scale='log', energies=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    sim_cross_sec = 'sim_' + cross_sec
    data_cross_sec = 'data_' + cross_sec
    data_y = list(dfs[0][data_cross_sec])
    fontsize = 20
    
    if energies == None: energies = list(dfs[0]['Energy (GeV)'])
    seeds = [int(df['seed'][0]) for df in dfs]
    
    mean_values = []; stdevs = []
    for energy in energies:
        mean, stdev = find_mean_stdev_at_energy(sim_cross_sec, energy, seeds, dfs)
        mean_values.append(mean)
        stdevs.append(stdev)
    
    ax.plot(energies, data_y, marker='o', linestyle='-', color = 'orange', label = 'Data')
    ax.errorbar(energies, mean_values, yerr=stdevs, fmt='-o', color='steelblue', ecolor='blue', label='Simulation')
    ax.legend(fontsize = fontsize)
    ax.set_xscale(scale)
    ax.set_xlabel('Energy (GeV)', fontsize=fontsize)
    ax.set_ylabel(cross_sec, fontsize=fontsize)
    seed_range = f'[{seeds[0]},...,{seeds[-1]}]'    
    title = f'Plotting minimum Chi-Sq from {seed_range} {len(seeds)} seeds'
    ax.set_title(title, fontsize=fontsize)
            
    return fig, seeds





