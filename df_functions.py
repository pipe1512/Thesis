#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:26:16 2022

@author: felipeabedrapo
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick


def sort_xlist_ylist(xdata, ydata):
    temp_list = []
    for x, y in zip(xdata, ydata):
        temp_list.append((x,y))
    temp_list = sorted(temp_list)
    xlist = [elem[0] for elem in temp_list]
    ylist = [elem[1] for elem in temp_list]
    
    return xlist, ylist


def round_list_to_decimal(mylist, decimals = 2):
    new_list = [round(elem, decimals) for elem in mylist]
    return new_list

    
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



def find_mean_stdev_at_energy(param, energy, dfs):
    param_values = []
    for df in dfs:
        seed = df['seed'][0]
        row = df.loc[df['energy'] == energy]
        try:
            value = float(row[param])
        except TypeError:
            # print(f'energy: {energy} row: {row} param:{param}')
            # print(f'value : {row[param]} type: {type(row[param])}')
            print(f'No value found for enery {energy} seed {seed}')
            print('---------------------------------------')
        param_values.append(value)
        
    mean = np.mean(param_values)
    stdev = np.std(param_values)
        
    return round(mean, 4), round(stdev, 4)


def get_means_stdevs_for_df_param(param, dfs):
    energies = [int(energy) for energy in dfs[0]['energy']]
    energies = sorted(energies)
    mean_values = []; stdevs = []
    for energy in energies:
        mean, stdev = find_mean_stdev_at_energy(param, energy, dfs)
        mean_values.append(mean)
        stdevs.append(stdev)
        
    return mean_values, stdevs


def get_difference_in_cross_sec(info, percentage=False):
    mean_values, stdevs, data_y = info
    if percentage == False:
        diff = [sim_val - data_val for data_val, sim_val in zip(data_y, mean_values)]
    else:
        diff = [(sim_val - data_val) / data_val * 100 for data_val, sim_val in zip(data_y, mean_values)]
    return diff




#################################### PLOTTING ###################################

def plot_columns_df(df, x_column, y_column, ax=None, color='steelblue', linestyle = ''):
    if ax == None:
        fig, ax = plt.subplots(figsize= [20, 12.6])
    fontsize = 20
    ax.set_xlabel(x_column, fontsize=fontsize)
    ax.set_ylabel(y_column, fontsize=fontsize)
    ax.plot(df[x_column], df[y_column], marker='o', linestyle=linestyle, color = color, label = f'Seed: {int(df["seed"][0])}')    
    
    


def plot_mean_values_from_dfs(param, dfs, ax=None, energies=None, fontsize=20, xscale='log', color='steelblue', ecolor='blue'):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    if energies == None: energies = dfs[0]['energy']
    seeds = [int(df['seed'][0]) for df in dfs]
    seeds = sorted(seeds)
    seed_range = f'[{seeds[0]},...,{seeds[-1]}]' 
    
    mean_values, stdevs = get_means_stdevs_for_df_param(param, dfs)
    line = ax.errorbar(energies, mean_values, yerr=stdevs, fmt='-o', color=color, ecolor=ecolor)
    
    return line, mean_values, stdevs, energies, seed_range




def plot_sim_vs_data_cross_sec(cross_sec, dfs, ax=None, energies=None, color='steelblue', ecolor = 'blue', label='Data'):
    
    # Getting Data
    sim_cross_sec = 'sim_' + cross_sec
    data_cross_sec = 'data_' + cross_sec
    data_y = list(dfs[0][data_cross_sec])
    mean_values, stdevs = get_means_stdevs_for_df_param(sim_cross_sec, dfs)

    # Basic Info
    if energies == None: energies = list(dfs[0]['energy'])
    seeds = [int(df['seed'][0]) for df in dfs]
    seeds = sorted(seeds)
    seed_range = f'[{seeds[0]},...,{seeds[-1]}]'     
    
    # Plotting
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    ax.plot(energies, data_y, marker='o', linestyle='-', color = 'orange', label = label)
    line = ax.errorbar(energies, mean_values, yerr=stdevs, fmt='-o', color=color, ecolor=ecolor)
    data = (mean_values, stdevs, data_y)

    return line, seed_range, data


def plot_cross_sec_as_percentage_of_total(cross_sec, dfs, ax=None, color='limegreen', label='Data'):
    
    
    # Data percentages
    data_total = dfs[0]['data_total']
    data_cross_sec = 'data_' + cross_sec
    data_values = dfs[0][data_cross_sec]
    data_y = [value / total * 100 for total, value in zip(data_total, data_values)]
    
    # Simulated percentages
    sim_total_mean, sim_total_stdev = get_means_stdevs_for_df_param('sim_total', dfs)
    sim_cross_sec_string = 'sim_' + cross_sec
    sim_values, stdevs = get_means_stdevs_for_df_param(sim_cross_sec_string, dfs)
    y = [value / total * 100 for total, value in zip(sim_total_mean, sim_values)]
    
    # Basic Info
    energies = list(dfs[0]['energy'])
    seeds = [int(df['seed'][0]) for df in dfs]
    seeds = sorted(seeds)

    
    # Plotting
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    ax.plot(energies, data_y, marker='o', linestyle='-', color = 'orange', label=label)
    line = ax.plot(energies, y, marker='o', linestyle='-', color = color)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    return line
                

def plot_chi_square_at_fixed_param(fixed_param, df, ax=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    energy = df['energy']
    chi_values = df['chi-sq/Ndf']
    params = ['sigma_D', 'k_0', 'alpha']
    params.remove(fixed_param)
    
    values1 = df[fixed_param]
    # values2 = df[params[1]]
    
    line = ax.plot(values1, chi_values, linestyle = '', marker='o')
    # line2 = ax.plot(values2, chi_values, linestyle = '', marker='o', label=f'{params[1]}')
    
    
    return line

def scatter_plot_param__from_dfs(param, dfs, ignore_energy=None, ax=None, plot=False):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    energies = []; param_values = []
    for df in dfs:
        energies += list(df['energy'])
        param_values += list(df[param])
    xdata = []; ydata = []
    for energy, value in zip(energies, param_values):
        if energy == ignore_energy: 
            continue
        xdata.append(np.log(energy))
        ydata.append(value)
        
        
    # print(len(xdata), len(ydata))
    if plot == True:
        ax.scatter(np.exp(xdata), ydata)
    
    return energies, param_values, xdata, ydata
    
        
    


#################################### FITTING ###################################


def linear_fit(energies, y_data, color = 'steelblue', ax=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    x_data = [np.log(energy) for energy in energies]
    print(x_data)
    slope, intercept, r, p, se = linregress(x_data, y_data)
    y_fit = np.array(x_data) * slope + intercept
    fit_line = ax.plot(energies, y_fit, marker='o', color=color)
    # ax.legend((fit_line), [f'y = {slope:.2e}x + {round(intercept, 3)}; r={round(r,3)}'], fontsize=20, loc='upper left')
    fit_info = (slope, intercept, r)
    
    return y_fit, fit_info, fit_line



def exponential_fit(x_data, y_data, p0=(100,0.1,5), ax=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    
    def monoExp(x, m, t, b):
        return m * np.exp(-t * x) + b
    
    
    # perform the fit
    p0 = p0 # start with values near those we expect
    params, cv = curve_fit(monoExp, x_data, y_data, p0)
    m, t, b = params
    # sampleRate = 20_000 # Hz
    # tauSec = (1 / t) / sampleRate
    
    # determine quality of the fit
    squaredDiffs = np.square(y_data - monoExp(x_data, m, t, b))
    squaredDiffsFromMean = np.square(y_data - np.mean(y_data))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")
    
    # plot the results
    # ax.plot(x_data, y_data, '.', label="data")
    ax.plot(x_data, monoExp(x_data, m, t, b), '--', label="fitted", color='salmon')
    
    # inspect the parameters
    print(f"Y = {m} * e^(-{t} * x) + {b}")
    # print(f"Tau = {tauSec * 1e6} µs")
    
    
    
def logarithmic_fit(xs, ys, ax=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    
    slope, intercept, r, p, se = linregress(xs, np.log(ys))
    y_fit = np.array(xs) * slope + intercept
    fit_line = ax.plot(xs, y_fit, marker='o')
    ax.legend((fit_line), [f'y = {slope:.2e}x + {round(intercept, 3)}; r={round(r,3)}'], fontsize=20, loc='upper left')

    
def poly_degree_2_fit(x, y, ax=None):
    if ax == None: fig, ax = plt.subplots(figsize= [20, 12.6])
    
    fit_data = np.polyfit(x, y, 2, full=True, cov=True)
    a, b, c = fit_data[0]
    residuals = fit_data[1]
    fit_y = [a * x_i**2 + b*x_i + c for x_i in x]
    # line = ax.plot(energies, fit_y, color='red', marker='o')
    return x, fit_y, fit_data



def exp_decrease_func(x, a, b):
    return a + (b/x)


def round_list_to_decimal(mylist, decimals = 2):
    new_list = [round(elem, decimals) for elem in mylist]
    return new_list


def const_plus_inv(x,a,b):
    if type(x) == list:
        x = np.array(x)
    return a + (b/x)

def const_plus_neg_exp(x, a, b, c):
    if type(x) == list:
        x = np.array(x)
        # print('Changed to array')
    return a + (b*(np.exp(-x)))


def logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c) ) )


def exp_decay(x, a, b, c):
    if type(x) == list:
        x = np.array(x)
        # print('Changed to array')
    return np.exp(-a*(x - b)) + c


####################### INTERSECTION #######################



    
def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return [px, py]
    
def find_intersection_points(xdata, chis, ax=None, hline=None):
    if hline == None:
        hline = min(chis) + 1
    else:
        if min(chis) < 1:
            hline = 1
        else:
            hline = min(chis) + 1
    xdata = list(xdata)
    chis = list(chis)
    point1 = None; point2 = None
    point3 = None; point4 = None
    # print(f'hline: {hline}')
    for i in range(len(chis) - 1):
        if chis[i] >= hline and chis[i+1] < hline:
            point1 = (xdata[i], chis[i])
            point2 = (xdata[i+1], chis[i+1])
            
        if chis[i] <= hline and chis[i+1] > hline:
            point3 = (xdata[i], chis[i])
            point4 = (xdata[i+1], chis[i+1])
           
    if point1 == None:
        point1 = (xdata[0], chis[0])
        point2 = point1
        print(f'No point1 found. Using chi[0] as error: {point1}')
        first_intersec = (xdata[0], chis[0])
    else:
        x1, y1 = point1; x2, y2 = point2
        x3, y3 = (min(xdata), hline); x4, y4 = (max(xdata), hline)
        first_intersec = findIntersection(x1, y1, x2, y2, x3, y3, x4, y4)
        # print(round_list_to_decimal(first_intersec))
    
    
    if point4 == None:
        point4 = (xdata[-1], chis[-1])
        print(f'No point1 found. Using chi[0] as error: {point1}')
        second_intersec = (xdata[-1], chis[-1])
    else:
        x1, y1 = point3; x2, y2 = point4
        x3, y3 = (min(xdata), hline); x4, y4 = (max(xdata), hline)
        second_intersec = findIntersection(x1, y1, x2, y2, x3, y3, x4, y4)
        # print(round_list_to_decimal(second_intersec))
    
    if ax != None:
        ax.plot(first_intersec[0], first_intersec[1], marker='x', linestyle='', color='red', markersize = 20, label = 'Intersection Point')
        ax.plot(second_intersec[0], second_intersec[1], marker='x', color='red', markersize = 20)
        
    return first_intersec, second_intersec, hline 







