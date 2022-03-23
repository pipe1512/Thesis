#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:26:40 2022

@author: felipeabedrapo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit


from reading_files import create_all_dfs_for_angantyr_model
from df_functions import plot_mean_values_from_dfs
from df_functions import plot_sim_vs_data_cross_sec
from df_functions import get_difference_in_cross_sec
from df_functions import plot_cross_sec_as_percentage_of_total
from df_functions import poly_degree_2_fit
from df_functions import exp_decay
from df_functions import round_list_to_decimal
from df_functions import scatter_plot_param__from_dfs
from df_functions import sort_xlist_ylist
from df_functions import find_intersection_points


def plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel = 'Energy (GeV)'):
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize = fontsize-5)
    ax.legend(fontsize = fontsize)

        


#%% LOADING DFS


seeds = True
energies = False

dfs1 = create_all_dfs_for_angantyr_model(angantyr_model = 1, seeds=seeds, energies=energies)
dfs2 = create_all_dfs_for_angantyr_model(angantyr_model = 2, seeds=seeds, energies=energies)

columns = dfs1[0].columns


#%% Exponential Fitting

fig, ax = plt.subplots(figsize=[20, 12])


param = 'k_0'
xscale = 'log'
model = 1

if model == 2:
    dfs = dfs2
if model == 1:
    dfs = dfs1
basic_energies = list(dfs[0]['energy'])


# Scatter Plot
if model == 1:
    ignore_energy = 10
else: ignore_energy = 0
info = scatter_plot_param__from_dfs(param, dfs, ignore_energy=ignore_energy, plot=True, ax=ax)
energies, param_values, xdata, ydata = info

# Mean Values
data = plot_mean_values_from_dfs(param, dfs, ax=ax, xscale='linear')
line, mean_values, stdevs, energies, seed_range = data
line[0].set_label(f'Angantyr {model} average')


#### EXPONENTIAL DECAY FIT
func = exp_decay
popt, pcov = curve_fit(func, xdata, ydata)
a,b, c = round_list_to_decimal(popt)
fitLabel = f'Fit: a={a}, b={b}, c={c}'
xdata, ydata = sort_xlist_ylist(xdata, ydata)
ax.plot(np.exp(xdata), func(xdata, *popt), color='red', marker='o', linestyle='-', zorder=30, label=fitLabel)

title = '$np.exp(-a*(log(x) - b)) + c$'


####### PICKLING
# =============================================================================
# fit_values = func(np.log(basic_energies), *popt)
# with open( f"Pickles/Angantyr{model}/{param}_fit", "wb" ) as f:
#  	pickle.dump(fit_values, f)
# ax.plot(basic_energies, fit_values, color='red', marker='o', zorder=20)
# 
#     
# avg_values = mean_values
# with open( f"Pickles/Angantyr{model}/{param}_mean_values", "wb" ) as f:
# 	pickle.dump(avg_values, f)
# 
# errors = stdevs
# with open( f"Pickles/Angantyr{model}/{param}_errors", "wb" ) as f:
# 	pickle.dump(errors, f)
# 
# =============================================================================


ylabel = param
plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel='Energy GeV')



#%% Poly-2 Fitting


fig, ax = plt.subplots(figsize=[20, 12])
title = ''


param = 'alpha'
xscale = 'log'

model = 2
if model == 2: dfs = dfs2[:30]
if model == 1: dfs = dfs1[:30]
basic_energies = list(dfs[0]['energy'])
log_basic_energies = [np.log(energy) for energy in basic_energies]



### Mean Values
data = plot_mean_values_from_dfs(param, dfs, ax=ax, xscale='log')
line, mean_values, stdevs, energies, seed_range = data
line[0].set_label(f'Angantyr {model} average')

### Scatter
scatter_info = scatter_plot_param__from_dfs(param, dfs, ax=ax)
all_energies, param_values, xdata, ydata = scatter_info
ax.scatter(all_energies, ydata, marker='o', color='steelblue')


#### POLYNOMIAL 2ND DEGREE FIT
xfit, yfit, fit_params = poly_degree_2_fit(xdata, ydata, ax=ax)
a,b,c = fit_params[0]
residuals = fit_params[1]
title = f'Fit model {model}: y = {round(a,2)}$log(x)^2$ + {round(b,2)}log(x) + {round(c,2)} | residual = {round(float(residuals),2)} '
all_energies, yfit = sort_xlist_ylist(all_energies, yfit)
ax.plot(all_energies, yfit, marker='o', color='salmon',label = 'Fit')



####### PICKLING
# fit_values = [a * energy**2 + b*energy + c for energy in list(log_basic_energies)]
# with open( f"Pickles/Angantyr{model}/{param}_fit", "wb" ) as f:
#  	pickle.dump(fit_values, f)
# ax.plot(basic_energies, fit_values, color='purple', marker='o', zorder = 1)

    
# avg_values = mean_values
# with open( f"Pickles/Angantyr{model}/{param}_mean_values", "wb" ) as f:
# 	pickle.dump(avg_values, f)

# errors = stdevs
# with open( f"Pickles/Angantyr{model}/{param}_errors", "wb" ) as f:
# 	pickle.dump(errors, f)


plotting_format(ax, xscale, param, title)


#%% PLOTTING SIM VS DATA FOR CROSS SECTIONS



fig, ax = plt.subplots(figsize=[20, 12])


cross_sec = 'elastic'

model = 1
if model == 2: dfs = dfs2[:30]
if model == 1: dfs = dfs1[:30]
basic_energies = list(dfs[0]['energy'])

# Angantyr 1
line, seed_range, info = plot_sim_vs_data_cross_sec(cross_sec, dfs, ax=ax, color='limegreen', ecolor='green')
mean_values, stdevs, data_y = info
line[0].set_label('Angantyr 1')

## Angantyr2
line2, seed_range2, info2 = plot_sim_vs_data_cross_sec(cross_sec, dfs2[:30], ax=ax, color='salmon', ecolor='red', label=None)
mean_values2, stdevs2, data_y2 = info2
line2[0].set_label('Angantyr 2')



### Fit 
# if model == 1: y_data = mean_values
# if model == 2: y_data = mean_values2
# log_energies, fity, fit_data = poly_degree_2_fit(np.log(basic_energies), y_data, ax=ax)
# a,b,c = fit_data[0]
# residuals = fit_data[1]
# ax.plot(np.exp(log_energies), fity, marker='o', label=f'Model {model} fit')
# title = f'Fit model {model}: y = {round(a,2)}$log(x)^2$ + {round(b,2)}log(x) + {round(c,2)} | residual = {round(float(residuals),2)} '


# Plotting format
ylabel = cross_sec + ' cross section'
plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel = 'Energy (GeV)')

# fig.savefig(f'./Plots/Comparison/{cross_sec}.png')

#%% CROSS SEC AS PERCENTAGE OF TOTAL


fig, ax = plt.subplots(figsize=[20, 12])
cross_sec = 'wounded_projectile'


line1 = plot_cross_sec_as_percentage_of_total(cross_sec, dfs1, ax=ax)
line2 = plot_cross_sec_as_percentage_of_total(cross_sec, dfs2, ax=ax, color='steelblue', label=None)


# Plotting Format
fontsize = 20
line1[0].set_label('Angantyr 1')
line2[0].set_label('Angantyr 2')

title = ''
ylabel = cross_sec + ' cross section'
plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel = 'Energy (GeV)')


#%% PLOT DIFFERENCES


fig, ax = plt.subplots(figsize=[20, 12])


percentage = True
diff1 = get_difference_in_cross_sec(info, percentage=percentage)
diff2 = get_difference_in_cross_sec(info2, percentage=percentage)
zeroes = [0] * len(diff1)
energies = dfs1[0]['energy']


ax.plot(energies, zeroes, linestyle='-', marker='o', label = 'Data', color='orange')
ax.plot(energies, diff1, linestyle='-', marker='o', label = 'Angantyr 1', color='limegreen')
ax.plot(energies, diff2, linestyle='-', marker='o', label = 'Angantyr 2', color='steelblue')

# Plotting Format
title = ''
ylabel = cross_sec + ' cross section'
plotting_format(ax, xscale, ylabel, title, fontsize=20, xlabel = 'Energy (GeV)')




#%% 3-D plot

model = 1
if model == 1:
    dfs = dfs1
if model == 2:
    dfs = dfs2

k0 = []
alpha = []
sigma = []
energies = []
full_data = []

for energy in basic_energies:
    print(f'Analyzing energy {energy}')
    if model == 1 and energy == 10:
        print(f'Skipping {energy}...')
        continue
    for df in dfs:
        row  = df.loc[df['energy'] == energy]
        if len(row) > 0:
            k0.append(float(row['k_0']))
            alpha.append(float(row['alpha']))
            sigma.append(float(row['sigma_D']))
            full_data.append( (energy, float(row['sigma_D']), float(row['k_0']), float(row['alpha'])) )
            energies.append(energy)
                             
            
            
log_energies = [np.log(energy) for energy in energies]



#%%


def filter_full_data_by_energy(full_data, energy):
    sigma_energy, k0_energy, alpha_energy, energies = [[] for i in range(4)]
    for elem in full_data:
        if elem[0] != energy:
            continue
        else:
            energies.append(elem[0])
            sigma_energy.append(elem[1])
            k0_energy.append(elem[2])
            alpha_energy.append(elem[3])
    return sigma_energy, k0_energy, alpha_energy, energies
            


sigma, k0, alpha, energies = filter_full_data_by_energy(full_data, 10000)
    
log_energies = [np.log(energy) for energy in energies]
    

#%%
            
            
#plotting a scatter for example

def scatter_plot_3d(xs, ys, zs, labels, view = [20, 45], _return = True, title=''):
    xlabel, ylabel, zlabel = labels
    fig = plt.figure(figsize = [12, 8])
    ax = fig.add_subplot(111,projection = "3d")
    ax.scatter(xs, ys, zs, c=zs, cmap='viridis', linewidth=0.5);
    ax.zaxis._set_scale('linear')
    ax.view_init(view[0], view[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    if _return == True:
        return fig
    else:
        plt.show(fig)
        plt.close(fig)


# =============================================================================
# for i in range(19):
#     view = [20, i*20]
#     xs = alpha
#     ys = sigma
#     zs = log_energies
#     labels = ['alpha', 'sigma', 'Log(Energies)']
#     fig = scatter_plot_3d(xs, ys, zs, labels, view = view, _return = True)
#     plt.savefig(f'Plots/Angantyr_{model}/3D/{labels[0]}_{labels[1]}_{labels[2]}/{view}')
#     plt.show(fig)
#     plt.close(fig)
# 
# for i in range(10):
#     view = [i * 10, 0]
#     xs = alpha
#     ys = sigma
#     zs = log_energies
#     labels = ['alpha', 'sigma', 'Log(Energies)']
#     fig = scatter_plot_3d(xs, ys, zs, labels, view = view, _return = True)
#     plt.savefig(f'Plots/Angantyr_{model}/3D/{labels[0]}_{labels[1]}_{labels[2]}/{view}')
#     plt.show(fig)
#     plt.close(fig)
# =============================================================================
    
    
    

for i in range(19):
    view = [20, i*20]
    xs = alpha
    ys = sigma
    zs = k0
    labels = ['alpha', 'sigma', 'k0']
    fig = scatter_plot_3d(xs, ys, zs, labels, view = view, _return = True)
    plt.savefig(f'Plots/Angantyr_{model}/3D/{labels[0]}_{labels[1]}_{labels[2]}/{view}')
    plt.show(fig)
    plt.close(fig)
    


for i in range(10):
    view = [i * 10, 0]
    xs = alpha
    ys = sigma
    zs = k0
    labels = ['alpha', 'sigma', 'k0']
    fig = scatter_plot_3d(xs, ys, zs, labels, view = view, _return = True)
    plt.savefig(f'Plots/Angantyr_{model}/3D/{labels[0]}_{labels[1]}_{labels[2]}/{view}')
    plt.show(fig)
    plt.close(fig)

                
                             
                             
                             
                             
                             
                             
            
            
