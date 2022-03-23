
"""
Created on Wed Feb 16 18:21:41 2022

@author: felipeabedrapo
"""


# Libraries
import matplotlib.pyplot as plt
import os


# Own Functions
from reading_utilities import open_dfs
from reading_utilities import plot_column_with_min_chi
from reading_utilities import plot_columns_df
from reading_utilities import linear_fit
from reading_utilities import create_df_from_seed
from reading_utilities import plot_mean_values_from_dfs


#%% CREATING DATAFRAMES

seeds = [201 + i for i in range(24)]
directories = [f'seed_{seed}/' for seed in seeds]
dfs = [create_df_from_seed(seed, max_chi=100000) for seed in seeds]


#%% SAVING DFS

for df in dfs:
    seed = df['seed'][0]
    filename = f'seed_{int(seed)}'
    directory = 'saved_dfs'
    df.to_csv(f'{directory}/{filename}.csv')



#%% LOADING DATAFRAMES

Angantyr_model = 2

seeds = [2 * 10**(Angantyr_model + 1) + i for i in range(1, 24)]
# seeds.append(-1)
# seeds.append(12345)
dfs = open_dfs(seeds)


#%% PLOTTING MINIMUM CHI-SQUARED

fig, ax = plt.subplots(figsize= [20, 12.6])

energies = list(dfs[0]['Energy (GeV)'])
column = 'k_0'
x, y = plot_column_with_min_chi(column, dfs, energies, seeds, minE=0, ax=ax)
# y_data = linear_fit(x,y, ax=ax)

# fig.savefig(f'Angantyr_{Angantyr_model}/Parameters/linear_k_0')


#%% PLOTTING SPECFIC SEEDS


y_column = 'sigma_D'
x_column = 'Energy (GeV)'

colors = ['salmon', 'steelblue', 'teal', 'green', 'purple', 'lightblue', 'red']
colors.extend(['pink', 'grey', 'black', 'khaki', 'chocolate', 'orange'])

fig, ax = plt.subplots(figsize= [20, 12.6])

seeds_to_plot = [2000 + i for i in range(1,24)]
linestyle = ''


for color, df in zip(colors, dfs):
    if df['seed'][0] not in seeds_to_plot: continue
    plot_columns_df(df, x_column, y_column, ax=ax, color=color, linestyle=linestyle)
    

ax.legend(fontsize=20)



#%% MEAN & VARIANCE ANALYSIS

from reading_utilities import plot_mean_values_from_dfs
from reading_utilities import find_mean_stdev_at_energy
from reading_utilities import plot_sim_vs_data_cross_sec


fig, ax = plt.subplots(figsize= [20, 12.6])

parameters = ['chi-sq/Ndf', 'k_0', 'alpha', 'sigma_D']

param = parameters[0]
info = plot_mean_values_from_dfs(param, dfs, ax=ax, xscale='log' )


# ax.set_xlim([100, 10000 + 100])
# ax.set_ylim([1, 3])





# ============= OVERLAYING PLOTS ================================================================
# # param = param.replace('/', '_')
# # plt.savefig(f'Angantyr_1/Parameters/{param}.png')
# 
# 
# # ax2 = ax.twinx()
# # plot_mean_values_from_dfs('alpha', dfs, ax=ax, xscale='log')
# # plot_mean_values_from_dfs('sigma_D', dfs, ax=ax2, xscale='log', color='salmon', ecolor='red')
# # ax.legend(fontsize = 20)
# 
# # fig.savefig('Angantyr_1/Parameters/overlayed.png')
# 
#     
# =============================================================================
    



#%% DATA VS SIMULATION
    
columns = dfs[0].columns
cross_secs = ['tot_cross_sec', 'non_diff_cross_sec', 'double_diff_cross_sec',
              'wounded_target', 'wounded_projectile', 'elastic']

cross_sec = 'wounded_target'
cross_sec = cross_secs[0]

overwrite = False

for cross_sec in cross_secs:
    print(f'Plot for {cross_sec}')
    fig, seeds = plot_sim_vs_data_cross_sec(cross_sec, dfs, scale='log', ax=None)
    if os.path.exists(f'Angantyr_{Angantyr_model}/{cross_sec}_{len(seeds)}_seeds.png') == True:
        overwrite = input('Overwrite? True/False: ')
        
    if overwrite == True:
        fig.savefig(f'Angantyr_{Angantyr_model}/{cross_sec}_{len(seeds)}_seeds.png')
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






