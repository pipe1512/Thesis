#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:38:03 2022

@author: felipeabedrapo

"""

import re
import pandas as pd
import numpy as np
import os


def get_all_filenames_in_directory(path_to_directory):
    if os.path.exists(f'{path_to_directory}') == False:
        print(f'Path does not exist: {path_to_directory}')
    path, folders, filenames = next(os.walk(path_to_directory), (None, None, []))
    filenames = [file for file in filenames if '.DS' not in file]
    return filenames

def sort_filenames(filenames):
    values = [int(file.split('.')[0]) for file in filenames]
    values = sorted(values)
    sorted_filenames = [f'{value}.txt' for value in values]
    return sorted_filenames

def sort_energy_filenames(filenames):
    values = [int(filename.split('.')[0].split('_')[1]) for filename in filenames]
    values = sorted(values)
    sorted_values = [f'energy_{value}' for value in values ]
    return sorted_values
        


def get_all_folders_in_directory(path_to_directory):
    if os.path.exists(f'{path_to_directory}') == False:
        print(f'Path does not exist: {path_to_directory}')
    path, folders, filenames = next(os.walk(path_to_directory), (None, None, []))
    return folders


def open_file_as_splitlines(path_to_directory, filename):
    with open (f"{path_to_directory}/{filename}", "r") as myfile:
        data = myfile.read().splitlines()
    return data
    

def find_starting_index_for_data(data):
    for i, line in enumerate(data):
        if 'Resulting cross sections (target value)' in line:
            return i

        
def get_params_cross_secs_from_file_data(data):
    starting_index = find_starting_index_for_data(data)
    indexes = [starting_index + k for k in range(1, 15)]
    
# =============================================================================
#     Order of row due to text file order:
#         # ['sim_total', 'data_total;
#         # 'sim_non_diff', 'data_non_diff', 'sim_double_diff', 'data_double_diff',
#         # 'sim_wounded_target', 'data_wounded_target', 
#         # 'sim_wounded_projectile', 'data_wounded_projectile', 'sim_elastic', 
#         # 'data_elastic', 'chi-sq/Ndf', 'sigma_D', 'k_0', 'alpha']       
# =============================================================================

    row = []  
    for i, index in enumerate(indexes):
        line = data[index][30:]
        if 'AXB diffractive:' in data[index] or 'elastic b-slope:' in data[index]:
            continue
        info = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(info) == 3:
            sim, pythia_data, extra_data = info
            row.append(float(sim)); row.append(float(pythia_data))
            
        if len(info) == 2:
            sim, pythia_data = info
            row.append(float(sim)); row.append(float(pythia_data))
            
        if len(info) == 1:
            row.append(float(info[0]))
            
    return row
    


def create_new_df():
    columns = ['energy', 'sim_total', 'data_total',
                'sim_non_diff', 'data_non_diff', 
                'sim_double_diff', 'data_double_diff',
                'sim_wounded_target', 'data_wounded_target', 
                'sim_wounded_projectile', 'data_wounded_projectile', 
                'sim_elastic', 'data_elastic', 
                'chi-sq/Ndf', 'sigma_D', 'k_0', 'alpha', 'seed']
    df = pd.DataFrame(columns = columns)
    return df


def create_df_from_seed(seed, angantyr_model):
    
    df = create_new_df()
    
    path_to_files = f'./Data_Collection/Angantyr_{angantyr_model}/seed_{seed}'
    files = get_all_filenames_in_directory(path_to_files)
    files = sort_filenames(files)
    
    for file in files:
        energy = int(file.split('.')[0])
        data = open_file_as_splitlines(path_to_files, file)
        size = os.path.getsize(f"{path_to_files}/{file}")
        if size < 10 * 10**3:
            print(f'Skipping {file} with size {size}')
            continue
        params_and_cross_secs = get_params_cross_secs_from_file_data(data)
        row = [energy] + params_and_cross_secs + [seed]
        row_series = pd.Series(row, index = df.columns)
        row_df = row_series.to_frame().T
        df = pd.concat([df, row_df], axis=0, ignore_index=True)
        
    return df


def create_df_from_energy(energy, param, angantyr_model):
    
    df = create_new_df()
    seed = np.nan
    
    path_to_files = f'./Data_Collection/Angantyr_{angantyr_model}/{param}/energy_{energy}'
    files = get_all_filenames_in_directory(path_to_files)
    # files = sort_filenames(files)
    
    for file in files:
        energy = energy
        data = open_file_as_splitlines(path_to_files, file)
        size = os.path.getsize(f"{path_to_files}/{file}")
        if size < 10 * 10**3:
            print(f'Skipping {file} with size {size}')
            continue
        params_and_cross_secs = get_params_cross_secs_from_file_data(data)
        row = [energy] + params_and_cross_secs + [seed]
        row_series = pd.Series(row, index = df.columns)
        row_df = row_series.to_frame().T
        df = pd.concat([df, row_df], axis=0, ignore_index=True)
        
    return df


def create_all_dfs_for_angantyr_model(angantyr_model, seeds=True, energies=False, param=None):
    
    dfs = []
    if seeds == True:
        path_to_directory = f'Data_Collection/Angantyr_{angantyr_model}'
        folders = get_all_folders_in_directory(path_to_directory)
        folders = [folder for folder in folders if 'seed' in folder]
        folders = sorted(folders)
        
    if energies == True:
        if param == None:
            raise Warning(f'You have to choose which parameter (alpha/sigma_D/k_0) you want to access. param = {param}')
        path_to_directory = f'Data_Collection/Angantyr_{angantyr_model}/{param}'
        folders = get_all_folders_in_directory(path_to_directory)
        folders = [folder for folder in folders if 'energy' in folder]
        folders = sort_energy_filenames(folders)

        
    
    for folder in folders:
        folder_name = folder.split('.')[0]
        filename = folder_name.split('_')[1]
        print(f'Creating df for {filename}...')
        if seeds == True:
            df = create_df_from_seed(filename, angantyr_model)
        if energies == True:
            df = create_df_from_energy(filename, param, angantyr_model)
        dfs.append(df)
    
    return dfs





    
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    