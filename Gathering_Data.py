#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:20:36 2022

@author: felipeabedrapo
"""
import os
import datetime
import multiprocessing as mp
import numpy as np
import time
import pickle

def create_file_from_energy_and_seed(energy, seed, angantyr_model, overwrite = False):
    energy = int(energy)
    seed = int(seed)
    path_to_folder = f'./Data_Collection/Angantyr_{angantyr_model}'
    folder = f'seed_{seed}'
    filename  = f'{energy}'
    if os.path.exists(f'{path_to_folder}/{folder}/{filename}.txt') == True and overwrite == False:
        return f'{energy}.txt already exists'
    
    else:
        os.system(f"./main111 {energy} {seed} {angantyr_model} > {path_to_folder}/{folder}/{filename}.txt")
        currentDT = datetime.datetime.now()
        print(f'{filename}.txt" was created at {str(currentDT)}')
        
    return f'{filename}.txt" was created at {str(currentDT)}'


def create_file_for_given_params_at_energy(sigma_D, k_0, alpha, energy, angantyr_model, save_folder):
    
    # Saving parameters as floats and ints
    sigma_D = float(sigma_D)
    k_0 = float(k_0)
    alpha = float(alpha)
    energy = int(energy)
    angantyr_model = int(angantyr_model)
    
    # Defining path to Directory
    path_to_folder = f'./Data_Collection/Angantyr_{angantyr_model}/{save_folder}'
    folder = f'energy_{energy}'
    sigma_D, k_0, alpha = round_list([sigma_D, k_0, alpha])
    filename = f'{sigma_D}_{k_0}_{alpha}'
    
    # Running Program
    if os.path.exists(f'{path_to_folder}/{folder}/{filename}.txt') == True:
        return f'{energy}.txt already exists'
    else:
        os.system(f"./mymain111 {energy} {sigma_D} {k_0} {alpha} {angantyr_model} > {path_to_folder}/{folder}/{filename}.txt")
        currentDT = datetime.datetime.now()
        print(f'{filename}.txt" was created at {str(currentDT)}')
    
    return f'{filename}.txt" was created at {str(currentDT)}'
    



def get_params_from_linear_fit(energy, energies, dfs):
    from utility_functions import get_linear_fit_values_at_energy
    
    sigma_D, k_0, alpha = get_linear_fit_values_at_energy(energy, energies, dfs)
    params = [sigma_D, k_0, alpha]
    
    return params

        


######################## Creating Files for given Energies  ########################


def round_list(mylist, num_dec = 2):
    if type(mylist) != list and type(mylist) != np.ndarray:
        raise Warning(f'{type(mylist)}')
    new_list = [round(elem, num_dec) for elem in mylist]
    return new_list


def collect_data_for_random_seed(seed, startE, endE, filesNum, angantyr_model, logscale=True, overwrite=False):
    if __name__ == '__main__':
    
        
        path_to_folder = f'./Data_Collection/Angantyr_{angantyr_model}'
        folder = f'seed_{seed}'
            
        if os.path.exists(f'{path_to_folder}/{folder}') == False: 
            print(f'Making folder seed_{seed}...')
            os.system(f"mkdir {path_to_folder}/seed_{seed}") 
        
        currentDT = datetime.datetime.now()
        import time
        start_time = time.time()
        print(f'Starting time for seed {seed}: ', currentDT)
        
        # Energy is in GeV
        starting_energy = startE
        final_energy = endE
        files_to_create = filesNum
        
        
        # Deciding energies
        if logscale == True:
            energies = np.geomspace(starting_energy, final_energy, num=files_to_create)
            energies = round_list(energies, num_dec=0)
        else:
            step_size = 50
            energies = [starting_energy + step_size*i for i in range(files_to_create)]
    
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply(create_file_from_energy_and_seed, args=[energy, seed, angantyr_model, overwrite]) for energy in energies]
        
        # Step 3: Don't forget to close
        pool.close()    
        
        print(results)
        currentDT = datetime.datetime.now()
        print(f'Ending time: {currentDT}')
        
        
        folder = f'seed_{seed}'
        filename  = f'{int(energies[-1])}'
        os.system(f"open {path_to_folder}/{folder}/{filename}.txt")
        end_time = time.time()
        total_time = end_time - start_time
        print("Run time: ", total_time)
                
        
######################## MORE FUNCS  ################################################



def get_fits_avg_values_stdevs(model, param_dev):
    with open( f"Pickles/Angantyr{model}/k_0_fit", "rb" ) as f:
    	k0_fit = pickle.load(f)
        
    with open( f"Pickles/Angantyr{model}/energies", "rb" ) as f:
    	energies = pickle.load(f)
    
    with open( f"Pickles/Angantyr{model}/sigma_D_mean_values", "rb" ) as f:
    	sigmaD_avgs = pickle.load(f)
        
    with open( f"Pickles/Angantyr{model}/alpha_mean_values", "rb" ) as f:
    	alpha_avgs = pickle.load(f)
        
    with open( f"Pickles/Angantyr{model}/{param_dev}_errors", "rb" ) as f:
    	stdevs = pickle.load(f)
        
    return energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs


def varying_sigma_with_fixed_k0_and_alpha(model, num=2, files_num = 20, step=3, overwrite=False):
    if __name__ == '__main__':
        start_time = time.time()
        param_dev = 'sigma_D'
        energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs = get_fits_avg_values_stdevs(model, param_dev)
        
        # Setting params
        num = num
        files_num = files_num
        step = 4 * (stdevs[num] / files_num)
        for energy in [energies[num]]:
            
            # Setting Range
            sigma_D = sigmaD_avgs[num]
            alpha = alpha_avgs[num]
            k_0 = k0_fit[num]
            sig_range = [sigma_D + i*step for i in range(int(-files_num/2), int(files_num/2 + 1))]
            
            
            # Creating Folder
            path_to_folder = f'Data_Collection/Angantyr_{model}/{param_dev}'
            folder = f'energy_{energy}'
            if os.path.exists(f'{path_to_folder}/{folder}') == False: 
                print(f'Angantyr model {model} chosen...')
                print(f'Making folder energy_{energy}...')
                if os.path.exists(f'{path_to_folder}') == False: 
                    os.system(f"mkdir {path_to_folder}")
                os.system(f"mkdir {path_to_folder}/energy_{energy}") 
            
            
            # Running program
            results = []
            pool = mp.Pool(mp.cpu_count())
            for s in sig_range:
                s, alpha, k_0 = round_list([s, alpha, k_0], num_dec=2)
                filename = f'-{s}_-{alpha}_-{k_0}.txt'
                print(f'Creating file {filename} in {path_to_folder}/{folder}')
                if os.path.exists(f'{path_to_folder}/{folder}/{filename}') == True and overwrite == True: 
                    print(f'File {filename} exists. Skipping...')
                results.append(pool.apply(create_file_for_given_params_at_energy, args=[-s, -k_0, -alpha, energy, model, param_dev]))
            
            print(results)
            
            end_time = time.time()
            total_time = end_time - start_time
            print("Run time: ", total_time)
            
            
            
def varying_alpha_with_fixed_k0_and_sigma(model, num=2, files_num = 20, step=3, overwrite=False):
    if __name__ == '__main__':
        start_time = time.time()
        param_dev = 'alpha'
        energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs = get_fits_avg_values_stdevs(model, param_dev)
        
        # Setting params
        num = num
        files_num = files_num
        step = 4 * (stdevs[num] / files_num)
        for energy in [energies[num]]:
            
            # Setting Range
            sigma_D = sigmaD_avgs[num]
            alpha = alpha_avgs[num]
            k_0 = k0_fit[num]
            alpha_range = [alpha + i*step for i in range(int(-files_num/2), int(files_num/2 + 1))]
            
            
            # Creating Folder
            path_to_folder = f'Data_Collection/Angantyr_{model}/{param_dev}'
            folder = f'energy_{energy}'
            if os.path.exists(f'{path_to_folder}/{folder}') == False: 
                print(f'Angantyr model {model} chosen...')
                print(f'Making folder energy_{energy}...')
                if os.path.exists(f'{path_to_folder}') == False: 
                    os.system(f"mkdir {path_to_folder}")
                os.system(f"mkdir {path_to_folder}/energy_{energy}") 
            
            
            # Running program
            results = []
            pool = mp.Pool(mp.cpu_count())
            for alpha in alpha_range:
                sigma_D, alpha, k_0 = round_list([sigma_D, alpha, k_0], num_dec=2)
                filename = f'-{sigma_D}_-{alpha}_-{k_0}.txt'
                print(f'Creating file {filename} in {path_to_folder}/{folder}')
                if os.path.exists(f'{path_to_folder}/{folder}/{filename}') == True and overwrite == True: 
                    print(f'File {filename} exists. Skipping...')
                results.append(pool.apply(create_file_for_given_params_at_energy, args=[-sigma_D, -k_0, -alpha, energy, model, param_dev]))
            
            print(results)
            
            end_time = time.time()
            total_time = end_time - start_time
            print("Run time: ", total_time)
            
            


def varying_k0_with_fixed_alpha_and_sigma(model, num=2, files_num = 20, step=3, overwrite=False):
    if __name__ == '__main__':
        start_time = time.time()
        param_dev = 'alpha'
        energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs = get_fits_avg_values_stdevs(model, param_dev)
        
        # Setting params
        num = num
        files_num = files_num
        step = 4 * (stdevs[num] / files_num)
        for energy in [energies[num]]:
            
            # Setting Range
            sigma_D = sigmaD_avgs[num]
            alpha = alpha_avgs[num]
            k_0 = k0_fit[num]
            alpha_range = [alpha + i*step for i in range(int(-files_num/2), int(files_num/2 + 1))]
            
            
            # Creating Folder
            path_to_folder = f'Data_Collection/Angantyr_{model}/{param_dev}'
            folder = f'energy_{energy}'
            if os.path.exists(f'{path_to_folder}/{folder}') == False: 
                print(f'Angantyr model {model} chosen...')
                print(f'Making folder energy_{energy}...')
                if os.path.exists(f'{path_to_folder}') == False: 
                    os.system(f"mkdir {path_to_folder}")
                os.system(f"mkdir {path_to_folder}/energy_{energy}") 
            
            
            # Running program
            results = []
            pool = mp.Pool(mp.cpu_count())
            for alpha in alpha_range:
                sigma_D, alpha, k_0 = round_list([sigma_D, alpha, k_0], num_dec=2)
                filename = f'-{sigma_D}_-{alpha}_-{k_0}.txt'
                print(f'Creating file {filename} in {path_to_folder}/{folder}')
                if os.path.exists(f'{path_to_folder}/{folder}/{filename}') == True and overwrite == True: 
                    print(f'File {filename} exists. Skipping...')
                results.append(pool.apply(create_file_for_given_params_at_energy, args=[-sigma_D, -k_0, -alpha, energy, model, param_dev]))
            
            print(results)
            
            end_time = time.time()
            total_time = end_time - start_time
            print("Run time: ", total_time)
            
            
def varying_param_with_two_fixed_params(param_chosen, model, num=2, files_num = 20, step=3, overwrite=False):
    if __name__ == '__main__':
        start_time = time.time()
        if param_chosen == 'k_0':
            energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs = get_fits_avg_values_stdevs(model, param_chosen)
            fixed_param1 = alpha_avgs
            fixed_param2 = sigmaD_avgs
            param_avgs = k0_fit
            
        if param_chosen == 'alpha':
            energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs = get_fits_avg_values_stdevs(model, param_chosen)
            fixed_param1 = k0_fit
            fixed_param2 = sigmaD_avgs
            param_avgs = alpha_avgs

        if param_chosen == 'sigma_D':
            energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs = get_fits_avg_values_stdevs(model, param_chosen)
            fixed_param1 = alpha_avgs
            fixed_param2 = k0_fit   
            param_avgs = sigmaD_avgs
        
        # Setting params
        num = num
        files_num = files_num
        step = 4 * (stdevs[num] / files_num)
        for energy in [energies[num]]:
            
            # Setting Range
            fixed_1 = fixed_param1[num]
            fixed_2 = fixed_param2[num]
            param_value = param_avgs[num]
            _range = [param_value + i*step for i in range(int(-files_num/2), int(files_num/2 + 1))]
            
            
            # Creating Folder
            path_to_folder = f'Data_Collection/Angantyr_{model}/{param_chosen}'
            folder = f'energy_{energy}'
            if os.path.exists(f'{path_to_folder}/{folder}') == False: 
                print(f'Angantyr model {model} chosen...')
                print(f'Making folder energy_{energy}...')
                if os.path.exists(f'{path_to_folder}') == False: 
                    os.system(f"mkdir {path_to_folder}")
                os.system(f"mkdir {path_to_folder}/energy_{energy}") 
            
            
            # Running program
            results = []
            pool = mp.Pool(mp.cpu_count())
            for value in _range:
                if param_chosen == 'sigma_D':
                    sigma_D = value
                    alpha = fixed_1
                    k_0 = fixed_2
                    
                if param_chosen == 'alpha':
                    k_0 = fixed_1
                    alpha = value
                    sigma_D = fixed_2
                    
                if param_chosen == 'k_0':
                    k_0 = value
                    alpha = fixed_1
                    sigma_D = fixed_2
                    
                    
                sigma_D, alpha, k_0 = round_list([sigma_D, alpha, k_0], num_dec=2)
                filename = f'-{sigma_D}_-{alpha}_-{k_0}.txt'
                print(f'Creating file {filename} in {path_to_folder}/{folder}')
                if os.path.exists(f'{path_to_folder}/{folder}/{filename}') == True and overwrite == True: 
                    print(f'File {filename} exists. Skipping...')
                results.append(pool.apply(create_file_for_given_params_at_energy, args=[-sigma_D, -k_0, -alpha, energy, model, param_chosen]))
            
            print(results)
            
            end_time = time.time()
            total_time = end_time - start_time
            print("Run time: ", total_time)
################################## COLLECTING DATA ##################################################################

# startE = 10
# endE = 14
# filesNum = 2

# seed = 220
# angantyr_model = 1

# collect_data_for_random_seed(seed, startE, endE, filesNum, 
#                               angantyr_model, logscale=True, 
#                               overwrite=True)
        


# for num in range(10, 21):
#     model = 2
#     varying_sigma_with_fixed_k0_and_alpha(model, num=num, files_num = 20)
    


# param_dev = 'sigma_D'
# model = 2
# energies, k0_fit, sigmaD_avgs, alpha_avgs, stdevs = get_fits_avg_values_stdevs(model, param_dev)


# for energy, k_0 in zip(energies, k0_fit):
#     sigma_D = 6.86
#     alpha = 0.556
#     save_folder = 'fixed_params'
#     print(f'Trying k0:{k_0} at energy {energy}...' )
#     create_file_for_given_params_at_energy(-sigma_D, -k_0, -alpha, energy, model, save_folder)



# for energy in energies:
#     os.system(f'mkdir Data_Collection/Angantyr_{model}/fixed_params/energy_{energy}')






# for num in range(10, 20):
#     model = 2
#     varying_alpha_with_fixed_k0_and_sigma(model, num=num, files_num = 20, overwrite=True)
    



for num in range(0, 1):
    model = 1
    varying_param_with_two_fixed_params('k_0', model, num=num, files_num = 20, overwrite=True)












