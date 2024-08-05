#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:02:24 2024

@author: jurados
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

PATH_INPUT = './data/spectra_wiserep/'
all_files = [file for file in os.listdir(path=PATH_INPUT) if '_raw' not in file]
print('The total amount of Supernovae ready to process is:',len(all_files))

wavelength_lsst = {
    'u': [3206.34, 4081.51],
    'g': [3876.02, 5665.33],
    'r': [5377.19, 7055.16],
    'i': [6765.77, 8325.05],
    'z': [8035.39, 9375.47],
    'y': [9089.07, 10915.01]
}

# Obtain the min and max values of the LSST's wavelength
wavelength_grid_min = min([wavelength for wavelength_list in wavelength_lsst.values() for wavelength in wavelength_list])
wavelength_grid_max = max([wavelength for wavelength_list in wavelength_lsst.values() for wavelength in wavelength_list])

print(f'The min wavelength value to create the grid is: {wavelength_grid_min:.2f} Angstrom')
print(f'The max wavelength value to create the grid is: {wavelength_grid_max:.2f} Angstrom')

# Number grid's bins
nwavelength_grid = 1838

# Array equal spacing of wavelengths 
wavelength_grid_array = np.logspace(np.log10(wavelength_grid_min),np.log10(wavelength_grid_max),nwavelength_grid)

def smooth_flux(data:pd.DataFrame, dv:float = 200, dvsmooth:float = 2000) -> pd.DataFrame:
    """
    
    Params
    ------
    data: pd.DataFrame with the spectra data
    dv: velocity 

    Return
    ------
    pd.Dataframe mainly flux rolled of the log_lambda
    """

    CSPEED = 3e5 # light_speed in km/s

    dlog10lambda = dv / CSPEED / np.log(10) * (24 * 3600) # pseudo seconds
    dlog10lambdasmooth = dvsmooth / CSPEED / np.log(10) * 24 * 3600 # pseudo seconds

    #data["log10lambda"] = np.log10(data["lambda"])
    #data["log10lambdasmooth"] = np.log10(data["lambda"])

    # fool pandas to make it think log10lambda is days
    #data["log10lambda_idx"] = data["log10lambda"].apply(lambda x: pd.Timedelta(x, 'days'))

    result = []
    for name, group in data.groupby(['mjd', 'instrument']):
        group = group.sort_values(by='lambda', ascending=True)
        group["log10lambda"] = np.log10(group["lambda"])
        group["log10lambda_idx"] = group["log10lambda"].apply(lambda x: pd.Timedelta(x, 'days'))
        group = group.set_index('log10lambda_idx')
        group["flux_log10lambda_rolling"] = group.flux_lambda.rolling(f'{int(dlog10lambda)}s', center=True).mean()
        group["flux_log10lambda_rolling_smooth"] = group.flux_lambda.rolling(f'{int(dlog10lambdasmooth)}s', center=True).mean()
        group["eflux_log10lambda_rolling"] = (group.flux_log10lambda_rolling - group.flux_lambda.rolling(f'{int(dlog10lambdasmooth)}s', center=True).mean()).rolling(f'{int(dlog10lambdasmooth)}s', center=True).std()
        result.append(group)
    data = pd.concat(result)
    data.reset_index(inplace=True)
    data = data.drop('log10lambda_idx',axis=1)
    data
    
    return data

def obtain_interpolated_flux(x, y, lambda_grid:np.array) -> np.array:

    f = interpolate.interp1d(x, y, fill_value=np.nan, bounds_error=False)

    # computing the new flux in the lambda_grid
    flux_new = f(lambda_grid)

    return flux_new

# Lista para almacenar los resultados
def arrange_spectra(sn_name:str, data:pd.DataFrame, oid:int,
                    lambda_grid:np.array, nlambda_grid: int) -> pd.DataFrame:
    
    #CSPEED = 3e5 # light_speed in km/s
    results = []
    #dlog10lambdasmooth = 2000 / CSPEED / np.log(10) * 24 * 3600
    for (mjd, inst_name), group in data.groupby(['mjd', 'instrument']):
        flux_lambda = obtain_interpolated_flux(x=group['lambda'], y=group['flux_log10lambda_rolling'], lambda_grid=lambda_grid)
        flux_lambda_smooth = obtain_interpolated_flux(x=group['lambda'], y=group['flux_log10lambda_rolling_smooth'], lambda_grid=lambda_grid)
        #eflux_lambda = flux_lambda-flux_lambda_smooth

        data_flux = {
            'oid': oid,
            'snname':sn_name,
            'instrument': inst_name,
            'mjd': mjd,
            'lambda_grid_min': lambda_grid.min(),
            'lambda_grid_max': lambda_grid.max(),  
            'nlambda_grid': nlambda_grid,
            'lambda_data_min': group['lambda'].min(),
            'lambda_data_max': group['lambda'].max(),
            #'flux_lambda': mjd_group.flux_log10lambda_rolling.tolist(),
            'flux_lambda': flux_lambda,
            'flux_lambda_smooth': flux_lambda_smooth,
            'e_flux_lambda': group.eflux_log10lambda_rolling.tolist(),
            #'e_flux_lambda': eflux_lambda,
            }
        results.append(data_flux)

        oid += 1

    unique_table = pd.DataFrame(results)
    return unique_table, oid

if __name__ == '__main__':
    master_dataframe = pd.DataFrame()
    indx_ini = 14000
    indx_fin = 18000
    oid = indx_ini
    for file in all_files[indx_ini:indx_fin]:
        try:
            sn_name = file.split('.')[0]
            data = pd.read_csv(PATH_INPUT+file)
            data = smooth_flux(data=data)
            result_table, oid = arrange_spectra(sn_name,data,oid, wavelength_grid_array, nwavelength_grid)
            master_dataframe = pd.concat([master_dataframe, result_table])
            oid = oid
        except:
            continue
        
    master_dataframe.to_pickle(f'./master_spectra_table_{indx_ini}_{indx_fin}.pkl')

#fig, ax = plt.subplots()
#for name, group in result_table.groupby(['mjd', 'instrument']):
#    print(group.columns)
#    print(group.flux_lambda)
#    fig, ax = plt.subplots()
    #x = np.logspace(np.log10(group.lambda_grid_min), np.log10(group.lambda_data_max), 
                    #group.nlambda_grid.values[0])
#    ax.plot(x, np.array(group.flux_lambda)[0],label=f'{name[0]:.2f}MJD, instrument {name[1]}')
#    ax.plot(x, np.array(group.flux_lambda_smooth)[0],label=f'{name[0]:.2f}MJD, instrument {name[1]}')
#    ax.legend()
