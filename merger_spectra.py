import os
import re
import requests
import zipfile
import json
import datetime
import shutil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import interpolate

from astropy.time import Time

PATH_INPUT = './data/wiserep_spectra/'

all_files = [file for file in os.listdir(path=PATH_INPUT) if '_raw' not in file]

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

def smooth_flux(data: pd.DataFrame, dv: float = 200, dvsmooth: float = 2000) -> pd.DataFrame:
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

    data["log10lambda"] = np.log10(data["lambda"])

    # fool pandas to make it think log10lambda is days
    data["log10lambda_idx"] = data["log10lambda"].apply(lambda x: pd.Timedelta(x, 'days'))
    data.set_index("log10lambda_idx", inplace=True)

    # sorting the index
    data = data.sort_index()
    
    # obtain the flux rolled of the log_lambda
    data["flux_log10lambda_rolling"] = data.flux_lambda.rolling(f'{int(dlog10lambda)}s', center=True).mean()
    #delta = (data.flux_log10lambda_rolling - data.flux_lambda.rolling(f'{int(dlog10lambdasmooth)}s', center=True).mean()).rolling(f'{int(dlog10lambdasmooth)}s', center=True).std()
    data["eflux_log10lambda_rolling"] = (data.flux_log10lambda_rolling - data.flux_lambda.rolling(f'{int(dlog10lambdasmooth)}s', center=True).mean()).rolling(f'{int(dlog10lambdasmooth)}s', center=True).std()
    
    # erase the indexing
    data.reset_index(inplace=True)
    data = data.drop('log10lambda_idx',axis=1)
    
    return data

# Lista para almacenar los resultados
def arrange_spectra(sn_name:str,data: pd.DataFrame,oid:int,
                    lambda_grid:np.array, nlambda_grid: int) -> pd.DataFrame:
    results = []
    for inst_name, inst_group in data.groupby('instrument'):
        for mjd, mjd_group in inst_group.groupby('mjd'):
            #flux_lambda = obtain_interpolated_flux(data=data, lambda_grid=lambda_grid)
            
            #flux_lambda, eflux_lambda

            data_flux = {
                'oid': oid,
                'snname':sn_name,
                'instrument': inst_name,
                'mjd': mjd,
                'lambda_grid_min': lambda_grid.min(),
                'lambda_grid_max': lambda_grid.max(),
                'nlambda_grid': nlambda_grid,
                'lambda_data_min': data['lambda'].min(),
                'lambda_data_max': data['lambda'].max(),
                'flux_lambda': data.flux_log10lambda_rolling.tolist(),
                'e_flux_lambda': data.eflux_log10lambda_rolling.tolist()}
            results.append(data_flux)

            oid += 1

    unique_table = pd.DataFrame(results)
    return unique_table, oid

master_dataframe = pd.DataFrame()
indx_ini = 4000
indx_fin = 6000
oid = indx_fin - indx_ini
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

master_dataframe.to_csv(f'./master_spectra_table_{indx_ini}_{indx_fin}.csv')

print('End ...')