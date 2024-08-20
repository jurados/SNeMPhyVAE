import os

import numpy as np
import pandas as pd
#import polars as pl
import matplotlib.pyplot as plt

import astropy

import scipy

def crossmatch_object_alerce(alerce_lc: pd.DataFrame, object: pd.DataFrame) -> pd.DataFrame:
    lightcurves = pd.merge(left=alerce_lc, right=object,
                       on='oid')
    return lightcurves

def process_light_curve_parsnip(ligth_curve):

    new_light_curve = ligth_curve.copy()

    SIDEREAL_SCALE = 86400. / 86164.0905

    time = ligth_curve['mjd'].to_numpy()
    sidereal_time = time * SIDEREAL_SCALE

    # Initial guess of the phase. Round everything to 0.1 days, and find the decimal
    # that has the largest count.
    mode, count = scipy.stats.mode(np.round(sidereal_time % 1 + 0.05, 1), keepdims=True)
    guess_offset = mode[0] - 0.05

    # Shift everything by the guessed offset
    guess_shift_time = sidereal_time - guess_offset

    # Do a proper estimate of the offset.
    sidereal_offset = guess_offset + np.median((guess_shift_time + 0.5) % 1) - 0.5

    # Shift everything by the final offset estimate.
    shift_time = sidereal_time - sidereal_offset

    # Selecting the 
    s2n = ligth_curve['magpsf'] / ligth_curve['sigmapsf']
    s2n_mask = np.argsort(s2n)[-5:]

    cut_times = shift_time[s2n_mask]

    max_time = np.round(np.median(cut_times))

    # Convert back to a reference time in the original units. This reference time
    # corresponds to the reference of the grid in sidereal time.
    reference_time = ((max_time + sidereal_offset) / SIDEREAL_SCALE)
    grid_times = (time - reference_time) * SIDEREAL_SCALE
    time_indices = np.round(grid_times).astype(int) + 300 // 2 # 300 days
    time_mask = (
        (time_indices >= -100)
        & (time_indices < 300 + 100)
    )
    new_light_curve['grid_time'] = grid_times
    new_light_curve['time_index'] = time_indices
    new_light_curve = new_light_curve[time_mask]

    return new_light_curve  

def create_grid(lightcurve: pd.DataFrame):

    # Build a grid for the input
    grid_flux = np.zeros((1,
                            300))
    grid_weights = np.zeros_like(grid_flux)

    mask = (lightcurve['time_index'] >= 0) & (lightcurve['time_index'] < 300)
    lightcurve = lightcurve[mask]
   
   # Fill in the input array.
    grid_flux[0, lightcurve['time_index']] = lightcurve['magpsf']

    return grid_flux