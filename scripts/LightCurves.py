import numpy as np
import pandas as pd
import scipy

from settings import *

class LightCurves():
    
    def __init__(self, instrument:str = None, snii_only:bool =False):
        
        self.lightcurves = None
        self.instrument  = instrument
        self.snii_only   = snii_only 
    
    def _load_data(self) -> pd.DataFrame:
        """
        
        Params
        -------
        - snii_only: bool,
            Obtain only type II SN: [SNII, SNIIn, SNIIb]
            
        Returns
        -------
        - ligthcurves: pd.DataFrame,
            DataFrame with supernovae's lightcurves after
            perfom the cross-match between the object table and 
            alerce table through merge function on the 'oids'.
        """
        object_table_path           = './Data/object_ZTF_ALeRCE_19052024.pkl'
        lightcurves_alercextns_path = './Data/lcs_transients_20240517.pkl' 
        
        object_table           = pd.read_pickle(filepath_or_buffer=object_table_path)
        lightcurves_alercextns = pd.read_pickle(filepath_or_buffer=lightcurves_alercextns_path)
        
        self.lightcurves = pd.merge(left=lightcurves_alercextns, right=object_table, on='oid')
        
        if self.snii_only:
            self.lightcurves = self.lightcurves[self.lightcurves['true_label'].isin(['SNIIn','SNIIb','SNII'])]
        
        return self.lightcurves
        
    def _clean(self) -> pd.DataFrame:
        """Clean data
        
        """
        if self.lightcurves is None:
            raise ValueError('First you need load the data with _load_data()')
        
        sn_list_erased = []
        for sn_oid, sn_data in self.lightcurves.groupby(by='oid'):
            
            fid_count = sn_data.fid.value_counts() 
            if len(fid_count) == 1 and fid_count.iloc[0] < 10:
                sn_list_erased.append(sn_oid)
            if len(fid_count) == 2 and ((fid_count.iloc[0] < 10) or (fid_count.iloc[1] < 10)):
                sn_list_erased.append(sn_oid)
            
        self.lightcurves = self.lightcurves[~self.lightcurves.oid.isin(sn_list_erased)]
        
        return self.lightcurves
        
    def obtain_data(self) -> pd.DataFrame:
        """
        Carga y limpia los datos.
        
        Retorna:
            pd.DataFrame: DataFrame final procesado.
        """
        self._load_data()
        return self._clean()
        
    def _determine_time_grid(self, lightcurve: pd.DataFrame):

        time = lightcurve['mjd'].to_numpy()
        sidereal_time = time * initial_settings['sideral_scale']

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

        # Selecting the five highest signal-to-noise observations
        s2n = lightcurve['flux'] / lightcurve['fluxerr']
        s2n_mask = np.argsort(s2n)[-5:]

        cut_times = shift_time[s2n_mask]

        max_time = np.round(np.median(cut_times))

        # Convert back to a reference time in the original units. This reference time
        # corresponds to the reference of the grid in sidereal time.
        reference_time = ((max_time + sidereal_offset) / initial_settings['sideral_scale'])

        return reference_time

    def time_to_grid(self, time: np.ndarray, reference_time: float) -> np.ndarray:
        """Convert a time in the original units to one on the internal ParSNIP grid

        Parameters
        ----------
        time : float
            Real time to convert
        reference_time : float
            Reference time for the grid

        Returns
        -------
        float
            Time on the internal grid
        """
        return (time - reference_time) * initial_settings['sideral_scale']

    def grid_to_time(self, grid_time: np.ndarray, reference_time: float):
        """Convert a time on the internal grid to a time in the original units

        Parameters
        ----------
        grid_time : float
            Time on the internal grid
        reference_time : float
            Reference time for the grid

        Returns
        -------
        float
            Time in original units
        """
        return grid_time / initial_settings['sideral_scale'] + reference_time

    def _get_reference_time(self, lightcurve):

        reference_time = self._determine_time_grid(lightcurve)

        return reference_time

    def process_light_curve(self, lightcurve):
        """Preprocess a light curve for the ParSNIP model

        Parameters
        ----------
        light_curve: `~pd.DataFrame`
            Raw light curve
        settings: dict
            ParSNIP model settings
        raise_on_invalid: bool
            Whether to raise a ValueError for invalid light curves. If False, None is
            returned instead. By default, True.
        ignore_missing_redshift : bool
            Whether to ignore missing redshifts, by default False. If False, a missing
            redshift value will cause a light curve to be invalid.

        Returns
        -------
        `~pd.DataFrame`
            Preprocessed light curve

        Raises
        ------
        ValueError
            For any invalid light curves that cannot be handled by ParSNIP if
            raise_on_invalid is True. The error message will describe why the light curve is
            invalid.
        """
        reference_time = self._determine_time_grid(lightcurve)

        new_lightcurve = lightcurve.copy()

        grid_times = self.time_to_grid(new_lightcurve['mjd'], reference_time)
        time_indices = np.round(grid_times).astype(int) + initial_settings['time_window'] // 2 # 300 days
        time_mask = (
            (time_indices >= -initial_settings['time_pad'])
            & (time_indices < initial_settings['time_window'] + initial_settings['time_pad'])
        )
        new_lightcurve['grid_time']      = grid_times
        new_lightcurve['time_index']     = time_indices
        new_lightcurve['reference_time'] = reference_time

        new_lightcurve = new_lightcurve[time_mask]
        new_lightcurve['flux_scale'] = max(new_lightcurve['flux'])

        return new_lightcurve

    def new_light_curves(self, lightcurves):
        """ Create a new data set adding the grid_time and
        time_index columns
        """

        lightcurve_processed_list = []
        for id_group, group in lightcurves.groupby('oid'):
            lightcurve_processed = self.process_light_curve(group)
            lightcurve_processed_list.append(lightcurve_processed)
        
        new_lightcurves = pd.concat(lightcurve_processed_list, ignore_index=True)
        return new_lightcurves

    def _get_bands(self, lightcurves):

        if getattr(self, 'instrument', None) == 'ztf':
            bands_fid = lightcurves.fid.unique()
            bands = [key_band for key_band, value in band_info.items() if value[-1] in bands_fid]
            initial_settings.update({'bands': bands})
        else:
            initial_settings.update({'bands': None})

    def mag2flux(self,lightcurves):
        """Convert the magnitud to flux using a system by default use AB_system

        Parameters
        ----------
        light_curves : `~pd.DataFrame`
            Raw light curve
        settings : dict
            ParSNIP model settings

        Returns
        -------
        `~pd.DataFrame`
            magnitud to flux

        """

        for sn_oid, group in lightcurves.groupby(by='oid'):
            lightcurves.loc[group.index, 'flux']    = 10**(-0.4 * (group['magpsf']+48.60))
            lightcurves.loc[group.index, 'fluxerr'] = -0.4*np.log(10)*10**(-0.4 * (group['magpsf']+48.60)) * group['sigmapsf']

        return lightcurves.drop(columns=['magpsf', 'sigmapsf'])
        
    def forward(self, lightcurves):

        lightcurves = lightcurves.copy()
        self._get_bands(lightcurves)
        lightcurves = self.mag2flux(lightcurves)
        #print(light_curves)
        new_lightcurves = self.new_light_curves(lightcurves)

        return new_lightcurves