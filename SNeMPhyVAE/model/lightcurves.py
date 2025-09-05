# ============= IMPORT LIBRARIES =============
import numpy as np
import pandas as pd

import scipy

# Import custom settings
from TMPhy_VAE.settings import initial_settings, band_info
# =============================================

class LightCurves():

    def __init__(self, instrument:str = None, snii_only:bool =False):

        self.lightcurves = None
        self.instrument  = instrument
        self.snii_only   = snii_only

    def _load_data(self):
        """

        Params
        -------
        - snii_only: bool,
            Obtain only type II SN: [SNII, SNIIn, SNIIb]

        Returns
        -------
        - ligthcurves: '~pd.DataFrame',
            DataFrame with supernovae's lightcurves after
            perfom the cross-match between the object table and
            alerce table through merge function on the 'oids'.
        """
        object_table_path           = './data/object_ZTF_ALeRCE_19052024.pkl'
        lightcurves_alercextns_path = './data/lcs_transients_20240517.pkl'

        object_table           = pd.read_pickle(filepath_or_buffer=object_table_path)
        lightcurves_alercextns = pd.read_pickle(filepath_or_buffer=lightcurves_alercextns_path)

        self.lightcurves = pd.merge(left=lightcurves_alercextns, right=object_table, on='oid')

        if self.snii_only:
            self.lightcurves = self.lightcurves[self.lightcurves['true_label'].isin(['SNIIn','SNIIb','SNII'])]

        return self.lightcurves

    def _clean(self, lightcurves, min_points=10, sigma_flux = 1e-27, max_time_gap=80):
        """Clean light curves by removing filters with
        insufficient observations, low flux dispersion, and large time gaps.

        Parameters
        ----------
        min_points : int, optional
            Minimum number of observations required per filter (default: 10)
        sigma_flux : float, optional
            Minimum standard deviation of flux per filter (default: 1e-27)
        max_time_gap : float, optional
            Maximum allowed time gap (in days) between consecutive observations (default: 30)

        Returns
        -------
        '~pd.DataFrame'
            Cleaned light curves DataFrame
        """
        if lightcurves is None:
            raise ValueError('First you need load the data with _load_data()')

        # Keep only filters with enough observations
        mask_count = (
            lightcurves.groupby(['oid', 'fid'])['oid'].transform('size') >= min_points
        )
        lightcurves = lightcurves[mask_count]

        cleaned_lcs = []
        for (oid, fid), group in lightcurves.groupby(['oid', 'fid']):
            group = group.sort_values(by='mjd')
            time_diffs = group['mjd'].diff()

            # Keep points where time gap is small
            mask_time = (time_diffs > max_time_gap).cumsum()
            groups = group.groupby(mask_time)
            best_group = max(groups, key=lambda g: len(g[1]))[1]

            if len(best_group) >= min_points:
                cleaned_lcs.append(best_group)

        lightcurves = pd.concat(cleaned_lcs, ignore_index=True)

        # Keep only filters with significant flux variation
        mask_sigma = (
            lightcurves.groupby(['oid', 'fid'])['flux'].transform('std') > sigma_flux
        )
        lightcurves = lightcurves[mask_sigma]

        return lightcurves.reset_index(drop=True)

    def obtain_data(self):
        """
        Carga y limpia los datos.

        Returns
        -------
            '~pd.DataFrame': DataFrame final procesado.
        """
        self.lightcurves = self._load_data()
        #self.lightcurves = self._clean(self.lightcurves)
        return self.lightcurves

    def _determine_time_grid(self, lightcurve: pd.DataFrame):

        time = lightcurve['mjd'].to_numpy(dtype=np.float32)
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
        s2n = (lightcurve['flux'] / lightcurve['fluxerr']).to_numpy(dtype=np.float32)
        #print(s2n)
        s2n_mask = np.argsort(s2n)[-5:]

        s2n_mask_2 = s2n[s2n_mask] > 5.
        if np.any(s2n_mask_2):
            cut_times = shift_time[s2n_mask][s2n_mask_2]
        else:
        # No observations with signal-to-noise above 5. Just use whatever we
        # have...
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

        # Build a preprocessed light curve object.
        new_lightcurve = lightcurve.copy()

        # Map each band to its corresponding index.
        #band_map = {j: i for i, j in enumerate(initial_settings['bands'])}
        #new_lightcurve['band_index'] = [band_map.get(i, -1) for i in new_lightcurve['fid']]

        #
        grid_times = self.time_to_grid(new_lightcurve['mjd'], reference_time).to_numpy(dtype=np.float32)
        time_indices = np.round(grid_times).astype(int) + initial_settings['time_window'] // 2 # 300 days
        time_mask = (
            (time_indices >= -initial_settings['time_pad'])
            & (time_indices < initial_settings['time_window'] + initial_settings['time_pad'])
        )
        new_lightcurve['grid_time']      = grid_times
        new_lightcurve['time_index']     = time_indices
        new_lightcurve['reference_time'] = reference_time

        # Cut out observations that are in unused bands or outside of the time window.
        #band_mask = new_lightcurve['band_index'] != -1
        #new_lightcurve = new_lightcurve[band_mask & time_mask]
        new_lightcurve = new_lightcurve[time_mask]

        s2n = new_lightcurve['flux'] / new_lightcurve['fluxerr']
        s2n_mask = s2n > 5.
        if np.any(s2n_mask):
            #scale = np.max(new_lightcurve['flux'][s2n_mask])
            new_lightcurve['flux_scale'] = np.max(new_lightcurve['flux'][s2n_mask])
        else:
            #scale = np.max(new_lightcurve['flux'])
            new_lightcurve['flux_scale'] = np.max(new_lightcurve['flux'])

        #new_lightcurve = new_lightcurve[time_mask].reset_index(drop=True)
        #new_lightcurve['flux_scale'] = max(new_lightcurve['flux'])

        return new_lightcurve

    def new_light_curves(self, lightcurves):
        """ Create a new data set adding the grid_time and time_index columns

        Parameters
        ----------
        light_curves : `~pd.DataFrame`
            Raw light curve

        Returns
        -------
        `~pd.DataFrame`
            Preprocessed light curve
        """

        new_lightcurves = [self.process_light_curve(
            group.sort_values(by='mjd')
            .drop_duplicates(subset=['fid', 'mjd'])
            .groupby('fid', group_keys=False)
            .apply(lambda x: x.reset_index(drop=True))
        ) for _, group in lightcurves.groupby('oid')]

        new_lightcurves = pd.concat(new_lightcurves, ignore_index=True)

        return new_lightcurves

    def _get_bands(self, lightcurves):

        if getattr(self, 'instrument', None) == 'ztf':
            bands_fid = lightcurves.fid.unique()
            bands = [key_band for key_band, value in band_info.items() if value[-1] in bands_fid]
            initial_settings.update({'bands': bands})
        else:
            initial_settings.update({'bands': {'bands': ['ztfg', 'ztfr']}})

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

            # Usar los valores de A_V para corregir las magnitudes
            corrected_mag = group['magpsf'] + group['Av_MW']

            # Calculate flux and fluxerr
            flux = 10**(-0.4 * (corrected_mag + 48.60))
            fluxerr = abs(flux * -0.4 * np.log(10) * group['sigmapsf'])

            # Assign calculated values directly to the DataFrame
            lightcurves.loc[group.index, 'flux'] = flux
            lightcurves.loc[group.index, 'fluxerr'] = fluxerr

        return lightcurves.drop(columns=['magpsf', 'sigmapsf'])

    def process_lightcurves(self, lightcurves):

        #lightcurves = lightcurves.copy()
        self._get_bands(lightcurves)
        lightcurves = self.mag2flux(lightcurves)
        lightcurves = self.new_light_curves(lightcurves)
        new_lightcurves = self._clean(lightcurves)
        #new_lightcurves = new_lightcurves.drop_duplicates(subset=['oid', 'fid', 'mjd'])

        return new_lightcurves