# ============= IMPORT LIBRARIES =============
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

# Import custom settings
if os.getcwd().endswith('notebooks'):
    PROJECT_ROOT = os.path.dirname(os.getcwd())
    from SNeMPhyVAE.model.settings import initial_settings, band_info
else:
    PROJECT_ROOT = os.getcwd()
    from settings import initial_settings, band_info

# =============================================

class Spectra():

    def __init__(self, settings=initial_settings, snii_only=False):

        self.snii_only = snii_only
        self.initial_settings = settings

    def _load_data(self):
        """
        Load spectra data from pickle files and merge them. There are two main files:
        
        - object_table: Contains metadata about the objects.
        - spectra_alercexwiserep: Contains the spectral data.
        
        Returns:
        --------
        spectra: '~pd.DataFrame'
            Merged DataFrame containing spectra and their associated metadata,
            based on the 'oid' column.
        """
        
        object_table_path           = '../SNeMPhyVAE/data/object_ALeRCExWiserep20240630_20240703.pkl'
        spectra_alercexwiserep_path = '../SNeMPhyVAE/data/spectra_ALeRCE20240801_x_wisrep_20240622.pkl'

        object_table           = pd.read_pickle(filepath_or_buffer=object_table_path)
        spectra_alercexwiserep = pd.read_pickle(filepath_or_buffer=spectra_alercexwiserep_path)

        spectra = pd.merge(left=spectra_alercexwiserep, right=object_table, on='oid')

        if self.snii_only:
            snii_labels = [label for label in spectra.true_label.unique() if 'SN II' in label]
            spectra = spectra[spectra.true_label.isin(snii_labels)]

        return spectra

    def obtain_data(self):
        """
        Load and filter the spectra data to remove entries with all NaN flux values.
        
        Returns:
        --------
        data: '~pd.DataFrame'
            Filtered DataFrame containing only spectra with valid flux data.
        """
        data = self._load_data()
        mask = data.flux_lambda_smooth.apply(lambda x: np.all(np.isnan(x)))

        return data[~mask].copy().reset_index(drop=True)

    def _preprocess_flux(self, spectrum):
        """
        Generates the wavelength grid and extracts non-zero flux indices.
        Also performs de-redshifting to the rest-frame.

        Parameters:
        -----------
        spectrum: `~pd.Series`
            Spectral record that must contain 'lambda_grid_min', 'lambda_grid_max',
            and 'flux_lambda_smooth' keys.

        Returns:
        --------
        flux_nonzero: `~np.ndarray`
            Array containing flux values at non-zero indices.
        wave_nonzero: `~np.ndarray`
            Array containing wavelengths corresponding to non-zero indices.
        spectra_dict: dict
            Dictionary with intermediate information (complete grid, indices, etc.)
            for use in subsequent processing stages.
        """
        # Create wavelength grid and extract flux
        wave_range = np.logspace(
            np.log10(spectrum.lambda_grid_min),
            np.log10(spectrum.lambda_grid_max),
            1838
        )
        
        # 1. De-redshifting (To rest-frame)
        redshift   = np.nan_to_num(float(spectrum['redshift']))
        wave_range = wave_range / (1 + redshift)

        # Find the min and max indices of non-NaN values
        flux = spectrum['flux_lambda_smooth'].copy()
        mask = ~np.isnan(flux)
        #print("mask:", len(mask), np.sum(mask))
        #plt.plot(wave_range[mask], flux[mask])
        #plt.show()
        
        #nonnan_maks = ~np.isnan(flux)
        #if np.any(nonnan_maks):
        #    idx = np.where(nonnan_maks)[0]
        #    minIdx = max(0, idx[0]-1)
        #    maxIdx = min(len(flux)-1, idx[-1]+1)
        
        #flux = np.nan_to_num(flux, nan=0.0)
        spectra_dict = {
            'oid':             spectrum.oid,
            'wave_range':      wave_range,
            'flux':            flux,
            'flux_spline':     np.zeros_like(wave_range),
            'flux_continuum':  np.zeros_like(wave_range),
            'flux_apodized':   np.zeros_like(wave_range),
            'flux_normalized': np.zeros_like(wave_range),
            'final_spectrum':  np.zeros_like(wave_range),
            'mask':            mask,
        }

        return flux, wave_range, spectra_dict

    def normalize_spectrum(self, flux, spectra_dict):
        """
        Normalizes the spectrum using the minmax method.

        Parameters:
        -----------
        spectrum: '~pd.Series'
            Spectral record.

        Returns:
        --------
        flux_normalized: '~np.ndarray'
            Normalized flux.
        """
        mask = spectra_dict['mask']
        norm_flux = flux[mask]
        
        if norm_flux.size == 0:
            print(f"Warning: Empty flux array for spectrum")
            return np.zeros_like(flux)

        if np.all(norm_flux == 0):
            return flux

        spectra_dict['flux_normalized'][mask] = (norm_flux - np.min(norm_flux)) / (np.max(norm_flux) - np.min(norm_flux))
        return spectra_dict['flux_normalized']

    def continuum_fitting(self, flux, wave, spectra_dict, nknots=13) -> np.ndarray:
        """
        Obtain the continuum fitting of a given spectrum using spline interpolation.
        
        Parameters:
        -----------
        flux: '~np.ndarray'
            specum flux (normalized).
        wave: '~np.ndarray'
            wavelengths.
        spectra_dict: dict
            Dictionary with intermediate information obtained from _grid_flux.

        Returns:
        --------
        spectrum_flux '~np.ndarray'
            Spectrum with the continuum fitting applied (flux divided by the continuum).
        """

        mask = spectra_dict['mask']

        # This is added following the AstroDASH tutorial
        flux_working = flux.copy()
        flux_working[mask] = flux_working[mask]
        #continuum_flux = np.copy(flux)

        wave_spline = wave[mask]
        flux_spline = flux_working[mask]
        
        if len(wave_spline) < 13:
            print('There are not enought data. < 13')
            return flux
        
        indx = np.linspace(0, len(wave_spline)-1, nknots, dtype=int)
        
        wave_knots = wave_spline[indx]
        flux_knots = flux_spline[indx]
        
        # Fit a spline to the knots
        # If k=3 (cubic spline) is too oscillatory.
        spline = UnivariateSpline(wave_knots, flux_knots, k=3) 
        # This is based on AstroDASH tutorial remake the spline
        spline_wave  = np.linspace(wave_spline.min(), wave_spline.max(), nknots)
        spline_point = spline(spline_wave)
        spline       = UnivariateSpline(spline_wave, spline_point, k=3)
        spline_point = spline(wave_spline)
         
        
        spectra_dict['flux_spline'][mask] = spline_point
        # Save the continuum flux in the spectra_dict dictionary
        
        spectra_dict['flux_continuum'][mask] = flux[mask] / spline_point

        return spectra_dict['flux_continuum']

    def apodization(self, flux, spectra_dict, fraction=0.05):
        """
        Apply apodization to the spectrum using a 'cosine bell' window 
        at the start and end.

        Parameters
        ----------
        flux: '~np.ndarray'
            Flux vector.
        fraction: float
            Spectrum fraction to apply the window (default 5%).

        Returns
        -------
        flux_apodized: '~np.ndarray'
            Apodized flux vector.
        """
        mask      = spectra_dict['mask']
        apod_flux = flux[mask]
        
        if len(apod_flux) == 0: # There are not enough data
            return flux

        n_apod = max(1, int(len(apod_flux) * fraction))

        window = np.ones(len(apod_flux))
        x = np.linspace(0, np.pi/2, n_apod)
        window[:n_apod]  = np.sin(x)**2
        window[-n_apod:] = np.sin(x[::-1])**2

        spectra_dict['flux_apodized'][mask] = apod_flux * window
        return spectra_dict['flux_apodized']

    def preprocess_spectrum(self, spectra, slice_spectrum=False):
        """
        Process a spectra set and add columns with the processed information.

        Parameters:
        -----------
        spectra : '~pd.DataFrame'
            DataFrame containing spectra records or a single spectrum as a pd.Series.
        slice_spectrum : bool
            If True, the final spectrum will be sliced to a target size using smoothing and 
            interpolation.

        Returns:
        --------
        new_spectra : '~pd.DataFrame'
            DataFrame with the processed spectra, including columns like:
            'flux', 'wave', 'flux_continuum', 'flux_normalized', and 'final_spectrum'.
        """

        results = []

        if isinstance(spectra, pd.Series):
            spectra = pd.DataFrame([spectra])

        # Iterarate over each spectrum
        for _, spectrum in spectra.iterrows():

            flux, wave, spectra_dict = self._preprocess_flux(spectrum)
            mask = spectra_dict['mask']            
            # Add the minimum flux as a small constant to avoid 
            # negative flux values
            flux[mask] += np.abs(np.min(flux[mask]))
            
            continuum_result = self.continuum_fitting(flux, wave, spectra_dict)
            if continuum_result is None:
                continue

            # This is made by the AstroDASH tutorial
            continuum_result = continuum_result - 1
            
            norm_result = self.normalize_spectrum(continuum_result, spectra_dict)

            spec_flux = self.apodization(norm_result, spectra_dict)
            if len(spec_flux) == 0:
                continue

            #spec_flux = self.normalize_spectrum(spec_flux, spectra_dict)
            #if len(spec_flux) == 0:
            #    continue

            #np.clip(spec_flux, a_min=0.0, a_max=None, out=spec_flux)
            #mask = spec_flux > 0

            # Actualizar los arrays en el estado en las posiciones de índices no nulos
            #mask = spectra_dict['nonzero_mask']
            #spectra_dict['flux_normalized'][mask] = norm_flux
            #spectra_dict['continuum_flux'][mask]  = spectra_dict['flux_continuum_fit']

            #spectra_dict['final_spectrum'][mask]  = spec_flux
            spectra_dict['final_spectrum']  = spec_flux

            #print(type(spec_flux))
            #final_spectrum[idx]  = self.slice_spectrum(spec_flux, target_size=self.initial_settings['spectrum_bins'])

            # Crear un diccionario con los resultados; se usa la grilla completa
            results.append({
                'oid':             spectrum.oid,
                'mjd':             spectrum.mjd,
                'redshift':        spectrum.get('redshift', np.nan),
                'wave':            spectra_dict['wave_range'],
                'flux':            spectra_dict['flux'],
                'flux_spline':     spectra_dict['flux_spline'],
                'flux_continuum':  spectra_dict['flux_continuum'],
                'flux_apodized':   spectra_dict['flux_apodized'],
                'flux_normalized': spectra_dict['flux_normalized'],
                'final_spectrum':  spectra_dict['final_spectrum'],
                'mask':            spectra_dict['mask'],
                'lambda_grid_min': spectrum.lambda_grid_min,
                'lambda_grid_max': spectrum.lambda_grid_max,
                'nlambda_grid':    spectrum.nlambda_grid,
            })

        result_df = pd.DataFrame(results)

        if slice_spectrum:
            result_df['wave_sliced'] = result_df['wave'].apply(lambda x: self.slice_wavelength(x))
            result_df['flux_sliced'] = result_df.apply(
            lambda row: self.slice_spectrum(
            row['final_spectrum'], 
            row['wave'], 
            method='savgol', 
            window_size=9, 
            polyorder=2
        ), 
        axis=1
    )
        else:
            result_df['wave_sliced'] = result_df['wave']
            result_df['flux_sliced'] = result_df['final_spectrum']
        return result_df

    def spectra_reference_time(self, spectra, lightcurves):
        # Seleccionar solo las columnas necesarias de light_curves
        reference_times = lightcurves[['oid', 'reference_time']].drop_duplicates()

        # Realizar un merge en la columna 'oid'
        spectra = spectra.merge(reference_times, on='oid', how='left')
        spectra['time_index'] = np.round((spectra['reference_time'] - spectra['mjd']) * self.initial_settings['sideral_scale']).astype(int) + self.initial_settings['time_window'] // 2
        return spectra
    
    def slice_wavelength(self, wave, target_size=None):
        """
        Reduce the size of a wavelength array using interpolation.

        Parameters:
        ------------
        - wave: '~np.ndarray': 
            Input wavelength array.
        - target_size: int or None
            Desired output size. If None, no size reduction is applied.
        Returns:
        --------
            np.ndarray: Wavelength array reduced to target_size.
        """
        if target_size is None: 
            target_size = self.initial_settings['spectrum_bins']

        if target_size == len(wave):
            return wave

        wave_reduced = np.linspace(wave.min(), wave.max(), target_size)
        return wave_reduced

    def slice_spectrum(self, spectrum, wave, method='moving_average', window_size=9, polyorder=2, target_size=None):
        """
        Smooth and reduce the size of a spectrum using interpolation or averaging.

        Parameters:
        ------------
        - spectrum: '~np.ndarray': 
            Input spectrum.
        - method: str 
            Smoothing method ('savgol', 'gaussian', 'moving_average').
        - window_size: int 
            Size of the smoothing window (must be odd).
        - polyorder: int 
            Polynomial order for Savitzky-Golay filter (if used).
        - sigma: float 
            Standard deviation for Gaussian filter (if used).
        - target_size: int or None
            Desired output size. If None, no size reduction is applied.
        Returns:
        --------
            np.ndarray: Espectro suavizado y reducido.
        """
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import savgol_filter

        if target_size is None: 
            target_size = self.initial_settings['spectrum_bins']

        # Ensure window_size is odd
        window_size = max(3, min(window_size, len(spectrum)-1))
        if window_size % 2 == 0:
            window_size += 1 # Make it odd

        # Apply smoothing
        if method == 'savgol':
            if polyorder >= window_size:
                polyorder = max(1, window_size - 1)
            smoothed = savgol_filter(spectrum, window_length=window_size, polyorder=polyorder, mode='interp')

        elif method == 'moving_average':
            kernel   = np.ones(window_size) / window_size
            smoothed = np.convolve(spectrum, kernel, mode='same')

        # Size reduction if target_size is specified
        if target_size is not None and target_size != len(smoothed):
            wave_target = np.linspace(wave.min(), wave.max(), target_size)
            smoothed = np.interp(wave_target, wave, smoothed)

        return smoothed

if __name__ == "__main__":

    spectra_processor = Spectra(snii_only=True)
    spectra = spectra_processor.obtain_data()
    print(f"Total spectra: {len(spectra)}")

    preprocessed_spectra = spectra_processor.preprocess_spectrum(spectra, slice_spectrum=True)
    #print(preprocessed_spectra.head())
    
    oidx_list = preprocessed_spectra.oid.unique()
    oidx = np.random.choice(oidx_list)
    
    spec = spectra[spectra.oid == oidx].iloc[0]
    prespec = preprocessed_spectra[preprocessed_spectra.oid == oidx].iloc[0]
    #print(prespec)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                             sharex=True, 
                             #gridspec_kw={'hspace': 0.05}
                             )
    axes[0].plot(prespec.wave, spec.flux_lambda , label='Original Flux', color='orange', alpha=0.5)
    axes[1].plot(prespec.wave, prespec.flux, label='Processed Flux', color='red', alpha=0.5)
    axes[1].plot(prespec.wave, prespec.flux_spline, label='Continuum Flux', color='green', alpha=0.5)
    axes[2].plot(prespec.wave_sliced, prespec.flux_sliced, label='PreProcess Flux', color='blue', alpha=0.5)
    
    for ax in axes:
        ax.set_ylabel('Flux')
        ax.legend(frameon=False)
        
    fig.suptitle(f'Spectrum OID: {oidx}', fontsize=16)
    
    plt.show()