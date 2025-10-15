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
            1838#,self.initial_settings['spectrum_bins']
        )
        
        # 1. De-redshifting (To rest-frame)
        redshift = np.nan_to_num(float(spectrum['redshift']))
        wave_range /= (1 + redshift)

        # Reeplace negative flux and NaN values by 0
        flux = np.nan_to_num(spectrum['flux_lambda_smooth'], nan=0.0)

        spectra_dict = {
            'oid':             spectrum.oid,
            'wave_range':      wave_range,
            'flux':            flux,
            'continuum_flux':  np.zeros_like(flux),
            'continuum_norm':  np.zeros_like(flux),
            'flux_apodized':   np.zeros_like(flux),
            'flux_normalized': np.zeros_like(flux),
            'final_spectrum':  np.zeros_like(flux)
        }

        #return flux[nonzero_mask], wave_range[nonzero_mask], spectra_dict
        return flux, wave_range, spectra_dict

    def normalize_spectrum(self, flux, spectra_dict):
        """
        Normalizes the spectrum by dividing the flux by its maximum value.

        Parameters:
        -----------
        spectrum: '~pd.Series'
            Spectral record.

        Returns:
        --------
        flux_normalized: '~np.ndarray'
            Normalized flux.
        """
        if flux.size == 0:
            print(f"Warning: Empty flux array for spectrum")
            return np.zeros_like(flux)

        flux = np.nan_to_num(flux, nan=0.0)
        flux = np.clip(flux, 0, None)

        if np.all(flux == 0):
            return flux

        max_flux = np.max(flux)
        spectra_dict['flux_normalized'][spectra_dict['mask']] = (flux - np.min(flux))/ (np.max(flux) - np.min(flux))
        return (flux - np.min(flux))/ (np.max(flux) - np.min(flux))#, wave

    def continuum_fitting(self, flux: np.ndarray, wave: np.ndarray, spectra_dict: dict):
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

        if flux.size < 13:
            print('There are not enought data. < 13')
            return flux

        wave = wave[spectra_dict['mask']]
        flux = flux[spectra_dict['mask']]

        indx = np.linspace(0, len(wave)-1, 13,  dtype=int)
        indx = np.unique(indx)
        wave_knots = wave[indx]
        flux_knots = flux[indx]

        sort_idx = np.argsort(wave_knots)
        wave_knots = wave_knots[sort_idx]
        flux_knots = flux_knots[sort_idx]

        # Although, the knots should be unique and sorted, we ensure it here.
        _, unique_idx = np.unique(wave_knots, return_index=True)
        wave_knots = wave_knots[unique_idx]
        flux_knots = flux_knots[unique_idx]

        spline = UnivariateSpline(wave_knots, flux_knots, k=3)  # k=3 es spline cúbico, s=0 ajuste exacto
        # Evalaute the spline at the knots to check
        continuum = spline(wave)

        # Save the continuum flux in the spectra_dict dictionary
        spectra_dict['continuum_flux'][spectra_dict['mask']] = flux / continuum

        return flux / continuum

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
        if len(flux) == 0: # There are not enough data
            return flux

        n_apod = max(1, int(len(flux) * fraction))

        window = np.ones(len(flux))
        x = np.linspace(0, np.pi/2, n_apod)
        window[:n_apod]  = np.sin(x)**2
        window[-n_apod:] = np.sin(x[::-1])**2

        spectra_dict['flux_apodized'][spectra_dict['mask']] = flux * window
        return flux * window

    def process_spectrum(self, spectra, slice_spectrum=False):
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
            flux += np.abs(np.min(flux))

            # Index mask for valid flux values
            mask = (flux > 0) & np.isfinite(flux)
            spectra_dict['mask'] = mask

            spec_flux = self.continuum_fitting(flux, wave, spectra_dict)
            if len(spec_flux) == 0:
                continue

            spec_flux = spec_flux - 1

            spec_flux = self.apodization(spec_flux, spectra_dict)
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
            spectra_dict['final_spectrum'][spectra_dict['mask']]  = spec_flux

            #print(type(spec_flux))
            #final_spectrum[idx]  = self.slice_spectrum(spec_flux, target_size=self.initial_settings['spectrum_bins'])

            # Crear un diccionario con los resultados; se usa la grilla completa
            results.append({
                'oid':             spectrum.oid,
                'mjd':             spectrum.mjd,
                'redshift':        spectrum.get('redshift', np.nan),
                'flux':            spectra_dict['flux'],
                'wave':            spectra_dict['wave_range'],
                'flux_continuum':  spectra_dict['continuum_flux'],
                'flux_apodized':   spectra_dict['flux_apodized'],
                'flux_normalized': spectra_dict['flux_normalized'],
                'final_spectrum':  spectra_dict['final_spectrum'],
                'lambda_grid_min': spectrum.lambda_grid_min,
                'lambda_grid_max': spectrum.lambda_grid_max,
                'nlambda_grid':    spectrum.nlambda_grid,
            })

        result_df = pd.DataFrame(results)

        if slice_spectrum:
            result_df['final_spectrum'] = result_df['final_spectrum'].apply(lambda x: self.slice_spectrum(x))
        return result_df

    def spectra_reference_time(self, spectra, lightcurves):
        # Seleccionar solo las columnas necesarias de light_curves
        reference_times = lightcurves[['oid', 'reference_time']].drop_duplicates()

        # Realizar un merge en la columna 'oid'
        spectra = spectra.merge(reference_times, on='oid', how='left')
        spectra['time_index'] = np.round((spectra['reference_time'] - spectra['mjd']) * self.initial_settings['sideral_scale']).astype(int) + self.initial_settings['time_window'] // 2
        return spectra

    def slice_spectrum(self, spectrum, method='moving_average', window_size=9, polyorder=2, sigma=1.0, target_size=None):
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

        if target_size is None: target_size = self.initial_settings['spectrum_bins']

        #spectrum = np.asarray(spectrum, dtype=np.float32)
        if len(spectrum) == 0:
            return np.zeros(target_size if target_size else len(spectrum))

        # Ensure window_size is odd
        window_size = max(3, min(window_size, len(spectrum)-1))
        if window_size % 2 == 0:
            window_size += 1

        # Apply smoothing
        if method == 'savgol':
            smoothed = savgol_filter(spectrum, window_length=window_size, polyorder=polyorder, mode='interp')

        elif method == 'gaussian':
            smoothed = gaussian_filter1d(spectrum, sigma=sigma, mode='reflect')

        elif method == 'moving_average':
            kernel   = np.ones(window_size) / window_size
            smoothed = np.convolve(spectrum, kernel, mode='same')

        else:
            raise ValueError("Method dont available. Use 'gaussian' or 'moving_average'.")

        # Size reduction if target_size is specified
        if target_size is not None and target_size < len(smoothed):
            x_original = np.linspace(0, 1, len(smoothed))
            x_target = np.linspace(0, 1, target_size)
            smoothed = np.interp(x_target, x_original, smoothed)

        return smoothed

if __name__ == "__main__":

    spectra_processor = Spectra(snii_only=True)
    spectra = spectra_processor.obtain_data()
    print(f"Total spectra: {len(spectra)}")

    processed_spectra = spectra_processor.process_spectrum(spectra, slice_spectrum=True)
    print(processed_spectra.head())
    
    oidx_list = processed_spectra.oid.unique()
    oidx = np.random.choice(oidx_list)
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    spec = processed_spectra[processed_spectra.oid == oidx].iloc[0]
    ax[0].plot(spec.wave, spec.flux, label='Original Flux', color='blue', alpha=0.5)
    ax[0].set_ylabel('Flux')
    ax[0].set_title(f'Spectrum OID: {spec.oid}')
    ax[0].legend()

    plt.show()
    # Guardar los espectros procesados en un archivo pickle
    #processed_spectra.to_pickle('../SNeMPhyVAE/data/processed_spectra_SNeII_ALeRCE20240801_x_wiserep_20240622.pkl')
    #print("Processed spectra saved.")