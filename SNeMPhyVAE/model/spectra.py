# ============= IMPORT LIBRARIES =============
import os
import sys
import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline

# Import custom settings
if os.getcwd().endswith('notebooks'):
    PROJECT_ROOT = os.path.dirname(os.getcwd())
    print(f"PROJECT_ROOT from spectra: {PROJECT_ROOT}")
    from SNeMPhyVAE.model.settings import initial_settings, band_info
else:
    PROJECT_ROOT = os.getcwd()
    from settings import initial_settings, band_info

#from settings import initial_settings
#from SNeMPhyVAE.model.settings import initial_settings

# =============================================

class Spectra():

    def __init__(self, settings=initial_settings, snii_only=False):

        self.snii_only = snii_only
        self.initial_settings = settings

    def _load_data(self):

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

        data = self._load_data()
        mask = data.flux_lambda_smooth.apply(lambda x: np.all(np.isnan(x)))
        #for _, spectrum in data.iterrows():
        #    if np.isnan(spectrum.flux_lambda_smooth).all():
        #        print(f"Warning: Empty flux array for spectrum with oid: {spectrum.oid}")
        #        continue

        return data[~mask].copy().reset_index(drop=True)

    def _preprocess_flux(self, spectrum):
        """
        Genera la grilla de longitudes de onda y extrae los índices y el flujo no nulo.

        Parameters:
        -----------
        spectrum : '~pd.Series'
            Registro de espectro que debe contener 'lambda_grid_min', 'lambda_grid_max'
            y 'flux_lambda_smooth'.

        Returns:
        --------
        flux_nonzero : '~np.ndarray'
            Array con los valores de flujo en los índices no nulos.
        wave_nonzero : '~np.ndarray'
            Array con las longitudes de onda correspondientes a los índices no nulos.
        state : dict
            Diccionario con información intermedia (por ejemplo, la grilla completa y los índices)
            para ser utilizado en etapas posteriores.
        """
        # Crear la grilla de longitudes de onda usando escala logarítmica
        wave_range = np.logspace(
            np.log10(spectrum.lambda_grid_min),
            np.log10(spectrum.lambda_grid_max),
            1838#,self.initial_settings['spectrum_bins']
        )
        # 1. De-redshifting (To rest-frame)
        redshift = np.nan_to_num(float(spectrum['redshift']))
        wave_range /= (1 + redshift)

        # Reemplazar NaN por 0 y valores negativos por 0
        flux = np.nan_to_num(spectrum['flux_lambda_smooth'], nan=0.0)

        state = {
            'oid':             spectrum.oid,
            'wave_range':      wave_range,
            'flux':            flux,
            'continuum_flux':  np.zeros_like(flux),
            'continuum_norm':  np.zeros_like(flux),
            'flux_apodized':   np.zeros_like(flux),
            'flux_normalized': np.zeros_like(flux),
            'final_spectrum':  np.zeros_like(flux)
        }

        #return flux[nonzero_mask], wave_range[nonzero_mask], state
        return flux, wave_range, state

    def normalize_spectrum(self, flux, state):
        """
        Normaliza el espectro dividiendo el flujo por su valor máximo.

        Parameters:
        -----------
        spectrum : '~pd.Series'
            Registro del espectro.

        Returns:
        --------
        flux_normalized : '~np.ndarray'
            Flujo normalizado.
        """
        if flux.size == 0:
            print(f"Warning: Empty flux array for spectrum")
            return np.zeros_like(flux)

        flux = np.nan_to_num(flux, nan=0.0)
        flux = np.clip(flux, 0, None)

        if np.all(flux == 0):
            return flux

        max_flux = np.max(flux)
        state['flux_normalized'][state['mask']] = flux / max_flux
        return flux/(max_flux)#, wave

    def continuum_fitting(self, flux: np.ndarray, wave: np.ndarray, state: dict):
        """
        
        Obtain the continuum fitting of a given spectrum using spline interpolation.
        
        Parameters:
        -----------
        flux : '~np.ndarray'
            specum flux (normalized).
        wave : '~np.ndarray'
            wavelengths.
        state : dict
            Dictionary with intermediate information obtained from _grid_flux.

        Returns:
        --------
        spectrum_flux : '~np.ndarray'
            Spectrum with the continuum fitting applied (flux divided by the continuum).
        """

        if flux.size < 13:
            print('There are not enought data. < 13')
            return flux

        wave = wave[state['mask']]
        flux = flux[state['mask']]

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

        # Save the continuum flux in the state dictionary
        state['continuum_flux'][state['mask']] = flux / continuum

        return flux / continuum

    def apodization(self, flux, state, fraction=0.05):
        """
        Apply apodization to the spectrum using a 'cosine bell' window at the start and end.

        Parameters
        ----------
        flux : np.ndarray
            Flux vector.
        fraction : float
            Spectrum fraction to apply the window (default 5%).

        Returns
        -------
        flux_apodized : np.ndarray
            Apodized flux vector.
        """
        if len(flux) == 0: # There are not enough data
            return flux

        n_apod = max(1, int(len(flux) * fraction))

        window = np.ones(len(flux))
        x = np.linspace(0, np.pi/2, n_apod)
        window[:n_apod]  = np.sin(x)**2
        window[-n_apod:] = np.sin(x[::-1])**2

        state['flux_apodized'][state['mask']] = flux * window
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

            flux, wave, state = self._preprocess_flux(spectrum)
            flux += np.abs(np.min(flux))

            # Index mask for valid flux values
            mask = (flux > 0) & np.isfinite(flux)
            state['mask'] = mask

            spec_flux = self.continuum_fitting(flux, wave, state)
            if len(spec_flux) == 0:
                continue

            spec_flux = spec_flux - 1

            spec_flux = self.apodization(spec_flux, state)
            if len(spec_flux) == 0:
                continue

            #spec_flux = self.normalize_spectrum(spec_flux, state)
            #if len(spec_flux) == 0:
            #    continue

            #np.clip(spec_flux, a_min=0.0, a_max=None, out=spec_flux)
            #mask = spec_flux > 0

            # Actualizar los arrays en el estado en las posiciones de índices no nulos
            #mask = state['nonzero_mask']
            #state['flux_normalized'][mask] = norm_flux
            #state['continuum_flux'][mask]  = state['flux_continuum_fit']

            #state['final_spectrum'][mask]  = spec_flux
            state['final_spectrum'][state['mask']]  = spec_flux

            #print(type(spec_flux))
            #final_spectrum[idx]  = self.slice_spectrum(spec_flux, target_size=self.initial_settings['spectrum_bins'])

            # Crear un diccionario con los resultados; se usa la grilla completa
            results.append({
                'oid':             spectrum.oid,
                'mjd':             spectrum.mjd,
                'redshift':        spectrum.get('redshift', np.nan),
                'flux':            state['flux'],
                'wave':            state['wave_range'],
                'flux_continuum':  state['continuum_flux'],
                'flux_apodized':   state['flux_apodized'],
                'flux_normalized': state['flux_normalized'],
                'final_spectrum':  state['final_spectrum'],
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