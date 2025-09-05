# ============= IMPORT LIBRARIES =============
import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline

# Import custom settings
from TMPhy_VAE.settings import initial_settings

# =============================================

class Spectra():

    def __init__(self, snii_only=False):

        self.snii_only = snii_only

    def _load_data(self):

        object_table_path           = './data/object_ALeRCExWiserep20240630_20240703.pkl'
        spectra_alercexwiserep_path = './data/spectra_ALeRCE20240801_x_wisrep_20240622.pkl'

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
            1838#,initial_settings['spectrum_bins']
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
        Parameters:
        -----------
        flux : '~np.ndarray'
            Flujo del espectro (normalizado).
        wave : '~np.ndarray'
            Longitudes de onda correspondientes.
        state : dict
            Diccionario con información intermedia obtenido de _grid_flux.

        Returns:
        --------
        spectrum_flux : '~np.ndarray'
            Espectro con el ajuste del continuo aplicado (flujo dividido por el continuo).
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

        # Además, eliminar cualquier duplicado (por si acaso)
        _, unique_idx = np.unique(wave_knots, return_index=True)
        wave_knots = wave_knots[unique_idx]
        flux_knots = flux_knots[unique_idx]

        #print('wave_knots len', len(wave_knots))
        #print('flux_knots len', len(flux_knots))
        #print('Es creciente:', np.all(np.diff(wave_knots) >= 0))

        spline = UnivariateSpline(wave_knots, flux_knots, k=3)  # k=3 es spline cúbico, s=0 ajuste exacto
        # Evaluar el continuo sobre toda la grilla
        continuum = spline(wave)
        #continuum = spline(wave_knots)

        # Guardar en el estado para su uso posterior
        state['continuum_flux'][state['mask']] = flux / continuum

        return flux / continuum

    def apodization(self, flux, state, fraction=0.05):
        """
        Aplica apodización tipo 'cosine bell' al principio y al final del espectro.

        Parameters
        ----------
        flux : np.ndarray
            Vector de flujo a apodizar.
        fraction : float
            Fracción del espectro donde aplicar la ventana (por defecto 5%).

        Returns
        -------
        flux_apodized : np.ndarray
            Vector de flujo apodizado.
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
        Procesa un conjunto de espectros y agrega columnas con la información procesada.

        Parameters:
        -----------
        spectra : '~pd.DataFrame'
            DataFrame que contiene registros de espectros.

        Returns:
        --------
        new_spectra : '~pd.DataFrame'
            DataFrame con los espectros procesados, incluyendo columnas como:
            'flux', 'wave', 'flux_continuum', 'flux_normalized' y 'final_spectrum'.
        """
        results = []

        if isinstance(spectra, pd.Series):
            spectra = pd.DataFrame([spectra])

        # Iterar sobre cada espectro (cada fila)
        for _, spectrum in spectra.iterrows():

            flux, wave, state = self._preprocess_flux(spectrum)
            flux += np.abs(np.min(flux))

            # Índices donde el flujo es no nulo
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
            #final_spectrum[idx]  = self.slice_spectrum(spec_flux, target_size=initial_settings['spectrum_bins'])

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
        spectra['time_index'] = np.round((spectra['reference_time'] - spectra['mjd']) * initial_settings['sideral_scale']).astype(int) + initial_settings['time_window'] // 2
        return spectra

    def slice_spectrum(self, spectrum, method='moving_average', window_size=9, polyorder=2, sigma=1.0, target_size=initial_settings['spectrum_bins']):
        """
        Suaviza un espectro y reduce su tamaño mediante interpolación o promediado.

        Parameters:
        ------------
        - spectrum (np.ndarray): Espectro de entrada (1D).
            method (str): Método de suavizado ('gaussian', 'moving_average').
            window_size (int): Tamaño de la ventana de suavizado (debe ser impar).
            sigma (float): Desviación estándar para el filtro gaussiano.
            target_size (int): Número de puntos deseado en el espectro reducido (ej. 300).

        Returns:
        --------
            np.ndarray: Espectro suavizado y reducido.
        """
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import savgol_filter

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
            raise ValueError("Método no reconocido. Usa 'gaussian' o 'moving_average'.")

        # Size reduction if target_size is specified
        if target_size is not None and target_size < len(smoothed):
            x_original = np.linspace(0, 1, len(smoothed))
            x_target = np.linspace(0, 1, target_size)
            smoothed = np.interp(x_target, x_original, smoothed)

        return smoothed