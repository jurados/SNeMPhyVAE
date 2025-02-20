import numpy as np
import pandas as pd

from settings import *

class Spectra():
    
    def __init__(self, snii_only=False):
        
        self.specta = None
        self.snii_only = snii_only
        
    def _load_data(self):
        
        object_table_path           = './Data/object_ALeRCExWiserep20240630_20240703.pkl'
        spectra_alercexwiserep_path = './Data/spectra_ALeRCE20240801_x_wisrep_20240622.pkl' 
        
        object_table           = pd.read_pickle(filepath_or_buffer=object_table_path)
        spectra_alercexwiserep = pd.read_pickle(filepath_or_buffer=spectra_alercexwiserep_path)
        
        self.spectra = pd.merge(left=spectra_alercexwiserep, right=object_table, on='oid')
        
        if self.snii_only:
            snii_labels = [label for label in self.spectra.true_label.unique() if 'SN II' in label]
            self.spectra = self.spectra[self.spectra.true_label.isin(snii_labels)]
            
        return self.spectra
        
    def obtain_data(self):
        
        return self._load_data()
        
    def _grid_flux(self, spectrum: pd.Series) -> (np.ndarray, np.ndarray, dict):
        """
        Genera la grilla de longitudes de onda y extrae los índices y el flujo no nulo.

        Parámetros:
        -----------
        spectrum : pd.Series
            Registro de espectro que debe contener 'lambda_grid_min', 'lambda_grid_max'
            y 'flux_lambda_smooth'.

        Retorna:
        --------
        flux_nonzero : np.ndarray
            Array con los valores de flujo en los índices no nulos.
        wave_nonzero : np.ndarray
            Array con las longitudes de onda correspondientes a los índices no nulos.
        state : dict
            Diccionario con información intermedia (por ejemplo, la grilla completa y los índices)
            para ser utilizado en etapas posteriores.
        """
        # Crear la grilla de longitudes de onda usando escala logarítmica
        wave_range = np.logspace(
            np.log10(spectrum.lambda_grid_min),
            np.log10(spectrum.lambda_grid_max),
            initial_settings['spectrum_bins']
        )
        
        flux = np.array((spectrum['flux_lambda_smooth'].tolist()))
        # Reemplazar NaN por 0 y valores negativos por 0
        flux = np.nan_to_num(flux, nan=0.0)
        flux[flux < 0] = 0.0

        # Índices donde el flujo es no nulo
        idx = np.nonzero(flux)[0]
        
        # Inicializar arrays con la forma del flujo (estos se pueden actualizar más adelante)
        continuum_flux  = np.zeros_like(flux)
        flux_normalized = np.zeros_like(flux)
        final_spectrum  = np.zeros_like(flux)

        flux_nonzero = flux[idx]
        wave_nonzero = wave_range[idx]
        
        state = {
            'wave_range': wave_range,
            'flux': flux,
            'idx': idx,
            'continuum_flux': continuum_flux,
            'flux_normalized': flux_normalized,
            'final_spectrum': final_spectrum
        }
        
        return flux_nonzero, wave_nonzero, state

    def normalize_spectrum(self, spectrum: pd.Series) -> (np.ndarray, np.ndarray, dict):
        """
        Normaliza el espectro dividiendo el flujo por su valor máximo.

        Parámetros:
        -----------
        spectrum : pd.Series
            Registro del espectro.

        Retorna:
        --------
        flux_normalized : np.ndarray
            Flujo normalizado.
        wave : np.ndarray
            Longitudes de onda correspondientes (solo para los índices no nulos).
        state : dict
            Diccionario con la grilla completa y otros datos intermedios.
        """
        flux_nonzero, wave_nonzero, state = self._grid_flux(spectrum)
        if flux_nonzero.size == 0:
            print(f"Warning: Empty flux array for spectrum with oid: {spectrum.oid}")
            return np.zeros_like(wave_nonzero), wave_nonzero, state
        
        flux_normalized = flux_nonzero / np.max(flux_nonzero)
        return flux_normalized, wave_nonzero, state

    def continuum_fitting(self, flux: np.ndarray, wave: np.ndarray, state: dict) -> np.ndarray:
        """
        Ajusta el continuo mediante un polinomio de grado 4 y normaliza el espectro.

        Parámetros:
        -----------
        flux : np.ndarray
            Flujo del espectro (normalizado).
        wave : np.ndarray
            Longitudes de onda correspondientes.
        state : dict
            Diccionario con información intermedia obtenido de _grid_flux.

        Retorna:
        --------
        spectrum_flux : np.ndarray
            Espectro con el ajuste del continuo aplicado (flujo dividido por el continuo).
        """
        continuum_coeffs = np.polyfit(wave, flux, deg=4)
        continuum_fit = np.polyval(continuum_coeffs, wave)
        # Guardar en el estado para su uso posterior
        state['flux_continuum_fit'] = continuum_fit
        spectrum_flux = flux / continuum_fit
        return spectrum_flux

    def forward(self, spectra: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa un conjunto de espectros y agrega columnas con la información procesada.

        Parámetros:
        -----------
        spectra : pd.DataFrame
            DataFrame que contiene registros de espectros.

        Retorna:
        --------
        new_spectra : pd.DataFrame
            DataFrame con los espectros procesados, incluyendo columnas como:
            'flux', 'wave', 'flux_continuum', 'flux_normalized' y 'final_spectrum'.
        """
        processed_spectra = []
        
        # Iterar sobre cada espectro (cada fila)
        for _, spectrum in spectra.iterrows():
            # Normalizar el espectro
            norm_flux, wave_nonzero, state = self.normalize_spectrum(spectrum)
            # Ajustar y remover el continuo
            spec_flux = self.continuum_fitting(norm_flux, wave_nonzero, state)
            
            # Actualizar los arrays en el estado en las posiciones de índices no nulos
            idx  = state['idx']
            flux = state['flux']
            flux_normalized = state['flux_normalized']
            continuum_flux  = state['continuum_flux']
            final_spectrum  = state['final_spectrum']
            
            flux_normalized[idx] = norm_flux
            continuum_flux[idx]  = state['flux_continuum_fit']
            final_spectrum[idx]  = spec_flux
            
            # Crear un diccionario con los resultados; se usa la grilla completa
            spec_dict = {
                'oid': spectrum.oid,
                'mjd': spectrum.mjd,
                'flux': flux,  # Array original procesado
                'wave': state['wave_range'],
                'flux_continuum': continuum_flux,
                'flux_normalized': flux_normalized,
                'final_spectrum': final_spectrum,
                'lambda_grid_min': spectrum.lambda_grid_min,
                'lambda_grid_max': spectrum.lambda_grid_max,
                'nlambda_grid': spectrum.nlambda_grid,
            }
            processed_spectra.append(spec_dict)
        
        new_spectra = pd.DataFrame(processed_spectra)
        return new_spectra
    
    def spectra_reference_time(self, spectra, lightcurves):
        # Seleccionar solo las columnas necesarias de light_curves
        reference_times = lightcurves[['oid', 'reference_time']].drop_duplicates()
    
        # Realizar un merge en la columna 'oid'
        spectra = spectra.merge(reference_times, on='oid', how='left')
        spectra['time_index'] = np.round((spectra['reference_time'] - spectra['mjd']) * initial_settings['sideral_scale']).astype(int) + initial_settings['time_window'] // 2
        return spectra