import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as astro_constants

class Rainbow():

    def __init__(self, wave_aa, time, reference_time,rise_time, fall_time,
                 amplitude, Tmin, delta_T, k_sig):
        
        self.wave_aa = wave_aa
        self.time = time
        self.reference_time = reference_time
        self.rise_time = rise_time
        self.fall_time = fall_time
        self.amplitude = amplitude
        self.Tmin = Tmin
        self.delta_T = delta_T
        self.k_sig = k_sig

    def temperature(self, Tmin, time, reference_time, delta_T, k_sig):

        num = delta_T
        den = 1.0 + np.exp((time - reference_time) / k_sig)
        
        return Tmin + num / den

    def bolometric_flux(self, time, reference_time, amplitude, rise_time, fall_time):
        
        num = np.exp(-(time - reference_time) / fall_time)
        den = 1.0 + np.exp((time - reference_time) / rise_time)
        
        return amplitude * num / den

    def blackbody_nu(self,wave_aa, T):
        """ Black-body spectral model
        Params
        -------
        - wave_aa: wavelength list in Angstron units
        - T: Temperature

        Returns
        -------
        - Spectral flux density like a BlackBody
        """

        CSPEED  = astro_constants.c.to('cm s-1').value    # [cm s-1]    
        HPLANCK = astro_constants.h.to('erg s').value     # [erg s]
        KBOLTZ  = astro_constants.k_B.to('erg K-1').value # [erg K-1]

        # Obtain the frequency in [cm] units
        nu = CSPEED / (wave_aa * 1e-8)

        # Black body radiation in frequency
        # Note: expm1 = exp(x) - 1
        B_nu = (2*HPLANCK*nu**3/CSPEED**2)/np.expm1(HPLANCK*nu/(KBOLTZ*T))

        return B_nu

    def spectral_flux_density(self, band_wave_aa):
        """Compute the observed spectral
        flux density per unit frequency nu

        Params
        -------
        band_wave_aa: float or array-like
            Wavelength(s) in Angstrom.

        Returns
        -------
        - Spectral flux density like a BlackBody
        """

        SSTEFANBOLTZ = astro_constants.sigma_sb.to('erg s-1 cm-2 K-4').value # [erg s-1 cm-2 K-4]
        
        T     = self.temperature(self.Tmin, self.time, self.reference_time, 
                                 self.delta_T, self.k_sig)
        F_bol = self.bolometric_flux(self.time, self.reference_time, self.amplitude,
                                     self.rise_time, self.fall_time)
        Flux = np.pi * self.blackbody_nu(band_wave_aa, T) / (SSTEFANBOLTZ * T**4) * F_bol
        #Flux = Flux.to('erg s-1 cm-2').value
        Flux /= np.max(Flux)
        
        return Flux

    """
    def plot_spectrum(self, band_wave_aa):
        Plot the observed spectral
        flux density per unit frequency nu

        Params
        -------
        band_wave_aa: float or array-like
            Wavelength(s) in Angstrom.

        Returns
        -------
        - Spectral flux density like a BlackBody
        

        Flux = self.spectral_flux_density(band_wave_aa)
        
        plt.plot(band_wave_aa, Flux)
        plt.xlabel("WaveLength")
        plt.ylabel("Flujo espectral")
        plt.show()
    """
print('Hola cargo, jeje')
print('Lo modifique desde aca')
rainbow_params = {
    'reference_time': 60000, 'rise_time': 6.0061, 'fall_time': 29.7081, 
    'amplitude': 1.5025e-15, 'Tmin': 5004.5664, 'delta_T': 50000, 'k_sig': 4.062
}
reference_time = 60000
wave_aa = np.linspace(1000, 10000, 500)  # Longitudes de onda
time = np.sort(np.random.uniform(low=reference_time - 3 * 6.0061, high=reference_time + 3 * 29.7081, size=10))

for t in time:
    rainbow = Rainbow(wave_aa, t, **rainbow_params)

    # Obtener el espectro a 5000 Å
    # Now, pass wave_aa to spectral_flux_density to calculate flux for all wavelengths
    flux = rainbow.spectral_flux_density(wave_aa)

    # 
    plt.plot(wave_aa, flux, label=f't = {t:.1f} días')
plt.xlabel("Longitud de onda (Å)")
plt.ylabel("Flujo (unidades arbitrarias)")
plt.legend()
plt.title("Evolución espectral en diferentes tiempos")
plt.grid(True)
plt.show()
