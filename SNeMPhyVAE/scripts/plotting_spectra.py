import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate

rc_params = {
    'lines.linewidth': 2,         # Set line width to 2 points
    'font.family': 'STIXGeneral', # Set color cycle for axes
    'ytick.direction': 'in',      # Set figure size in inches
    'xtick.direction': 'in',      # Set default font family
}

plt.rcParams.update(rc_params)

def obtain_lambda_grid(df:pd.Series) -> list[float]:

    lambda_grid = np.logspace(start=np.log10(df.lambda_grid_min),
                stop=np.log10(df.lambda_grid_max),
                num=df.nlambda_grid)

    return lambda_grid

def interpolate_flux(df:pd.Series) -> list[float]:
    
    lambda_grid = obtain_lambda_grid(df)
    
    flux = np.array(df.flux_lambda)
    lambda_data = np.logspace(start=np.log10(df.lambda_data_min),
                stop=np.log10(df.lambda_data_max),
                num=len(flux))

    f = interpolate.interp1d(lambda_data, flux, fill_value=np.nan, bounds_error=False)
    flux_new = f(lambda_grid)
    flux_new[np.isnan(flux_new)] = 0
    return flux_new 

def plot_spectra(df_spectra:pd.DataFrame, use_lambda_grid:bool = False, sn_name = None):
    
    fig, ax = plt.subplots()
    text_y = 1
    
    # Convert single series to DataFrame
    if isinstance(df_spectra, pd.Series):
        df_spectra = pd.DataFrame([df_spectra])
    
    if use_lambda_grid == True:
        for _, spectrum in df_spectra.iterrows():
            lambda_grid = obtain_lambda_grid(spectrum)
            flux = interpolate_flux(spectrum)
            ax.plot(lambda_grid, flux,label=f'{spectrum.mjd:.4f}MJD')
            # ax.text(
            #     x=max(lambda_grid) - 500,
            #     y=max(flux-text_y*4),
            #     s=f'{spectrum.mjd}MJD'
            # )
            text_y += 1
        
    else: 
        for  _, spectrum in df_spectra.iterrows():
            
            flux = spectrum.flux_lambda
            wavelength = np.logspace(start=np.log10(spectrum.lambda_data_min),
                                        stop=np.log10(spectrum.lambda_data_max),
                                        num=len(flux))
            ax.plot(wavelength, flux, label=f'{spectrum.mjd:.4f}MJD')    
            # ax.text(x=max(wavelength)-500,
            #         y=max(flux-text_y*4),
            #         s=f'{spectrum.mjd}MJD')
            text_y += 1
    
    if sn_name != None:
        ax.set_title(f'{sn_name}')

    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel('Flux')
    ax.legend(frameon=False)
    plt.show()

if __name__ == "__main__":
    spectra_master = pd.read_pickle('spectra_ALeRCE20240704_x_wisrep_20240622.pkl')
    spectra_test = spectra_master[spectra_master.snname == 'SN2016aqw']
    plot_spectra(spectra_test, use_lambda_grid=True)

    #for _, spectrum in spectra_test.iterrows():
    #    plot_spectra(df_spectra=spectrum, use_lambda_grid=False)



    