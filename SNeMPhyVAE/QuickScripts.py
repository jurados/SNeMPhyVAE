import argparse
from TMPhy_VAE.settings import initial_settings

if __name__ == "__main__":
    
    for key, value in initial_settings.items():
        print(f"{key}: {value}")
    
    parser = argparse.ArgumentParser(description="Change the initial conditions of the model")
    parser.add_argument("-sn", "--spectra_nbins", type=int, required=True, help="Number of bins for the spectra")
    parser.add_argument("-sp", "--spectra_penalty", type=float, required=True, help="Penalty for the spectra")
    parser.add_argument("-ls", "--latent_space_dim", type=int, required=True, help="Latent space dimension")
    args = parser.parse_args()

    initial_settings['spectrum_bins'] = args.spectra_nbins
    initial_settings['penalty_spectra'] = args.spectra_penalty
    initial_settings['latent_size'] = args.latent_space_dim

    print("Updated initial settings:")
    for key, value in initial_settings.items():
        print(f"{key}: {value}")
