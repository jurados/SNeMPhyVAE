import argparse
from settings import initial_settings

def update_settings(sp_req=False, sn_req=False, ls_req=False):
    
    parser = argparse.ArgumentParser(description="Change the initial conditions of the model")
    parser.add_argument("-sn", "--spectra_nbins",    type=int,   default=500, required=sp_req, help="Number of bins for the spectra")
    parser.add_argument("-sp", "--spectra_penalty",  type=float, default=1,   required=sn_req, help="Penalty for the spectra")
    parser.add_argument("-ls", "--latent_space_dim", type=int,   default=3,   required=ls_req, help="Latent space dimension")
    args = parser.parse_args()

    initial_settings['spectrum_bins'] = args.spectra_nbins
    initial_settings['penalty_spectra'] = args.spectra_penalty
    initial_settings['latent_size'] = args.latent_space_dim
    

if __name__ == "__main__":
    
    print("Current initial settings:")
    for key, value in initial_settings.items():
        print(f"{key}: {value}")
    
    update_settings()

    print("=="*10)
    print("Updated initial settings:")
    for key, value in initial_settings.items():
        print(f"{key}: {value}")
