import argparse
from settings import initial_settings

def update_settings(sertings_dict, sp_req=False, sn_req=False, ls_req=False):

    parser = argparse.ArgumentParser(description="Change the initial conditions of the model")
    parser.add_argument("-sn", "--spectra_nbins",    type=int,   default=500, required=sp_req, help="Number of bins for the spectra")
    parser.add_argument("-sp", "--spectra_penalty",  type=float, default=1,   required=sn_req, help="Penalty for the spectra")
    parser.add_argument("-ls", "--latent_space_dim", type=int,   default=3,   required=ls_req, help="Latent space dimension")
    args = parser.parse_args()

    sertings_dict['spectrum_bins'] = args.spectra_nbins
    sertings_dict['penalty_spectra'] = args.spectra_penalty
    sertings_dict['latent_size'] = args.latent_space_dim
    
    return sertings_dict

if __name__ == "__main__":
    
    print("Current initial settings:")
    for key, value in initial_settings.items():
        print(f"{key}: {value}")
    
    initial_settings = update_settings()

    print("=="*10)
    print("Updated initial settings:")
    for key, value in initial_settings.items():
        print(f"{key}: {value}")
