# ==================  INTRODUCTION ==================
# This script is used to train the MPhy_VAE model for the TMPhy challenge.
# It uses the PyTorch Lightning framework for training and WandB for logging.
# It is designed to handle light curves and spectra data, and it includes 
# a custom dataset class for loading.

# ================== IMPORT LIBRARIES ==================
import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import astropy.units as u
import sncosmo

# ================== IMPORT PYTORCH LIBRARIES ==================

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import WandbLogger

import wandb
# ================== IMPORT CUSTOM LIBRARIES ==================

from settings import initial_settings, band_info
from lightcurves import LightCurves
from spectra import Spectra
from metrics_callback import MetricsCallback

# =============================================================

wandb_key = open('../WANDB_API.key', 'r').read()
wandb.login(key=wandb_key)

#from metrics_callback import MetricsCallback

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Trabajando con: ', device)

# Load and process the light curves
lightcurves = LightCurves(snii_only=False).obtain_data()
MLightCurve = LightCurves(instrument='ztf')
lightcurves = MLightCurve.process_lightcurves(lightcurves)
print('Lightcurves', lightcurves.head())
print('Lightcurves columns:', lightcurves.columns)
print('Lightcurves shape:', lightcurves.shape)
print('Number of lightcurves:', len(lightcurves.oid.unique()))

# Load and process the spectra
spectra = Spectra(snii_only=False).obtain_data()
spectra = spectra[~spectra.oid.isin(['ZTF23aaoohpy', 'ZTF20aatzhhl', 'ZTF22aazuuin'])]
MSpectra = Spectra()
spectra  = MSpectra.process_spectrum(spectra, slice_spectrum=True)

print('Spectra', spectra.head())
print('Spectra columns:', spectra.columns)
print('Spectra shape:', spectra.shape)
print('Number of spectra:', len(spectra.oid.unique()))
common_oids = set(spectra.oid.unique()) & set(lightcurves.oid.unique())
spectra     = spectra[spectra.oid.isin(common_oids)]
lightcurves = lightcurves[lightcurves.oid.isin(common_oids)]

# Obtain the redshift
lightcurves = lightcurves.merge(spectra[['oid', 'redshift']], on='oid', how='left')
lightcurves['redshift'] = lightcurves['redshift'].replace('nan', 0.1)

spectra = MSpectra.spectra_reference_time(spectra, lightcurves)

# Find common OIDs
try:
    oids = list(set(spectra.oid.unique()) & set(lightcurves.oid.unique()))#) & set(rainbow_spectra.oid.unique()))# & set(photo_vel.oid.unique()))
except:
    pass

test_oids = []
if 'ZTF22aaeviey' in oids:
    test_oids.append('ZTF22aaeviey')
    oids.remove('ZTF22aaeviey')  # Remove it from the main list

# 2. Split the 80% remaining into TRAIN (80%) and TEST (20%)
remaining_train_oids, remaining_test_oids = train_test_split(oids, test_size=0.2)

# 3. Ensure the special OID is in the test set
test_oids.extend(remaining_test_oids)
train_oids = remaining_train_oids

# Divide the data into training and validation sets per modality
train_data_lightcurves = lightcurves[lightcurves.oid.isin(train_oids)]
train_data_spectra     = spectra[spectra.oid.isin(train_oids)]
#train_data_rainbow     = rainbow_spectra[rainbow_spectra.oid.isin(train_oids)]
#train_data_photovel    = photo_vel[photo_vel.oid.isin(train_oids)]

#val_data_lightcurves = lightcurves[lightcurves.oid.isin(val_oids)]
#val_data_spectra     = spectra[spectra.oid.isin(val_oids)]
#val_data_rainbow     = rainbow_spectra[rainbow_spectra.oid.isin(val_oids)]

test_data_lightcurves = lightcurves[lightcurves.oid.isin(test_oids)]
test_data_spectra     = spectra[spectra.oid.isin(test_oids)]


#train_data_lightcurves = pd.read_pickle('./data/train_data_lightcurves.pkl')
#train_data_spectra     = pd.read_pickle('./data/train_data_spectra.pkl')
#test_data_lightcurves  = pd.read_pickle('./data/test_data_lightcurves.pkl')
#test_data_spectra      = pd.read_pickle('./data/test_data_spectra.pkl')
#
#special_oid = 'ZTF22aaeviey'

#train_data_lightcurves.to_pickle('train_data_lightcurves.pkl')
#test_data_lightcurves.to_pickle('test_data_lightcurves.pkl')
#train_data_spectra.to_pickle('train_data_spectra.pkl')
#test_data_spectra.to_pickle('test_data_spectra.pkl')

#Función para reordenar un DataFrame, poniendo special_oid primero
#def reorder_df(df, oid_col='oid', special_oid=special_oid):
#    if special_oid in df[oid_col].values:
#        # Separar el special_oid y el resto
#        special_data = df[df[oid_col] == special_oid]
#        other_data = df[df[oid_col] != special_oid]
#        # Concatenar con special_oid primero
#        return pd.concat([special_data, other_data], ignore_index=True)
#    return df
#
## Aplicar a ambos DataFrames de test
#test_data_lightcurves = reorder_df(test_data_lightcurves)
#test_data_spectra     = reorder_df(test_data_spectra)

class CompleteDataset:
    def __init__(self, lightcurves, spectra):#0 rainbow, photovels):
        self.lightcurves = lightcurves
        self.spectra     = spectra
        #self.rainbow     = rainbow
        #self.photovels   = photovels

        # Agrupar por 'oid' y almacenar los grupos
        #self.oids = list(set(lightcurves.oid) & set(spectra.oid))# & set(rainbow.oid) & set(photovels.oid))
        oids = list(set(lightcurves['oid']) & set(spectra['oid'])) #& set(rainbow['oid']) & set(photovels['oid']))
        if 'ZTF22aaeviey' in oids:
            # Ensure 'ZTF22aaeviey' is at the beginning
            oids.remove('ZTF22aaeviey')
            oids.insert(0, 'ZTF22aaeviey')
        self.oids = oids

    def __len__(self):
        return len(self.oids)

    def __getitem__(self, idx):

        oid = self.oids[idx]
        lightcurve_group = self.lightcurves[self.lightcurves['oid'] == oid]
        spectrum_group   = self.spectra[self.spectra['oid'] == oid]
        #rainbow_group    = self.rainbow[self.rainbow['oid'] == oid]
        #photovel_group   = self.photovels[self.photovels['oid'] == oid]

        lightcurve_data = {
            'oid':            oid,
            'true_label':     lightcurve_group['true_label'],
            'mjd':            lightcurve_group['mjd'].to_numpy(dtype=np.float32),
            'redshift':       lightcurve_group['redshift'].astype(float).mean(),
            'fid':            lightcurve_group['fid'].to_numpy(dtype=np.uint8),
            'flux':           lightcurve_group['flux'].to_numpy(dtype=np.float32),
            'fluxerr':        lightcurve_group['fluxerr'].to_numpy(dtype=np.float32),
            'grid_time':      lightcurve_group['grid_time'].to_numpy(dtype=np.float32),
            'time_index':     lightcurve_group['time_index'].to_numpy(dtype=np.uint8),
            'reference_time': lightcurve_group['reference_time'].to_numpy(dtype=np.float32),
            'flux_scale':     lightcurve_group['flux_scale'].to_numpy(dtype=np.float32),
        }

        spectra_data = {
            'oid':             oid,
            'mjd':             spectrum_group['mjd'].to_numpy(),
            'flux':            spectrum_group['final_spectrum'].to_numpy(),
            'wave':            spectrum_group['wave'].to_numpy(),
            'lambda_grid_min': spectrum_group['lambda_grid_min'],
            'lambda_grid_max': spectrum_group['lambda_grid_max'],
            'nlambda_grid':    spectrum_group['nlambda_grid'].astype(int),
            'reference_time':  spectrum_group['reference_time'].astype(float),
            'time_index':      spectrum_group['time_index'].astype(int),
        }
        return lightcurve_data, spectra_data
    
train_dataset = CompleteDataset(train_data_lightcurves, train_data_spectra)
test_dataset  = CompleteDataset(test_data_lightcurves, test_data_spectra)
train_loader  = DataLoader(train_dataset, batch_size=64, collate_fn=list, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=64, collate_fn=list, shuffle=False)

def parse_settings(bands, settings={}, ignore_unknown_settings=False):
    """Parse the settings for a ParSNIP modl
    Parameters
    ----------
    bands : List[str]
        Bands to use in the encoder model
    settings : dict, optional
        Settings to override, by default {}
    ignore_unknown_settings : bool, optional
        If False (default), raise an KeyError if there are any unknown settings.
        Otherwise, do nothing.

    Returns
    -------
    dict
        Parsed settings dictionary

    Raises
    ------
    KeyError
        If there are unknown keys in the input settings
    """
    if 'derived_settings_calculated' in settings:
        # We are loading a prebuilt-model, don't recalculate everything.
        prebuilt_model = True
    else:
        prebuilt_model = False

    use_settings = initial_settings.copy()
    use_settings['bands'] = bands

    for key, value in initial_settings.items():
        if key not in initial_settings:
            if ignore_unknown_settings:
                continue
            else:
                raise KeyError(f"Unknown setting '{key}' with value '{value}'.")
        else:
            use_settings[key] = value

    #if not prebuilt_model:
    #    use_settings = update_derived_settings(use_settings)

    #if use_settings['model_version'] != default_settings['model_version']:
        # Update the settings to the latest version
    #    use_settings = update_settings_version(use_settings)

    return use_settings

class ResidualBlock(nn.Module):
    """1D residual convolutional neural network block

    This module operates on 1D sequences. The input will be padded so that length of the
    sequences is be left unchanged.

    Parameters
    ----------
    in_channels : int
        Number of channels for the input
    out_channels : int
        Number of channels for the output
    dilation : int
        Dilation to use in the convolution
    """
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        if self.out_channels < self.in_channels:
            raise Exception("out_channels must be >= in_channels.")

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, dilation=dilation,
                               padding=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3,
                               dilation=dilation, padding=dilation)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)

        # Add back in the input. If it is smaller than the output, pad it first.
        if self.in_channels < self.out_channels:
            pad_size = self.out_channels - self.in_channels
            pad_x = F.pad(x, (0, 0, 0, pad_size))
        else:
            pad_x = x

        # Residual connection
        out = out + pad_x
        out = F.relu(out)

        return out

class Conv1dBlock(nn.Module):
    """1D convolutional neural network block

    The Output Size using Conv1d in Pytorch could be calculated using:

    L_out = [Lin + 2 x padding - dilation x (kernel_size - 1) - 1] / stride + 1

    Parameters
    ----------
    in_channels : int
        Number of channels for the input
    out_channels : int
        Number of channels for the output
    dilation : int
        Dilation to use in the convolution
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=5, dilation=dilation, padding=2*dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x

class GlobalMaxPoolingTime(nn.Module):
    """Time max pooling layer for 1D sequences

    This layer applies global max pooling over all channels to elminate the channel
    dimension while preserving the time dimension.
    """
    def forward(self, x):
        out, inds = torch.max(x, 2)
        return out

def replace_nan_grads(parameters, value=0.0):
    """Replace NaN gradients

    Parameters
    ----------
    parameters : Iterator[torch.Tensor]
        Model parameters, usually you can get them by `model.parameters()`
    value : float, optional
        Value to replace NaNs with
    """
    for p in parameters:
        if p.grad is None:
            continue
        grads = p.grad.data
        grads[torch.isnan(grads)] = value

class MPhy_VAE(L.LightningModule):
    """Multimodal model of transients light curves,
    spectra and photospheric velocties
    """

    #def __init__(self, batch_size, device, bands, settings={}, ignore_unknown_settings=False, **kwargs):
    def __init__(self, batch_size, device, bands, settings={}, ignore_unknown_settings=False, **kwargs):

        super().__init__()

        # Parse settings
        self.settings = parse_settings(
            bands,
            settings,
            ignore_unknown_settings=ignore_unknown_settings
            )

        # Setup the bands
        self._setup_band_weights()

        # Setup the color law. We scale this so that the color law has a B-V color of 1,
        # meaning that a coefficient multiplying the color law is the b-v color.
        #color_law = extinction.fm07(self.model_wave, 3.1)
        #self.color_law = torch.FloatTensor(color_law).to(self.device)

        # Setup the timing
        self.input_times = (torch.arange(initial_settings['time_window'], device=device)
                            - initial_settings['time_window'] // 2)

        # Setup the model
        self._build_model()

        # Set up the training
        # self.epoch = 0

        # Send the model weights to the desired device
        #self.to(self.device, force=True)

        # Añade estas líneas para inicializar las listas de losses
        self.train_loss_epoch = []
        self.val_loss_epoch   = []
        self.latest_results   = None
        self.predictions      = []

        self.train_loss = 0.0
        self.val_loss = 0.0
        self.train_batches = 0
        self.val_batches = 0

        self.to(device)

    def _get_data(self, light_curves:list[dict]):
        """
        Parameters
        ----------
        - light_curves : list[dict]
            List of light curves

        Returns
        -------
        data : dict
            Dictionary with the data for the model
            - 'input_data' : A ~torch.FloatTensor that is used as input to the ParSNIP
            encoder.
            - 'compare_data': - 'compare_data' : A `~torch.FloatTensor` containing data
                that is used for comparisons with the output of the ParSNIP decoder.
            - 'redshift' : A `~torch.FloatTensor` containing the redshifts of each
                light curve.
            - 'band_indices' : A `~torch.LongTensor` containing the band indices for
                each observation that will be compared
        """
        redshifts = []
        compare_data = []
        compare_band_indices = []

        # This will be apply as part of RAINBOW.
        # The values of wavelength are in Angstrom units.
        #wave_aa = np.linspace(initial_settings['min_wave'], initial_settings['max_wave'],
        #                      initial_settings['spectrum_bins'])

        # Build the grid of flux for the input
        # In our case at first approximation could be equal to
        # [total_transients // batch_size, 2 or 3 (g,r, i), 300]
        grid_flux    = np.zeros((len(light_curves), len(initial_settings['bands']), initial_settings['time_window']))
        grid_weights = np.zeros_like(grid_flux)

        #for idx, light_curve in enumerate(light_curves.):
        for idx, light_curve in enumerate(light_curves):
            # This is because is easier work with pandas or astropy.tables
            # if we want create masks
            light_curve = pd.DataFrame(light_curve)

            redshifts.append(np.round(np.mean(light_curve['redshift']), decimals = 3))

            # Mask observations outside the window
            mask = (light_curve['time_index'] >= 0) & (light_curve['time_index'] < initial_settings['time_window'])
            light_curve = light_curve[mask]

            # Scale the flux and fluxerr appropiately.
            light_curve['flux']    /= light_curve['flux_scale']
            light_curve['fluxerr'] /= light_curve['flux_scale']

            # Calculate weights. The values of weights will be between 0 and infinity.
            weights = 1 / (light_curve['fluxerr']**2 + initial_settings['error_floor']**2)

            # Fill the input array. It's necessary to substract 1 to fid column,
            # because this will be used as an index. Something to remark is the grid_weights
            # note that is being multiply by "error_floor**2", then we get some as
            # error_floor**2 / (fluxerr**2 + error_floor**2"). In case that the fluxerr = 0
            # 1 and in other case will tend to 0. The values of grid_weights will be now
            # between 0 and 1.
            grid_flux[idx, light_curve['fid'].astype(int)-1, light_curve['time_index']]    = light_curve['flux']
            grid_weights[idx, light_curve['fid'].astype(int)-1, light_curve['time_index']] = initial_settings['error_floor']**2 * weights

            # obj_compare_data = [[grid_time],
            #                     [flux],
            #                     [fluxerr],
            #                     [weigths]]
            obj_compare_data = torch.FloatTensor(np.vstack([
                light_curve['grid_time'],
                light_curve['flux'],
                light_curve['fluxerr'],
                weights,
            ]))
            compare_data.append(obj_compare_data.T)
            # Porque fid es 1 y 2
            compare_band_indices.append(torch.LongTensor(light_curve['fid'].to_numpy().copy() - 1))

            #"""
            # Aplication of RAINBOW
            #self.rainbow_params = {
            #    'amplitude': light_curve.flux_scale.mean(),
            #    'reference_time': light_curve.reference_time.mean(),
            #    'rise_time': 71,
            #    'fall_time': 150,
            #    'Tmin': 5000,
            #    'delta_T': 5000,
            #    'k_sig': 100
            #}
            #time_index = np.arange(300)
            #times = LightCurves.grid_to_time(light_curve, time_index, light_curve.reference_time.mean())
            #rainbow_grid[idx, :, time_index] = [Rainbow(wave_aa, t, **self.rainbow_params).spectral_flux_density(wave_aa) for t in times]
            #rainbow_grid[idx, :, light_curve['time_index']] = [Rainbow(wave_aa, t, **self.rainbow_params).spectral_flux_density(wave_aa) for t in light_curve['mjd'].values]

        ####################################################

        #rainbow_grid = np.zeros((len(rainbow_spectra),
        #                         initial_settings['spectrum_bins'],
        #                         initial_settings['time_window'])
        #)
        #print('Rainbow_grid shape', rainbow_grid.shape)
        #for idx, rainbow_spectrum in enumerate(rainbow_spectra):
        #    try:
        #        for time_idx, flux in zip(rainbow_spectrum['time_index'],rainbow_spectrum['flux']):
        #            if 0 <= time_idx < initial_settings['time_window']:
        #                rainbow_grid[idx, :, time_idx] = flux
        #    except:
        #        print('Hola')

        ############## Photospheric velocities
        #grid_vph    = np.zeros((len(photo_velocities), 1, initial_settings['time_window']))

        #for idx, photo_vel in enumerate(photo_velocities):
        #    photo_vel = pd.DataFrame(photo_vel)

            # Mask observations outside the window
            #mask = (photo_vel['time_index'] >= 0) & (photo_vel['time_index'] < initial_settings['time_window'])
            #photo_vel = photo_vel[mask]

            #grid_vph[idx, 0, photo_vel['time_index']] = photo_vel['vphfe2']


        # Recopilar los datos de entrada.
        redshifts = np.array(redshifts)

        # TODO: Add this when I calculate the redshifts
        # This part is usefull if i predicted the redshift
        extra_input_data = [redshifts]

        # Stack everything together.
        input_data = np.concatenate(
                [i[:, None, None].repeat(initial_settings['time_window'], axis=2) for i in extra_input_data]
                + [grid_flux, grid_weights],#, grid_vph],
                axis=1
            )

        # Convert to torch tensors
        input_data = torch.FloatTensor(input_data).to(device)
        redshifts  = torch.FloatTensor(redshifts).to(device)

        #print('Input data shape', input_data.shape)

        #self.rainbow = torch.FloatTensor(rainbow_grid).to(device)

        # Pad all of the compare data to have the same shape.
        compare_data = nn.utils.rnn.pad_sequence(compare_data, batch_first=True) # This option keeps the dimension of batch on first column
        compare_data = compare_data.permute(0, 2, 1) # [batch_size, ]
        compare_band_indices = nn.utils.rnn.pad_sequence(compare_band_indices,batch_first=True)

        # OJo esto es nuevo, lo agrege para que hasta el input tenga dimension 300
        max_len = 300
        if compare_data.shape[2] < max_len:
            padding = torch.zeros((compare_data.shape[0], compare_data.shape[1], max_len - compare_data.shape[2]))
            padding_band = torch.zeros((compare_band_indices.shape[0], max_len - compare_band_indices.shape[1]))
            compare_data = torch.cat((compare_data, padding), dim=2)
            compare_band_indices = torch.cat((compare_band_indices, padding_band), dim=1)
        else:
            compare_data = compare_data[..., :max_len]
            compare_band_indices = compare_band_indices[..., :max_len]
        #print('Compare data shape', compare_data.shape)
        ###########################################

        compare_data         = compare_data.to(device)
        compare_band_indices = compare_band_indices.to(device)

        #print('Rainbow shape:', self.rainbow.shape)

        data = {
            'input_data':   input_data,
            'compare_data': compare_data,
            'redshift':     redshifts,
            'band_indices': compare_band_indices,
            #'rainbow': self.rainbow
        }

        # TODO: Add this when i try to predict the redshift

        return data

    def _build_model(self):
        """Build the model"""
        # N: Number of bands
        # 2*N: Represents the flux and fluxerr per each band
        input_size = len(initial_settings['bands']) * 2

        # TODO: add input_redshift and predict_redshift
        input_size += 1
        #input_size += initial_settings['spectrum_bins']*2

        # Including photospheric velocities
        #input_size += 1

        encode_block = ResidualBlock

        # Encoder architecture.  We start with an input of size input_size x
        # time_window We apply a series of convolutional blocks to this that produce
        # outputs that are the same size.  The type of block is specified by
        # settings['encode_block'].  Each convolutional block has a dilation that is
        # given by settings['encode_conv_dilations'].
        if (len(initial_settings['encode_conv_architecture']) !=
            len(initial_settings['encode_conv_dilations'])):
            raise Exception("Encoder layer size and dilations must habe the same length")

        # Encoder
        encode_layers = []

        # Convolutional layers.
        last_size = input_size
        # for layer_size in initial_initial_settings['encode_conv_architecture'][16, 32, 64]
        # for dilation   in initial_initial_settings['encode_conv_dilations'][1,1,1]
        for layer_size, dilation in zip(initial_settings['encode_conv_architecture'], initial_settings['encode_conv_dilations']):
            encode_layers.append(
                encode_block(in_channels=last_size, out_channels=layer_size, dilation=dilation)
                )
            last_size = layer_size

        # Fully connected layers for the encoder following the convolution blocks.
        # These are Conv1D layers with a kernel size of 1 that mix within the time
        # indexes.
        # for layer_size in initial_initial_settings['encode_fullyconnected_architecture][200]
        for layer_size in initial_settings['encode_fullyconnected_architecture']:
            encode_layers.append(nn.Conv1d(in_channels=last_size, out_channels=layer_size, kernel_size=1))
            encode_layers.append(nn.ReLU())
            last_size = layer_size

        self.encode_layers = nn.Sequential(*encode_layers)

        # ==================================================
        # Here begins the time-indexing

        # Fully connected layers for the time-indexing layer. These are Conv1D layers
        # with a kernel size of 1 that mix within time indexes.
        time_last_size = last_size
        encode_time_layers = []
        #for layer_size in self.initial_settings['encode_time_architecture'][200]
        for layer_size in initial_settings['encode_time_architecture']:
            encode_time_layers.append(nn.Conv1d(in_channels=time_last_size, out_channels=layer_size, kernel_size=1))
            encode_time_layers.append(nn.ReLU())
            time_last_size = layer_size

        # Final layer, go down to a single channel with no activation function.
        # This led us to "TimeIndexing" layer
        encode_time_layers.append(nn.Conv1d(time_last_size, 1, 1))
        self.encode_time_layers = nn.Sequential(*encode_time_layers)

        # ==================================================

        # Fully connected layers to calculate the latent space parameters for the VAE.
        encode_latent_layers = []
        latent_last_size = last_size
        #for layer_size in self.initial_settings['encode_latent_prepool_architecture'][200]
        for layer_size in initial_settings['encode_latent_prepool_architecture']:
            encode_latent_layers.append(nn.Conv1d(latent_last_size, layer_size, kernel_size=1))
            encode_latent_layers.append(nn.ReLU())
            latent_last_size = layer_size

        # Apply a global max pooling over the time channels.
        # GlobalMaxPoollingTime over the dim=2
        encode_latent_layers.append(GlobalMaxPoolingTime())

        # Apply fully connected layers to get the embedding.
        #for layer_size in self.initial_settings['encode_latent_postpool_architecture'][200]
        for layer_size in initial_settings['encode_latent_postpool_architecture']:
            encode_latent_layers.append(nn.Linear(latent_last_size, layer_size))
            encode_latent_layers.append(nn.ReLU())
            latent_last_size = layer_size

        self.encode_latent_layers = nn.Sequential(*encode_latent_layers)
        # Finally, use a last FC layer to get mu and logvar
        # The number 20 corresponds to original latent size
        # TODO: Why is +1 and +2
        mu_size = initial_settings['latent_size'] + 1
        logvar_size = initial_settings['latent_size'] + 2

        # TODO: ADd when i predict redshift

        # Linear layers for encoding mean and logvar
        # In ParSNIp the use latent_size = 3, so encode_mu_layer  = 4 and
        # encode_logvar = 5
        self.encode_mu_layer = nn.Linear(in_features=latent_last_size, out_features=mu_size)
        self.encode_logvar_layer = nn.Linear(in_features=latent_last_size, out_features=logvar_size)

        # MLP decoder. We start with an input that is the intrinsic latent space + one
        # dimension for time, and output a spectrum of size
        # self.initial_settings['spectrum_bins'].  We also have hidden layers with sizes given
        # by self.initial_settings['decode_layers'].  We implement this using a Conv1D layer
        # with a kernel size of 1 for computational reasons so that it decodes multiple
        # spectra for each transient all at the same time, but the decodes are all done
        # independently so this is really an MLP.
        decode_last_size = initial_settings['latent_size'] + 1
        decode_layers = []
        # for layer_size in initial_initial_settings['decode_architecture'][64, 32, 16]
        for layer_size in initial_settings['decode_architecture']:
            decode_layers.append(nn.Conv1d(in_channels=decode_last_size, out_channels=layer_size, kernel_size=1))
            decode_layers.append(nn.Tanh())
            decode_last_size = layer_size

        # Final layer. Use a FC layer to get us to the correct number of bins, and use
        # a softplus activation function to get positive flux.
        decode_layers.append(nn.Conv1d(in_channels=decode_last_size,
                                       out_channels=initial_settings['spectrum_bins'],
                                       kernel_size=1))
        # I change this the last 25/07/22
        decode_layers.append(nn.Softplus())
        #decode_layers.append(nn.Tanh())

        self.decode_layers = nn.Sequential(*decode_layers)


    def encode(self, input_data):
        """ Predict the latent variables for a set of light curves

        It used variational inference, and predict the parameters of a posterior
        distribution over the latent variables.

        Parameters
        ----------
        input_data : '~torch.FloatTensor'
            Input data representing a set of gridded light curves.

        Returns
        -------
        encoding_mu : '~torch.FloatTensor'
            The mean of the posterior distribution over the latent variables.
        encoding_logvar : ~torch.FloatTensor'
            The log variance of the posterior distribution over the latent variables.

        """
        # Apply common encoder blocks
        enc = self.encode_layers(input_data)

        # Reference time branch. First, apply additonal FC layers to get to an
        # output thas has a single channel.
        enc_time = self.encode_time_layers(enc)

        # Apply the time-indexing layer to calculate the reference time. This is a
        # special layer that is invariant to translations of the input.
        t_vec = torch.nn.functional.softmax(torch.squeeze(enc_time, 1), dim=1)

        ref_time_mu = (
            torch.sum(t_vec * self.input_times, 1)
            / initial_settings['time_sigma']
        )

        # Latent space branch.
        enc_latent = self.encode_latent_layers(enc)

        # Predict mu and logvar
        encoding_mu     = self.encode_mu_layer(enc_latent)
        encoding_logvar = self.encode_logvar_layer(enc_latent)

        # Prepend the time mu value to get the full encoding.
        encoding_mu = torch.cat([ref_time_mu[:, None], encoding_mu], 1)
        #print('encodig_mu shape', encoding_mu.shape)

        # Avoid crazy values for the log_Var
        encoding_logvar = torch.clamp(encoding_logvar, None, 5.)

        return encoding_mu, encoding_logvar

    def decode_spectra(self, encoding, phases, color, amplitude=None):
        """Predict the spectra at a given set of latent variables

        Parameters
        ---------
        encoding : '~torch.FloatTensor'
            Coordinates in the ParSNIP intrinsic latent space for each light curve
        phases : '~torch.FloatTensor'
            Phases to decode each light curve at
        color : '~torch.FloatTensor'
            Color of each light curve
        amplitude : '~torch.FloatTensor'
            Amplitude of each light curve

        Returns
        ---------
        model_spectra : '~torch.FloatTensor'
            The predicted spectra.
        """
        scale_phase = phases / (initial_settings['time_window'] // 2)

        repeat_encoding = encoding[:, :, None].expand((-1,-1, scale_phase.shape[1]))
        stack_encoding = torch.cat([repeat_encoding, scale_phase[:, None, :]], 1)

        # Apply intrinsic decoder
        model_spectra = self.decode_layers(stack_encoding)


        if color is not None:
            # Apply colors
            apply_colors = 10**(-0.4 * color[:, None])
            model_spectra = model_spectra * apply_colors[..., None]

        # TODO: amplitude
        if amplitude is not None:
            # Apply amplitudes
            model_spectra = model_spectra * amplitude[:, None, None]

        return model_spectra

    def decode(self, encoding, ref_times, color, times, redshifts, band_indices, amplitude=None):
        """Predict the light curves for a given set of latent variables

        Parameters
        ----------
        encoding : '~torch.FloatTensor'
            Coordinates in the ParSNIP intrinsic latent space for each light curve
        ref_times : '~torch.FloatTensor'
            Reference times for each light curve
        color: '~torch.FloatTensor'
            Color of each light curve
        times : '~torch.FloatTensor'
            Times to predict each light curve at
        redshifts : '~torch.FloatTensor'
            Redshifts of each light curve
        band_indices : '~torch.LongTensor'
            Band indices for each observation in the light curve
        amplitude: '~torch.FloatTensor'
            Amplitude to scale each light curve

        Returns
        -------
        model_spectra : '~torch.FloatTensor'
            The predicted spectra.
        model_flux : '~torch.FloatTensor'
            The predicted fluxes.
        """
        phase = ((times - ref_times[:, None]) / (1 + redshifts[:, None]))

        # Generate the restframe spectra
        model_spectra = self.decode_spectra(encoding, phase, color, amplitude)

        # Figure out the weights for each band
        band_weights = self._calculate_band_weights(redshifts)
        num_batches  = band_indices.shape[0]
        num_observations = band_indices.shape[1]
        batch_indices = (
            #torch.arange(num_batches, device=encoding.device)
            torch.arange(num_batches)
            .repeat_interleave(num_observations)
        )
        band_indices = band_indices.type(torch.long)
        obs_band_weights = (
            band_weights[batch_indices, :, band_indices.flatten()]
            .reshape((num_batches, num_observations, -1))
            .permute(0, 2, 1)
        )

        # Sum over each filter
        # Flux (Luminosity) = Sum (w_i * F_lambda_i)
        model_flux = torch.sum(obs_band_weights * model_spectra, axis=1)

        return model_spectra, model_flux

    def _reparameterize(self, mu, logvar, sample=True):
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def _sample(self, encoding_mu, encoding_logvar, sample=True):
        sample_encoding = self._reparameterize(encoding_mu, encoding_logvar, sample)

        time_sigma  = initial_settings['time_sigma']
        color_sigma = initial_settings['color_sigma']

        #TODO: Add when if i predict the redshift
        redshift = torch.zeros_like(sample_encoding[:, 0])

        # Rescale variables
        ref_times = sample_encoding[:, 0] * time_sigma
        color     = sample_encoding[:, 1] * color_sigma
        encoding  = sample_encoding[:, 2:]

        ref_times = torch.clamp(ref_times, -10 * time_sigma, 10 * time_sigma)
        color     = torch.clamp(color, -10 * color_sigma, 10 * color_sigma)
        redshift  = torch.clamp(redshift, 0. , initial_settings['max_redshift'])

        return redshift, ref_times, color, encoding

    def _compute_amplitude(self, weight, model_flux, flux):
        """
        This is use for calculate the amplitude, it does not have
        the units, beacuse this is normalizaed

        Parameters
        ----------
        - model_flux : '~torch.FloatTensor'

        Returns
        --------
        - amplitude_mu : '~torch.FloatTensor'
        - amplitude_logvar : '~torch.FloatTensor'
        """
        # This is
        # sum (wi * model_flux_i * flux_i) / sum(w_i + model_flux**2)
        # remember that wi = [0 or 1]
        num = torch.sum(weight * model_flux * flux, axis=1)
        den = torch.sum(weight * model_flux * model_flux, axis=1)

        # This replace values equal to 0 by 1e-5 to avoid infty values
        den[den == 0.] = 1e-5

        # WHY the logvar is log(1 / edn)
        amplitude_mu     = num / den
        amplitude_logvar = torch.log(1./ den)

        return amplitude_mu, amplitude_logvar

    def _setup_band_weights(self):
        """Setup the interpolation for the band weights used for photometry"""
        # This create the wavelength grid
        # Build the model in log wavelength
        model_log_wave = np.linspace(np.log10(initial_settings['min_wave']),
                                     np.log10(initial_settings['max_wave']),
                                     initial_settings['spectrum_bins'])
        # This calculate the delta_lambda
        model_spacing = model_log_wave[1] - model_log_wave[0]

        # NO SE PORQUE PASA ESSTO AÚN
        band_spacing = model_spacing / initial_settings['band_oversampling']

        # This calculate the max_wavelength including the redshift displacement
        # plus band_spacing
        band_max_log_wave = (
            np.log10(initial_settings['max_wave']* (1 + initial_settings['max_redshift']))
            + band_spacing
        )

        # Oversampling must be odd.
        assert initial_settings['band_oversampling'] % 2 == 1, "Band oversampling must be odd"
        pad = (initial_settings['band_oversampling'] - 1) // 2
        band_log_wave = np.arange(np.log10(initial_settings['min_wave']),
                                  band_max_log_wave,
                                  band_spacing)
        band_wave = 10**(band_log_wave)

        band_pad_log_wave = np.arange(
            np.log10(initial_settings['min_wave']) - band_spacing * pad,
            band_max_log_wave + band_spacing * pad,
            band_spacing
        )

        # ================================================================= #
        # THIS IS NEW, DELETE IT IF IT DOESN'T WORK

        #band_pad_wave_center = 10**(band_pad_log_wave)
        #mask = (
        #    band_pad_wave_center >= initial_settings['lambda_min_mask']
        #    ) & (
        #    band_pad_wave_center <= initial_settings['lambda_max_mask']
        #)
        #band_pad_log_wave = np.where(mask, band_pad_log_wave, 0.0)
        #band_pad_wave_lower  = 10**(band_pad_log_wave - band_spacing)
        #band_pad_wave_upper  = 10**(band_pad_log_wave + band_spacing)
        #band_pad_dwave = np.where(
        #    mask,
        #    band_pad_wave_upper - band_pad_wave_lower,
        #    0.0
        #)
        # ================================================================= #

        band_pad_dwave = (
            10**(band_pad_log_wave + band_spacing / 2.)
            - 10**(band_pad_log_wave - band_spacing / 2.)
        )

        ref = sncosmo.get_magsystem(initial_settings['magsys'])

        band_weights = []
        for band_name in initial_settings['bands']:
            band = sncosmo.get_bandpass(band_name)
            band_transmission = band(10**(band_pad_log_wave))

            # Convolve the bands to match the sampling of the spectrum.
            band_conv_transmission = np.convolve(
                band_transmission * band_pad_dwave,
                np.ones(initial_settings['band_oversampling']),
                mode='valid'
            )
            band_weight = (
                band_wave * band_conv_transmission / sncosmo.constants.HC_ERG_AA / ref.zpbandflux(band) * 10**(0.4 * -20.)
            )
            band_weights.append(band_weight)

        # get the locations at any redshift.
        band_interpolate_locations = torch.arange(
            0,
            initial_settings['spectrum_bins'] * initial_settings['band_oversampling'],
            initial_settings['band_oversampling']
        )

        trans = 0
        for band_name in initial_settings['bands']:
            band = sncosmo.get_bandpass(band_name)
            band_transmission += band(10**(band_pad_log_wave))

        # Save the variables that we need to do interpolation.
        # band_pad_log_wave: Contiene puntos en escala logarítmica, incluyendo un margen adicional.
        # band_pad_dwave: Calcula el tamaño del intervalo en escala lineal para cada punto en band_pad_log_wave,
        # considerando el espaciado en la escala logarítmica.
        self.band_transmission          = trans
        self.band_interpolate_locations = band_interpolate_locations.to(device)
        self.band_interpolate_spacing   = band_spacing
        self.band_interpolate_weights   = torch.FloatTensor(band_weights).to(device)
        self.model_wave                 = 10**(model_log_wave)

    def _calculate_band_weights(self, redshifts):
        """Calculate the band weights for a given set of redshifts

        Parameters
        ----------
        redshifts : 'list[float]'
            Redshifts to calculate the band light curve the band
            weights at.

        Returns
        -------
        band_weights : '~numpy.ndarray'
            Band weights for each redshift/band combination.
        """

        # Figure out the locations to sample at for each redshift.
        locs = (
            self.band_interpolate_locations + torch.log10(1 + redshifts)[:, None] / self.band_interpolate_spacing
        )
        flat_locs = locs.flatten()

        # Linear interpolation
        int_locs   = flat_locs.long()
        remainders = flat_locs - int_locs

        # This commando ... avoid de use of [:,:, int_locs]
        start = self.band_interpolate_weights[..., int_locs]
        end   = self.band_interpolate_weights[..., int_locs + 1]

        flat_result = remainders * end + (1 - remainders) * start
        result      = flat_result.reshape((-1,) + locs.shape).permute(1, 2, 0)

        # We need an extra term of 1 + z from the filter contraction.
        result /= (1 + redshifts)[:, None, None]

        return result

    def forward(self, light_curves,
                #rainbow_spectra,
                sample=True, to_numpy=False):
        """ Run a set of light curves

        Parameters
        ----------
        light_curves: list[~pd.DataFrame]
            List of light cruves
        sample: bool, optinal
            If True (default), sample from the posterior distribution. If False,
            use the MAP.
        to_numpy: bool, optinal.
            Whether to convert the outputs to numpy arrays, by default False

        Returns
        -------
        results: dict
            Results dictionary. If to_numpy is True, all of the elements will be
            numpy arrays. Otherwise, they will be Pytorch tensors.
        """
        data = self._get_data(light_curves)#, rainbow_spectra)

        # Metadata
        metadata = {
            'oid': [lc['oid']  for lc in light_curves],
            'true_label': [lc['true_label'].iloc[0] for lc in light_curves]
        }
        #print('METADATA')
        #print((len(metadata['oid']), len(metadata)))

        # Encode the light_curves
        encoding_mu, encoding_logvar = self.encode(data['input_data'])

        # Sample from the latent space.
        predicted_redshifts, ref_times, color, encoding = self._sample(
            encoding_mu, encoding_logvar, sample=sample
        )

        #TODO: Add if i predict the redshift
        use_redshifts = data['redshift']

        time        = data['compare_data'][:,0]
        obs_flux    = data['compare_data'][:,1]
        obs_fluxerr = data['compare_data'][:,2]
        obs_weight  = data['compare_data'][:,3]

        # Decode the light_curves

        model_spectra, model_flux = self.decode(
            encoding, ref_times, color, time, use_redshifts, data['band_indices']
        )


        # Analytically evaluate the conditional distribution for the amplitude and
        # sample from it.
        amplitude_mu, amplitude_logvar = self._compute_amplitude(obs_weight, model_flux,
                                                                 obs_flux)
        amplitude = self._reparameterize(amplitude_mu, amplitude_logvar)

        model_flux    = model_flux * amplitude[:, None]
        model_spectra = model_spectra * amplitude[:, None, None] #+ data['rainbow']
        #model_spectra = self._smooth_spectra(model_spectra)
        model_spectra, model_spectra_continuum, apod_spectra, spline_spectra, initial_spectra = self._astrodash_normalization(model_spectra, self.model_wave, is_smoothed=True, return_continuum=True)
        #model_astrodash = self._astrodash_normalization(model_spectra, self.model_wave, is_smoothed=True, return_continuum=True)
        #model_spectra, model_spectra_continuum, apod_spectra, spline_spectra, initial_spectra = model_astrodash
        #model_spectra_continuum *= amplitude[:, None, None]

        #model_wave = self.model_wave
        #mask = (model_wave > initial_settings['lambda_min_mask']) & (model_wave < initial_settings['lambda_max_mask'])
        #mask = torch.tensor(mask, dtype=torch.float32, device=device).view(1,-1,1)

        #print('mask', mask.shape)
        #print("Máscara ejemplo (primer elemento):")
        #print(mask[0, :, 0])

        #model_spectra = model_spectra  #* mask
        #model_rainbow = data['rainbow'] #* mask

        #model_spectra = model_spectra / torch.max(model_spectra, dim=1, keepdim=True)[0]
        #model_spectra_with = (model_spectra + model_rainbow) / torch.max(model_spectra + model_rainbow, dim=1, keepdim=True)[0]

        # Save the results in a dictionary
        results ={
            'oid':              metadata['oid'],
            'true_label':       metadata['true_label'],
            'ref_times':        ref_times,
            'color':            color,
            'encoding':         encoding,
            'amplitude':        amplitude,
            'redshift':         data['redshift'],
            'time':             time,
            'obs_flux':         obs_flux,
            'obs_fluxerr':      obs_fluxerr,
            'obs_weight':       obs_weight,
            'band_indices':     data['band_indices'],
            'model_wave':       self.model_wave,
            'model_flux':       model_flux,
            'model_spectra':    model_spectra,
            'encoding_mu':      encoding_mu,
            'encoding_logvar':  encoding_logvar,
            'amplitude_mu':     amplitude_mu,
            'amplitude_logvar': amplitude_logvar,
            #'rainbow':          model_rainbow,
            #'model_spectra_with': model_spectra_with
        }

        try:
            results['model_spectra_continuum'] = model_spectra_continuum
            results['apod_spectra']    = apod_spectra
            results['spline_spectra']  = spline_spectra
            results['initial_spectra'] = initial_spectra
        except:
            pass

        if to_numpy:
            # This convert the values to a numpy but first return them to cpu()
            results = {k:v.detach().cpu().numpy() for k,v in results.items()}

        return results

    def loss_function(self, results, spectra, return_components=False, return_individual=False):
    #def loss_function(self, results, return_components=False, return_individual=False):
        """
        Compute the loss function for the VAE on a set of light curves.

        Parameters
        ----------
        results : dict
            Dictionary containing the results of the model.
         return_components : bool, optional
            Whether to return the individual parts of the loss function, by default
            False.
        return_individual : bool, optional
            Whether to return the loss function for each light curve individually, by
            default False.
        Returns
        -------
        loss : torch.FloatTensor
            If return_components and return_individual are False, return a single value
            representing the loss function for a set of light curves.
            If return_components is True, then we return a set of four values
            representing the negative log likelihood, the KL divergence, the
            regularization penalty, and the amplitude probability.
            If return_individual is True, then we return the loss function for each
            light curve individually.
        """
        # Reconstruction likelihood
        # nll: normalized log-likelihodd
        # nll = 1/2 * weight_from_light_cruve * (obs_flux (y) - model_flux(hat_y))**2
        nll = (0.5 * results['obs_weight'] * (results['obs_flux'] - results['model_flux'])**2)

        # Kl divergence
        kld = -0.5 * (1 + results['encoding_logvar']
                      - results['encoding_mu']**2
                      - results['encoding_logvar'].exp())

        model_wave = self.model_wave
        mask = (model_wave > initial_settings['lambda_min_mask']) & (model_wave < initial_settings['lambda_max_mask'])
        mask = torch.tensor(mask, dtype=torch.float32, device=device).view(1,-1,1)

        #mask_spectra = torch.ones_like(results['model_spectra_with'])
        #idx_mask     = torch.where(wave_min < x <wave_max, results['model_spectra_with'], dim=2)
        #mask_spectra[idx_mask] = 0

        model_spectra = results['model_spectra'] * mask
        #model_rainbow = results['rainbow'] * mask

        # Regularization of spectra
        # diff = (model_spectra_i - model_spectra_j) / (model_spectra_i + model_spectra_j)
        # This is called Bray-Curtis distance, review the notes of ML in Notion
        #diff = (
        #    (results['model_spectra_with'][:, 1:, :] - results['model_spectra_with'][:, :-1, :])
        #    / (results['model_spectra_with'][:, 1:, :] + results['model_spectra_with'][:, :-1, :])
        #)
        diff = (
            (model_spectra[:, 1:, :] - model_spectra[:, :-1, :])
            / (model_spectra[:, 1:, :] + model_spectra[:, :-1, :] + 1e-8)
        )
        smooth = initial_settings['penalty'] * diff**2 #* #1e2*

        # Amplitude probability for the importance sampling integral
        amp_prob = -0.5 * ((results['amplitude'] - results['amplitude_mu'])**2
                            / results['amplitude_logvar'].exp()) # * 1e-1

        ###############################################################
        # SPECTRA

        grid_spectra = torch.zeros(
            (
            len(spectra),
            initial_settings['spectrum_bins'],
            initial_settings['time_window']
            ),
            device=device
        )
        for idx, spectrum in enumerate(spectra):
           spectrum = pd.DataFrame(spectrum)
           mask_spectrum = spectrum['time_index'].between(0, initial_settings['time_window']-1)
           spectrum = spectrum[mask_spectrum]

           if not spectrum.empty:
                # Convertir directamente a tensores
                times = torch.tensor(spectrum['time_index'].values, dtype=torch.long, device=device)
                fluxes = torch.tensor(np.stack(spectrum['flux'].values), device=device).float()

                # Asignación
                grid_spectra[idx, :, times] = fluxes.T

        #print(grid_spectra)
        #print(type(grid_spectra[0, 0, 0]))
        grid_spectra = grid_spectra * mask.to(device)

        # This is a new loss fuction
        # c = sum(i=0; N) Delta_spectra / N donde N is el numero de datos
        direct_spectra = initial_settings['penalty_spectra'] * ((model_spectra - grid_spectra)**2) / initial_settings['spectrum_bins']# / ((model_spectra)**2 + 1e-10) # / initial_settings['time_window']

        #print('grid_spectra shape:',grid_spectra.shape)

        #print('model spectra shape', results['model_spectra'].shape)


        ###################################

        # Redshift error
        # TODO: Add this when I calculate the redshift

        # Regularization
        #print("Shape of model_spectra['model_spectra']:", model_spectra['model_spectra'].shape)
        #print("Shape of self.rainbow:", self.rainbow.shape)
        #print("Valores mínimos en rainbow:", torch.min(results['rainbow']))
        #print("Número de ceros en rainbow:", torch.sum(results['rainbow'] == 0))

        #Lspec_rainbow = (model_spectra - results['rainbow'])**2/((results['rainbow'] * 0.5)**2 + 1e-8)
        #Lspec_rainbow = (model_spectra - model_rainbow)**2/((model_rainbow * 0.5)**2 + 1e-8)
        #print('Lspec_rainbow shape', Lspec_rainbow.shape)
        #print('penalty.shape', penalty.shape)
        # Registrar las métricas para que el callback pueda acceder

        if return_individual:
            nll            = torch.sum(nll, axis=1)
            kld            = torch.sum(kld, axis=1)
            smooth         = torch.sum(torch.sum(smooth, axis=2), axis=1)
            amp_prob       = torch.sum(amp_prob, axis=1)
            direct_spectra = torch.sum(direct_spectra, axis=1)
            #Lspec_rainbow = torch.sum(torch.sum(Lspec_rainbow, axis=2), axis=1)
        else:
            nll            = torch.sum(nll)
            kld            = torch.sum(kld)
            smooth         = torch.sum(smooth)
            amp_prob       = torch.sum(amp_prob)
            direct_spectra = torch.sum(direct_spectra)

            self.log("loss/nll",            nll,            on_step=False, on_epoch=True, prog_bar=False, batch_size=initial_settings['batch_size'])
            self.log("loss/kld",            kld,            on_step=False, on_epoch=True, prog_bar=False, batch_size=initial_settings['batch_size'])
            self.log("loss/smooth",         smooth,         on_step=False, on_epoch=True, prog_bar=False, batch_size=initial_settings['batch_size'])
            self.log("loss/amp_prob",       amp_prob,       on_step=False, on_epoch=True, prog_bar=False, batch_size=initial_settings['batch_size'])
            self.log("loss/direct_spectra", direct_spectra, on_step=False, on_epoch=True, prog_bar=False, batch_size=initial_settings['batch_size'])

        if return_components:
            return torch.stack([nll, kld, smooth, amp_prob, direct_spectra])
        else:
            return nll + kld + smooth + amp_prob + direct_spectra# + Lspec_rainbow

    def training_step(self, batch, batch_idx):
        light_curves = [item[0] for item in batch]
        spectra      = [item[1] for item in batch]
        #rainbow      = [item[2] for item in batch]

        results = self.forward(light_curves)#, rainbow)
        loss = self.loss_function(results, spectra)

        #print(f'Training Loss en steo: {loss}')

        # Acumular la pérdida y contar batches
        self.train_loss += loss.detach()
        self.train_batches += 1

        self.log(
            'train_loss', loss.mean(),
            batch_size=initial_settings['batch_size'],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        #self.latest_results = results

        return loss.mean()

    def validation_step(self, batch, batch_idx):

        def to_device(value, device):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            elif isinstance(value, np.ndarray):
                if value.dtype == object:  # Si es array de objetos
                    # Convertir cada elemento individualmente
                    return [to_device(x, device) for x in value]
                else:
                    return torch.tensor(value, device=device)
            elif isinstance(value, (list, tuple)):
                return [to_device(x, device) for x in value]
            elif isinstance(value, (int, float)):
                return torch.tensor(value, device=device)
            return value  # Para strings y otros tipos no convertibles

        #light_curves = torch.stack([item[0] for item in batch]).to(device)
        light_curves = [item[0] for item in batch]
        spectra      = [item[1] for item in batch]
        #rainbow      = [item[2] for item in batch]

        results = self.forward(light_curves)#, rainbow)
        loss    = self.loss_function(results, spectra)
        #loss = self.loss_function(results)

        # Acumular la pérdida y contar batches
        self.val_loss += loss.detach()
        self.val_batches += 1

        if batch_idx == 0:
            self.latest_lightcurves = light_curves
            self.latest_spectra     = spectra
            self.latest_results     = results

        self.log(
            'val_loss', loss.mean(),
            batch_size=initial_settings['batch_size'],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        return loss.mean()

    def test_step(self, batch, batch_idx):
        light_curves = [item[0] for item in batch]
        spectra      = [item[1] for item in batch]
        rainbow      = [item[2] for item in batch]

        results = self.forward(light_curves, rainbow)
        loss = self.loss_function(results, spectra)

        #self.predictions.append(results)

        # Registrar métricas (opcional)
        self.log('test_loss', loss.mean(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get('train_loss')
        if loss is not None:
            self.train_loss_epoch.append([self.current_epoch, loss.item()])

    def predict_step(self, batch, batch_idx):
        with torch.inference_mode():
            light_curves = [item[0] for item in batch]
            spectra = [item[1] for item in batch]
            rainbow = [item[2] for item in batch]

            results = self(light_curves, rainbow)
            return results


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=initial_settings['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=initial_settings['scheduler_factor'],
            #verbose=False
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _predict_time_series(self, light_curve, pred_times, pred_bands, sample, count):
        # Preprocess the light curve
        #light_curve

        grid_times = LightCurve().time_to_grid(pred_times, )
        grid_times = torch.FloatTensor(grid_times)[None, :].to(self.device)
        pred_bands = torch.LongTensor(pred_bands)[None, :].to(self.device)

        light_curves = [light_curve]

        # Run the model
        results = self.forward(light_curves, sample)

        #TODO: Add redshift predictions

        model_specta, model_flux = self.decode(
            results['encoding'],
            results['ref_times'],
            results['color'],
            grid_times,
            results['redshift'],
            pred_bands,
            results['amplitude']
        )

        model_flux = model_flux.detach().cpu().numpy()
        model_specta = model_specta.detach().cpu().numpy()

        if count is None:
            # Get rid of the batch index
            model_flux    = model_flux[0]
            model_spectra = model_spectra[0]

        cpu_results = {k:v.detach().cpu().numpy() for k,v in results.items()}

        # Return to orginal light curve scale
        model_flux   *= light_curve['flux_scale']
        model_specta *= light_curve['flux_scale']

        return model_flux, model_specta, cpu_results

    def predict_light_curve(self, light_curve, sample=False, count=None, sampling=1, pad=50.):
        """Predict a light curve

        Parameters
        ----------
        light_curve : pd.DataFrame
            Light curve to
        """
        min_time = np.min(light_curve['mjd']) - pad
        max_time = np.max(light_curve['mjd']) + pad
        model_times = np.arange(min_time, max_time + sampling, sampling)

        band_indices = np.arange(len(initial_settings['bands']))

        # np.tite repeats an array n_times
        # np.repeat repets an element in array n_times
        pred_times = np.tile(A = model_times, reps = len(band_indices))
        pred_bands = np.repeat(band_indices, len(model_times))

        model_flux, model_spectra, model_results = self._predict_time_series(
            light_curve, pred_times, pred_bands, sample, count
        )

        # Reshape the model_flux so that it has the shape (BATCH, BAND, TIMES)
        model_flux = model_flux.reshape((-1, len(initial_settings['bands']), len(model_times)))

        if count == 0:
            model_flux = model_flux[0]

        return model_flux, model_spectra, model_results


    def predict_spectrum(self, light_curve, time, sample=True, count=None):
        """Predict the spectrum of a light curve at a given time

        Parameters
        ----------
        light_curve : `~astropy.table.Table`
            Light curve
        time : float
            Time to predict the spectrum at
        sample : bool, optional
            If True, sample from the latent variable posteriors. Otherwise,
            use the MAP. By default False.
        count : int, optional
            Number of spectra to predict, by default None (single prediction)

        Returns
        -------
        `~numpy.ndarray`
            Predicted spectrum at the wavelengths specified by
            `~ParsnipModel.model_wave`
        """

        pred_times = np.array([time])
        pred_bands = np.array([0])

        model_flux, model_spectra, model_results = self._predict_time_series(
            light_curve, pred_times, pred_bands, sample, count
        )

        return model_spectra[...,0]

    def _smooth_spectra(self, model_spectra, method='moving_average', window_size=None, kernel_size=None, n_points=None):
        """
        Smooth a tensor (B, N, T) using the specified method,
        but only over non-zero values.
        """
        B, N, T = model_spectra.shape

        # Asegurar que window_size es impar
        if window_size is None:
            window_size = max(3, int(N / n_points))
            window_size = window_size if window_size % 2 != 0 else window_size + 1
        else:
            window_size = max(3, window_size)
            window_size = window_size if window_size % 2 != 0 else window_size + 1

        pad_size = (window_size - 1) // 2

        if method == 'moving_average':

            # Kernel shape for conv1d
            kernel = torch.ones(window_size, device=model_spectra.device) / window_size
            #kernel = torch.ones(kernel_size, device=model_spectra.device) / kernel_size
            kernel = kernel.view(1, 1, -1)  # [1, 1, window_size]

            # Reshape: [B, N, T] → [B*T, 1, N]
            tensor   = model_spectra.permute(0, 2, 1).reshape(B * T, 1, N)
            mask     = (tensor != 0).float()

            # Padding with reflect
            tensor_pad = F.pad(tensor, (pad_size, pad_size), mode='reflect')
            mask_pad   = F.pad(mask, (pad_size, pad_size), mode='reflect')

            if isinstance(n_points, int):
                stride = (N-kernel_size) // (n_points-1)
                stride = max(1, stride)
                print('stride', stride)
            else:
                stride = 1

            # Suavizar valores reales y normalizar por cantidad válida
            smoothed = F.conv1d(tensor_pad, kernel, padding=0, stride=stride)
            norm     = F.conv1d(mask_pad, kernel, padding=0, stride=stride)
            norm     = torch.clamp(norm, min=1e-6)  # evitar división por cero
            smoothed = smoothed / norm

            smoothed = smoothed.view(B, T, N).permute(0, 2, 1)
            smoothed[model_spectra == 0] = 0.0

            #smoothed = smoothed.view(B, T, -1).permute(0, 2, 1)

            return smoothed

        elif method == 'moving_median':
            mask = (model_spectra != 0)
            padded = F.pad(model_spectra, (pad_size, pad_size), mode='reflect')

            # Apply "unfold" to extract moving average
            unfolded = padded.unfold(dimension=2, size=window_size, step=1) # [B,N,T,window_size]
            unfolded_mask = F.pad(mask.float(), (pad_size, pad_size), mode='reflect')
            unfolded_mask = unfolded_mask.unfold(dimension=2, size=window_size, step=1)

            unfolded[unfolded_mask == 0] = float('nan')

            # Calcular mediana ignorando NaNs
            smoothed = torch.nanmedian(unfolded, dim=-1).values  # [B, N, T]
            smoothed[~mask] = 0.0
            return smoothed

        else:
            raise ValueError(f"Smoothing method is not available: {method}")

    def _astrodash_normalization(self, model_spectra, wave, n_points=13, apod_fraction=0.05, is_smoothed=False, return_continuum=False):
        """Normalize spectra using the AstroDASH methodology
        (Muthukrishna et al. 2019), adapted for VAE-generated model spectra
        (Spectra are in reference frame). The normalization process consists of:

            1. Continuum estimation via n-point cubic spline interpolation.
            2. Continuum division (spectral flattening).
            3. Edge apodization using a cosine bell function.
            4. Final normalization

        This implementation uses GPU-accelerated computations via
        torchcubicspline when available.

        Parameters
        ----------
        - model_spectra: '~torch.Tensor'
            Input spectra tensor from the VAE decoder with shape
            (batch_size, spectrum_bins, time_window).
        - n_points: int, optional
            Number of control points for cubic spline interpolation (default=13).
        - apod_fraction: float, optinal
            Fraction of spectrum edges to apodize (default=0.05 at each end).

        Returns
        -------
        - normalized_model_spectra: '~torch.Tensor'
            Normalized model spectra.
        """
        from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
        device = model_spectra.device
        batch_size, n_wave, n_time = model_spectra.shape

        wave = torch.FloatTensor(wave).to(device)
        # --- Paso 1: Crear máscara de datos válidos ---
        mask = model_spectra > 0 # Printo de mascara para verificar
        if not mask.any():
            return model_spectra

        # Determinar longitudes de onda con al menos un valor > 0
        valid_wave_mask = mask.any(dim=0).any(dim=1)  # [n_wave]
        valid_wave_indices = torch.where(valid_wave_mask)[0]

        if len(valid_wave_indices) < n_points:
            raise ValueError(f"Solo {len(valid_wave_indices)} longitudes de onda tienen flujo > 0 (se requieren {n_points})")

        if not is_smoothed:

            # Seleccionar puntos de control equiespaciados dentro del rango válido
            sp_idx = torch.linspace(0, len(valid_wave_indices) - 1, n_points).long()
            wave_npoints = wave[valid_wave_indices][sp_idx]  # [n_points]
            spectra_npoints = model_spectra[:, valid_wave_indices][:, sp_idx, :]  # [B, n_points, T]

            # --- Paso 2: Ajustar spline cúbico natural ---
            #coeffs = natural_cubic_spline_coeffs(wave_npoints.to(device), spectra_npoints)
            coeffs = natural_cubic_spline_coeffs(wave_npoints, spectra_npoints)
            spline = NaturalCubicSpline(coeffs)

            wave_valid = wave[valid_wave_indices].to(device)
            spline_values_valid = spline.evaluate(wave_valid)
            spline_values = torch.zeros_like(model_spectra)
            spline_values[:, valid_wave_indices, :] = spline_values_valid

        #spline_values = spline.evaluate(wave.to(device))  # [B, n_wave, T]

        else:
            spline_values = self._smooth_spectra(model_spectra, method='moving_average', window_size=200, kernel_size=None, n_points=None)

        # --- Paso 3: División por el continuo ---
        continuum_divided = torch.zeros_like(model_spectra)
        continuum_divided[mask] = model_spectra[mask] / spline_values[mask]
        # Restarle -1 
        
        continuum_divided = continuum_divided - 1.0

        # --- Paso 4: Apodización ---
        n_apod = max(1, int(n_wave * apod_fraction))
        apod_window = torch.ones(n_wave, device=device)

        x = torch.linspace(0, np.pi / 2, n_apod, device=device)
        apod_window[:n_apod] = torch.sin(x)**2
        apod_window[-n_apod:] = torch.flip(torch.sin(x), dims=[0])**2

        apodized_spectra = continuum_divided * apod_window[None, :, None]  # [B, n_wave, T]
        apodized_spectra = torch.nan_to_num(apodized_spectra, nan=0.0, posinf=0.0, neginf=0.0)

        # Eliminar la renormalizacion del último paso


        # --- Paso 5: Normalización final (opcional, comentado) ---
        max_spectrum = torch.max(apodized_spectra, dim=1, keepdim=True)[0]
        #max_spectrum = torch.quantile(apodized_spectra, 0.95, dim=1, keepdim=True)[0]
        final_spectra = torch.zeros_like(model_spectra)
        #final_spectra[mask] = (apodized_spectra / torch.clamp(max_spectrum, min=1e-8))[mask]
        final_spectra[mask] = apodized_spectra[mask]
        final_spectra = torch.nan_to_num(final_spectra, nan=0.0, posinf=0.0, neginf=0.0)

        if return_continuum:
            #Esto porque Spline values no esta normalizado
            max_continuum = torch.max(spline_values, dim=1, keepdim=True)[0]
            continuum_norm = torch.zeros_like(spline_values)
            continuum_norm[mask] = (spline_values / torch.clamp(max_continuum, min=1e-8))[mask]
            continuum_norm = torch.nan_to_num(continuum_norm, nan=0.0, posinf=0.0, neginf=0.0)
            #return final_spectra, continuum_norm, apodized_spectra, spline_values
            return final_spectra, continuum_divided, apodized_spectra, spline_values, model_spectra
        else:
            return final_spectra

if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser(description="Change the initial conditions of the model")
    #parser.add_argument("-sn", "--spectra_bins", default=initial_settings['spectrum_bins'], type=int, required=False, help="Number of bins for the spectra")
    #parser.add_argument("-sp", "--spectra_penalty", default=initial_settings['penalty_spectra'], type=float, required=False, help="Penalty for the spectra")
    #parser.add_argument("-ls", "--latent_space_dim", default=initial_settings['latent_size'], type=int, required=False, help="Latent space dimension")
    #args = parser.parse_args()

    #initial_settings['spectrum_bins'] = args.spectra_nbins
    #initial_settings['penalty_spectra'] = args.spectra_penalty
    #initial_settings['latent_size'] = args.latent_space_dim


    today = pd.Timestamp.today(tz='America/Santiago').strftime('%Y%m%d_%H%M')
    epochs = 1
    model = MPhy_VAE(
        batch_size=initial_settings['batch_size'],
        device=device,
        bands=initial_settings['bands']
        )
    wandb_logger = WandbLogger(
        entity='fforster-uchile',
        project='SupernovaeMultimodalVAE',
        job_type='train',
        name = (
            f"MPhy_VAE_{today}_nbins={initial_settings['spectrum_bins']}_"
            f"LatentSize={initial_settings['latent_size']}_"
            #f"LossSpectra_WeightNOnormalized_{initial_settings['penalty_spectra']}"
            f"LossSpectra_WeightNormalized_{initial_settings['penalty_spectra']}"
        ),
        #name=f"MPhy_VAE_Rainbow_bins_{initial_settings['spectrum_bins']}",
        #name=f"{today}_nbis={initial_settings['spectrum_bins']}_lossSpectra",
        #name=f"TEST_{today}",
        #name=f"TEST_{today}_presentContinuum_NLHPC",
        #name=f"BORRAR_TEST_{today}",
        #name=f"TEST_20250507_15:40",
        config={
            'epochs': epochs,
            'batch_size': initial_settings['batch_size'],
            'spectrum_bins': initial_settings['spectrum_bins'],
            'latent_size': initial_settings['latent_size'],
            'loss_spectra': 'NOnormalized',
            'learning_rate': initial_settings['learning_rate'],
            'scheduler_factor': initial_settings['scheduler_factor'],
        })
    #model = model.to(device)
    #timer = Timer()
    #metric = 
    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        #accelerator='auto',
        devices='auto',
        callbacks=[
            MetricsCallback(),
            Timer(),
        ],
        enable_checkpointing=False
        )
    trainer.fit(
        model= model,
        train_dataloaders = train_loader,
        val_dataloaders = test_loader,
        )

    #train_loss = model.train_loss_epoch
    #val_loss   = model.val_loss_epoch
    #results    = model.latest_results

    #print('='*30)
    #print('END TRAINING')
    #print('='*30)
    #print(f"Tiempo total de entrenamiento: {timer.time_elapsed('train'):.2f} segundos")

    wandb.finish()
    