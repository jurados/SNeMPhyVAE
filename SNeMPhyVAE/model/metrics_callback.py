# ============= IMPORT LIBRARIES =============

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import astronomical libraries
import sncosmo

# Import PyTorch Lightning and logging libraries
from lightning.pytorch.callbacks import Timer, Callback

import wandb

# Import custom settings
from settings import initial_settings

# =============================================

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        
        self.losses = {
            'nll': [],
            'kld': [],
            'smooth': [],
            'amp_prob': [],
            'direct_spectra': [],
        }        

        self.val_loss_epoch = []

    def on_validation_epoch_end(self, trainer, L_module):
        
        loss = trainer.callback_metrics.get('val_loss')
        if loss is not None:
            self.val_loss_epoch.append([trainer.current_epoch, loss.item])
        optimizer  = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr'] 

        if hasattr(L_module, 'latest_results') and hasattr(L_module, 'latest_spectra'):
            print('='*30)
            print('SENDING TO WandB FROM VAL')
            print('='*30)

            nrep = 5

            # Light curve variables
            print('self.latest_results_model_flux', L_module.latest_results['model_flux'][:nrep].shape)
            model_times  = L_module.latest_results['time'][:nrep].detach().cpu().numpy()
            model_flux   = L_module.latest_results['model_flux'][:nrep].detach().cpu().numpy()
            band_indices = L_module.latest_results['band_indices'][:nrep].detach().cpu().numpy()

            print('L_module.latest_lightcures', len(L_module.latest_lightcurves[:nrep]))
            #data_time    = L_module.latest_lightcurves[:nrep]['mjd']
            #data_flux    = L_module.latest_lightcurves[:nrep]['flux']
            #data_fluxerr = L_module.latest_lightcurves[:nrep]['fluxerr']
            #data_band    = L_module.latest_lightcurves[:nrep]['fid']
            #oid          = L_module.latest_lightcurves[:nrep]['oid']
            #ref_time     = L_module.latest_lightcurves[:nrep]['reference_time'].mean()
            #flux_scale   = L_module.latest_lightcurves[:nrep]['flux_scale'].mean()
            data_time    = [lc['mjd'] for lc in L_module.latest_lightcurves[:nrep]]
            data_flux    = [lc['flux'] for lc in L_module.latest_lightcurves[:nrep]]
            data_fluxerr = [lc['fluxerr'] for lc in L_module.latest_lightcurves[:nrep]]
            data_band    = [lc['fid'] for lc in L_module.latest_lightcurves[:nrep]]
            oid          = [lc['oid'] for lc in L_module.latest_lightcurves[:nrep]]
            ref_time     = [lc['reference_time'].mean() for lc in L_module.latest_lightcurves[:nrep]]
            flux_scale   = [lc['flux_scale'].mean() for lc in L_module.latest_lightcurves[:nrep]]

            # Spectum variables
            model_wave    = L_module.latest_results['model_wave']
            print('L_module.latest_results_model_spectra', L_module.latest_results['model_spectra'][:nrep].shape)
            model_spectra = L_module.latest_results['model_spectra'][:nrep].detach().cpu().numpy()

            try:
                model_spectra_continuum = L_module.latest_results['model_spectra_continuum'][:nrep].detach().cpu().numpy()
                model_spectra_apod      = L_module.latest_results['apod_spectra'][:nrep].detach().cpu().numpy()
                model_spectra_spline    = L_module.latest_results['spline_spectra'][:nrep].detach().cpu().numpy()
                model_spectra_initial   = L_module.latest_results['initial_spectra'][:nrep].detach().cpu().numpy()
            except:
                pass
            #data_spectra  = L_module.latest_spectra[:nrep]['flux']
            #data_tidx     = L_module.latest_spectra[:nrep]['time_index']
            data_spectra = [sc['flux'] for sc in L_module.latest_spectra[:nrep]]
            data_tidx    = [sc['time_index'].values[0] for sc in L_module.latest_spectra[:nrep]]

            #model_spectra_with = L_module.latest_results['model_spectra_with'][:nrep].detach().cpu().numpy()
            #model_rainbow      = L_module.latest_results['rainbow'][:nrep].detach().cpu().numpy()
            #print('data spectra shape', data_spectra[:nrep].shape)

            # Latent space variables
            model_ref_time = L_module.latest_results['ref_times'][:nrep].detach().cpu().numpy()
            model_color    = L_module.latest_results['color'][:nrep].detach().cpu().numpy()
            model_encoding = L_module.latest_results['encoding'][:nrep].detach().cpu().numpy()

            #print('model_encoding', model_encoding.shape)
            true_label = L_module.latest_results['true_label'][:nrep]
            #print('TRUE_LABEL', len(true_label), type(true_label))
            oid        = L_module.latest_results['oid'][:nrep]

            #pca = PCA(n_components=2)
            #latent_2d = pca.fit_transform(model_encoding)

            #print(data_tidx)
            #data_tidx = data_tidx.unique()[0]

            if trainer.current_epoch % 5 == 0:
                #print('Sending Light Curve')
                # ========================================
                # Light Curves
                # ========================================
                fig, axes = plt.subplots(nrows=1, ncols = nrep,figsize=(15, 5), sharey=False, gridspec_kw={'wspace':0})
                for i, ax in enumerate(axes):
                    for band_idx, band_name in enumerate(initial_settings['bands']):
                        mask = band_indices[i] == band_idx
                        color = 'green' if band_name == 'ztfg' else 'red'
                        tindx = np.argsort(model_times[i][mask])
                        ax.fill_between(model_times[i][mask][tindx],
                                        model_flux[i][mask][tindx]*flux_scale[i] * 0.95,
                                        model_flux[i][mask][tindx]*flux_scale[i] * 1.05,
                                        color = color,
                                        alpha=0.2)
                        ax.plot(model_times[i][mask][tindx],
                                model_flux[i][mask][tindx]*flux_scale[i],
                                '-',
                                color = color,
                                #label=f'Model {band_name}'
                                )

                    for fid in np.unique(data_band[i]):
                        mask = data_band[i] == fid
                        color = 'green' if fid == 1 else 'red'
                        ax.errorbar(
                            data_time[i][mask]-ref_time[i],
                            data_flux[i][mask],
                            data_fluxerr[i][mask],
                            color = color,
                            fmt='o',
                            label=f'Original {"g" if fid == 1 else "r"}'
                        )

                    ax.set_title(f'{oid[i]}')
                    ax.set_xlabel('Time', fontsize=10)
                    if i == 0:
                        ax.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=14)
                    ax.legend(fontsize=8, framealpha=0.95)

                # Sent to W&B
                fig.suptitle(f'Light Curves Prediction (Epoch {trainer.current_epoch})')
                trainer.logger.experiment.log({
                    "Light Curve": wandb.Image(fig),
                    "epoch": trainer.current_epoch,
                })
                #plt.show()
                plt.close()

                # ========================================
                # Spectra
                # ========================================

                mask_wave = (model_wave > initial_settings['lambda_min_mask']) & (model_wave < initial_settings['lambda_max_mask'])

                fig, axes = plt.subplots(nrows=2, ncols = nrep, figsize=(15, 5), sharex=True,
                                       gridspec_kw={'hspace' : 0, 'wspace': 0})
                
                # Create legend handles only once
                legend_handles = []
                legend_created = False
                
                for i, (ax_top, ax_bottom) in enumerate(zip(axes[0], axes[1])):
                    if i == len(axes[1])-1:
                        line_model = ax_top.plot(model_wave, model_spectra[i][:, 0],'-', color='C0', label='Model Spectra')
                        line_original = ax_top.plot(model_wave, data_spectra[i][0], ls='--', color='black', label='Original')
                    ax_top.plot(model_wave, model_spectra[i][:, 0],'-', color='C0')
                    ax_top.plot(model_wave, data_spectra[i][0], ls='--', color='black')
                    ax_top.set_title(f'{oid[i]}')
                    if i == 0:
                        ax_top.set_ylabel('Normalized Flux', fontsize=14)
                    try:
                        ax_top.plot(model_wave, model_spectra_apod[:, data_tidx],'-', alpha=0.5, label='Model Spectra Apodization')
                        ax_top.plot(model_wave, model_spectra_continuum[:, data_tidx],'-', alpha=0.5, label='Model Spectra Continuum')
                        line_splines = ax_top.plot(model_wave, model_spectra_spline[i][:, data_tidx],'-',
                                    color='C1', alpha=0.5,
                                    )
                        line_initial = ax_top.plot(model_wave, model_spectra_initial[i][:, data_tidx],'-',
                                    color='C2', alpha=0.5,
                                    )
                        # Only create legend handles once
                        pass
                        if not legend_created:
                            legend_handles.append((line_model[0], 'Model Spectra'))
                            legend_handles.append((line_original[0], 'Original'))
                            legend_handles.append((line_splines[0], 'Model Spectra Spline'))
                            legend_handles.append((line_initial[0], 'Model Spectra Initial'))
                            legend_created = True
                    except:
                        pass
                    try:
                        ax_bottom.plot(model_wave, model_spectra_initial[i][:, data_tidx], '-', color='C2', alpha=0.7)
                        ax_bottom.plot(model_wave, model_spectra_spline[i][:, data_tidx], '-', color='C1', alpha=0.7)
                        if i == 0:
                            ax_bottom.set_ylabel('Normalized Flux', fontsize=14)
                        ax_bottom.set_xlabel('Wavelength (Å)', fontsize=14)

                    except:
                        pass
                for idx_ax in axes.flatten():
                    idx_ax.axvline(initial_settings['lambda_min_mask'], color='gray', ls=':')
                    idx_ax.axvline(initial_settings['lambda_max_mask'], color='gray', ls=':')

                # Only create legend if we have handles
                if legend_handles:
                    lines, labels = zip(*legend_handles)
                    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.05,0.5))#, fontsize=12)
                
                fig.suptitle(f'Spectra Prediction (Epoch {trainer.current_epoch})')
                # Sent to W&B
                trainer.logger.experiment.log({
                    "Spectra All": wandb.Image(fig),
                    "epoch": trainer.current_epoch,
                })
                #plt.show()
                plt.close()

                legend_handles = []
                legend_created = False
                wave_grid = np.logspace(
                    np.log10(initial_settings['min_wave']),
                    np.log10(initial_settings['max_wave']),
                    300)
                trans = 0
                for band_name in initial_settings['bands']:
                    band = sncosmo.get_bandpass(band_name)
                    band_transmission = band(wave_grid)
                    trans += band_transmission
                model_wave = model_wave[mask_wave]
                fig, axes = plt.subplots(nrows=1, ncols = nrep, figsize=(17, 5), sharey=False, gridspec_kw={'wspace':0})
                for i, ax in enumerate(axes):
                    model_spectrum = model_spectra[i][mask_wave, data_tidx[i]]
                    try:
                        model_spectrum_continuum = model_spectra_continuum[i][mask_wave, data_tidx[i]]
                    except:
                        pass

                    data_spectrum  = data_spectra[i][0][mask_wave]
                
                    #ax.plot(model_wave, model_spectra[:, data_tidx],'-', label='Model Spectra')
                    line_spectra = ax.plot(model_wave, model_spectrum,'-')
                    #try:
                    #    ax.plot(model_wave, model_spectrum_continuum,'-', label='Model Spectra Continuum')
                    #except:
                    #    pass
                    #ax.plot(model_wave, data_spectra[0], ls='--', label='Original')
                    line_original = ax.plot(model_wave, data_spectrum, ls='--')
                    #ax.plot(model_wave, model_spectra_with[:, 0], label='Model Spectra')
                    #ax.plot(model_wave, model_rainbow[:, 0], label='Rainbow Spectra')

                    ax.axvline(initial_settings['lambda_min_mask'], color='black', ls=':')
                    ax.axvline(initial_settings['lambda_max_mask'], color='black', ls=':')
                    trans_patch = ax.fill_between(wave_grid, trans, color='C2', alpha=0.2)
                    ax.set_title(f'{oid[i]}')
                    if not legend_created:                                      
                        legend_handles.append((line_spectra[0], 'Model Spectra'))
                        legend_handles.append((line_original[0], 'Original'))
                        legend_handles.append((trans_patch, 'Transmision'))
                        legend_created = True
                    if i == 0:
                        ax.set_ylabel('Normalized Flux', fontsize=14)
                        
                    ax.set_xlabel('Wavelength (Å)', fontsize=14)
                    # Sent to W&B
                if legend_handles:                                          
                    lines, labels = zip(*legend_handles)
                    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1,0.5))
                
                fig.suptitle(f'Spectra Prediction (Epoch {trainer.current_epoch})')
                trainer.logger.experiment.log({
                    "Spectra": wandb.Image(fig),
                    "epoch": trainer.current_epoch,
                })
                #plt.show()
                plt.close()

                #plt.figure(figsize=(10, 6))
                #scatter = plt.scatter(
                #    latent_2d[:, 0],
                #    latent_2d[:, 1],
                #    #c=color_values,
                #    cmap='viridis')
                #plt.colorbar(scatter)#, labels='Clusters')
                #plt.title(f'Clustering Latent Space (Epoch {self.current_epoch})')
                #plt.xlabel('Principal Component 1')
                #plt.ylabel('Principal Component 2')
                #self.logger.experiment.log({
                #    "Clustering Latent Space": wandb.Image(plt),
                #    "epoch": self.current_epoch,
                #})
                #plt.show()
                #plt.close()

                # ========================================
                # Latent Spaces
                # ========================================

                unique_labels = np.unique(true_label)
                label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_num[label] for label in true_label])
                cmap = plt.cm.get_cmap('viridis', len(unique_labels))

                fig, (ax12, ax23) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), gridspec_kw={'wspace': 0.4})
                fig.suptitle(f'Clustering Latent Space (Epoch {trainer.current_epoch})')

                sc1 = ax12.scatter(
                    model_encoding[:, 0],
                    model_encoding[:, 1],
                    c=numeric_labels,
                    cmap=cmap,
                    #label=true_label
                )
                #plt.colorbar(scatter)#, labels='Clusters')
                ax12.set_xlabel('s1')
                ax12.set_ylabel('s2')
                ax12.legend()

                sc2 = ax23.scatter(
                    model_encoding[:, 1],
                    model_encoding[:, 2],
                    c=numeric_labels,
                    cmap=cmap,
                    #label=true_label
                )
                #plt.colorbar(scatter)#, labels='Clusters')
                ax23.set_xlabel('s2')
                ax23.set_ylabel('s3')
                #ax23.legend()

                handles = [plt.Line2D([], [], marker='o', color='w', markersize=10,
                            markerfacecolor=cmap(i), label=label)
                            for i, label in enumerate(unique_labels)]

                # Añadir leyenda solo una vez (al primer subplot)
                ax12.legend(handles=handles,
                        #title='Tipo de Supernova',
                        #bbox_to_anchor=(1.05, 1),
                        loc='upper left')
                ax23.legend(handles=handles,
                        #title='Tipo de Supernova',
                        #bbox_to_anchor=(1.05, 1),
                        loc='upper left')


                trainer.logger.experiment.log({
                    "Clustering Latent Space": wandb.Image(plt),
                    "epoch": trainer.current_epoch,
                })
                #plt.show()
                plt.close()

                # ========================================
                # Pairplot
                # ========================================
                df_latent = pd.DataFrame({
                    'ref_time': model_ref_time,
                    'color': model_color,
                    's1': model_encoding[:, 0],
                    's2': model_encoding[:, 1],
                    's3': model_encoding[:, 2],
                })#, index=index) # Pass index here
                sns.pairplot(df_latent)
                # Sent to W&B
                trainer.logger.experiment.log({
                    "Latent Space": wandb.Image(plt),
                    "epoch": trainer.current_epoch,
                })
                #plt.show(False)
                plt.close()
            
        

        trainer.logger.experiment.log({
            'learning_rate': current_lr, 
            'epoch': trainer.current_epoch,
        })

        # Log the losses
        for key in self.losses:
            metric = trainer.callback_metrics.get(f'loss/{key}')
            if key == 'amp_prob':
               self.losses[key].append(abs(metric.item()))
            else:
                self.losses[key].append(metric.item())

        if trainer.current_epoch % 5 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            for key, values in self.losses.items():
                ax.plot(values, label=key, alpha=0.7)
            ax.set_title('Loss Component Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Value')
            ax.set_yscale('log')
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium')
            plt.tight_layout()
            trainer.logger.experiment.log({
                'loss_curves': wandb.Image(fig),
                'epoch': trainer.current_epoch,
            })
            plt.close(fig)
        
