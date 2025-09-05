initial_settings = {
    'sideral_scale': 86400. / 86164.0905, # days per sidereal year

    # Initial settings
    #'bands': ['ps1::g', 'ps1::r'],
    'bands': None,
    'reference_time': None,

    # Lightcurve settings
    'time_window': 300,  # days
    'time_pad': 100,     # days
    'error_floor': 0.01,
    'time_sigma': 20.,    # days
    'color_sigma': 0.3,  # mag
    'magsys': 'ab',
    'zeropoint': 25.0,

    # Spectrum settings
    'min_wave': 3206.34,     # Angstrom
    'max_wave': 10915.01,    # Angstrom
    'spectrum_bins': 1838,   # This value is to keep the velocity equal to 200 Km/s
    'band_oversampling': 51, # This must be odd.
    'max_redshift': 1.,

    # Model Settings
    'batch_size': 64,
    'learning_rate': 1e-3,
    'penalty': 1e-3,
    'latent_size': 3,

    # Encoding settings. The settings here are not the same
    # that in Parsnip.
    'encode_conv_architecture': [40, 80, 120, 160, 200, 200, 200],
    'encode_conv_dilations': [1, 2, 4, 8, 16, 32, 64],
    'encode_fullyconnected_architecture': [200],
    'encode_time_architecture': [200],
    'encode_latent_prepool_architecture': [200],
    'encode_latent_postpool_architecture': [200],

    # Decode settings
    'decode_architecture': [40, 80, 160, 300],

}

band_info = {
    # Information about all of the different bands and how to handle them. We assume
    # that all data from the same telescope should be processed the same way.

    # Band name     Correct     Correct     Plot color      Plot marker ALeRCE
    #               Background  MWEBV                                    fid
    # PanSTARRS
    #'ps1::g':       (True,      True,       'C0',           'o',          1),
    #'ps1::r':       (True,      True,       'C2',           '^',          2),
    #'ps1::i':       (True,      True,       'C1',           'v',          3),
    #'ps1::z':       (True,      True,       'C3',           '<',          4),

    # PLAsTICC
    # 'lsstu':        (True,      False,      'C6',           'o',          1),
    # 'lsstg':        (True,      False,      'C4',           'v',          2),
    # 'lsstr':        (True,      False,      'C0',           '^',          3),
    # 'lssti':        (True,      False,      'C2',           '<',          4),
    # 'lsstz':        (True,      False,      'C3',           '>',          5),
    # 'lssty':        (True,      False,      'goldenrod',    's',          6),

    # ZTF
    'ztfg':         (False,     True,       'C0',           'o',          1),
    'ztfr':         (False,     True,       'C2',           '^',          2),
    'ztfi':         (False,     True,       'C1',           'v',          0),

    # SWIFT
    # 'uvot::u':      (False,     True,       'C6',           '<',          1),
    # 'uvot::b':      (False,     True,       'C3',           '>',          2),
    # 'uvot::v':      (False,     True,       'goldenrod',    's',          3),
    # 'uvot::uvm2':   (False,     True,       'C5',           'p',          4),
    # 'uvot::uvw1':   (False,     True,       'C7',           'P',          5),
    # 'uvot::uvw2':   (False,     True,       'C8',           '*',          6),
}