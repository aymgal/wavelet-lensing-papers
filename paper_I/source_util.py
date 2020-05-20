__author__ = 'aymgal'

# this files groups utility functions to extract and prepare source galaxy
# for the generation of simulated HST-like lensed images

import os
import numpy as np
import astropy.io.fits as pf


def prepare_highres_source(ra_source, dec_source, galaxy_name='NGC1309', magnitude=None, amp=1.2e3):
    source_dir = os.path.join('data', 'sources_TDLMC')
    source_filename = '{}_fix.fits'.format(galaxy_name)
     # load a source galaxy pre-generated with galsim
    with pf.open(os.path.join(source_dir, source_filename)) as f:
        source_image = f[0].data
    # remove negative pixels
    source_image[source_image < 0] = 0.
    # normalize so max is 1 (arbirary flux units)
    source_image /= source_image.max()
    # specify scale, centering and rotation angle for interpolation on source grid
    galaxy_scale = 1e-2   # effectively set the angular size
    # light profile in lenstronomy conventions
    source_model_list = ['INTERPOL']
    kwargs_interpol_source = {'image': source_image, 'scale': galaxy_scale,
                              'center_x': ra_source, 'center_y': dec_source, 'phi_G': 0}
    if magnitude is not None:
        kwargs_interpol_source['magnitude'] = magnitude
    else:
        kwargs_interpol_source['amp'] = amp
    kwargs_source = [kwargs_interpol_source]
    return source_model_list, kwargs_source
    

def prepare_highres_source_multiple(source_pos_list, galaxy_name_list, mag_list=None, amp_list=None):
    if mag_list is None:
        mag_list = [None]*len(source_pos_list)
    if amp_list is None:
        amp_list = [1.2e3]*len(source_pos_list)
    source_model_list, kwargs_source = [], []
    for (ra, dec), name, mag, amp in zip(source_pos_list, galaxy_name_list, mag_list, amp_list):
        source_model_list_s, kwargs_source_s = prepare_highres_source(ra, dec, galaxy_name=name, magnitude=mag, amp=amp)
        source_model_list += source_model_list_s
        kwargs_source += kwargs_source_s
    return source_model_list, kwargs_source
