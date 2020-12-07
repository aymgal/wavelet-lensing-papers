__author__ = 'aymgal'

import os
import numpy as np
import scipy
import astropy.io.fits as pf


def resize_psf_kernel(kernel, new_size, replace_negative=1e-15):
    """resize a PSF kernel image while making sure it is centered on a single pixel"""
    if new_size % 2 == 0:
        # make sure the new size is always an odd number
        new_size += 1
    dirac = np.zeros((new_size, new_size), dtype=kernel.dtype)
    dirac[new_size//2, new_size//2] = 1
    kernel_resized = scipy.signal.fftconvolve(dirac, kernel, mode='same')  # make sure it is centered on a pixel
    kernel_resized[kernel_resized < 0] = replace_negative
    return kernel_resized / kernel_resized.sum()  # normalize so sum is 1

def get_HST_psf_kernel():
    """using tiny time kernel"""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with pf.open(os.path.join(this_dir, 'data', 'PSFs', 'PSF_HST_F160W_tinytim.fits')) as f:
        psf_kernel = f[0].data
    # remove pixels near edges that display artifacts
    psf_kernel = psf_kernel[8:-8, 8:-8]
    # makes sure sum is 1
    psf_kernel /= psf_kernel.sum()
    return psf_kernel.astype(float)

def get_ELT_psf_kernel(size_kernel=159):
    """using SimCADO file to get E-ELT simulated PSF for the MICADO instrument (from MAORY simulation)"""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with pf.open(os.path.join(this_dir, 'data', 'PSFs', 'PSF_MCAO_H_Strehl21.fits')) as f:
        psf_kernel_maory_raw = f[0].data.astype(np.float32)
    psf_kernel_maory_odd = psf_kernel_maory_raw[1:, 1:]
    psf_kernel_maory_odd /= psf_kernel_maory_odd.sum()  # normalize so sum is 1
    # resize the PSF to smaller kernel
    if size_kernel < psf_kernel_maory_odd.shape[0]:
        psf_kernel = resize_psf_kernel(psf_kernel_maory_odd, size_kernel)
    return psf_kernel.astype(float)
