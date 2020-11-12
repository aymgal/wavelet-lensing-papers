#!/usr/bin/env python
# coding: utf-8

# # Complex source reconstruction at fixed lens mass
# 
# __Author__ : Aymeric Galan
# 
# __Created__ : 23/04/20
# 
# __Updated__ : 31/08/20
# 
# This notebooks gives a working example of the modelling of a complex and realistic source with a pixel-based method based on sparsity and wavelets. The lens mass model is assumed to be exactly known, or at least a good approximation of it.
# 
# The source object, when modelled using the pixel-based method, is reconstructed iteratively on a grid of pixels, subject to constraints of __sparsity in starlet space__ and __positivity in direct space__.


# 2/11/20 : Modified form sampling the mass parameters through MCMC. 


__author__ = 'aymgal'

import os
import sys
import copy
import scipy
import pickle as pkl
import astropy.io.fits as pf
import numpy as np

if sys.platform[:5] == 'linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('image', interpolation='none')  # setup some defaults
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

import time
import corner
from tqdm import tqdm

from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.Numerics.grid import RegularGrid
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Analysis.image_reconstruction import MultiBandImageReconstruction
from lenstronomy.Plots import lens_plot, chain_plot
from lenstronomy.Util import mask_util
from lenstronomy.Util import image_util
from lenstronomy.Util import param_util
import lenstronomy.Util.util as lenstro_util

from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from slitronomy.Util.plot_util import nice_colorbar, nice_colorbar_residuals, log_cmap
from slitronomy.Util import metrics_util

from TDLMCpipeline.Util.plots import plot_convergence_by_walker

sys.path.insert(0, '..')
import fixed_param
import mass_util
import source_util
import psf_util


SEED = 18


#### FOR SCRIPT / CLUSTER ####
script_bool = False
if len(sys.argv) < 3:
    raise ValueError("Missing input parameters")
supersampling_factor_source = int(sys.argv[1])
print("Source supersampling factor :", supersampling_factor_source)
num_cores = int(sys.argv[2])
print("Number of cores :", num_cores)
thread_count_slit = 1  # runs faster on cluster when 1 here
thread_count_sampling = num_cores
#### #### #### #### #### ####


# MCMC settings
if supersampling_factor_source > 4:
    n_burn, n_run, walkerRatio = 800, 500, 10
else:
    n_burn, n_run, walkerRatio = 500, 500, 10
burnin_corner = 2000
source_grid_offset = True
shear_gamma1gamma2 = True
n_scales_source = 6

# general settings
data_type = 'HST'  # 'HST', 'ELT'
complex_source_type = 'highres-single'  # 'highres-single', 'highres-group'
perfect_seeing = False


# #### Choice of colormaps for image plots

cmap_default = plt.get_cmap('viridis')
cmap_residuals = plt.get_cmap('RdBu_r')
cmap_flux = copy.copy(plt.get_cmap('cubehelix'))
cmap_flux.set_bad(color='black', alpha=1)
cmap_flux_with_neg = copy.copy(plt.get_cmap('cubehelix'))
cmap_flux_with_neg.set_bad(color='#222222', alpha=1)  # emphasize negative pixels when in log scale


# ## Prepare for simulation
# 
# Several modules have been created:
# - `fixed_param`: intrumental and lens configuration parameters.
# - `mass_util`: functions for proper setup of lens model parameters.
# - `psf_util`: functions to load and prepare pixelated PSFs.
# - `source_util`: functions to load prepare realistic source objetcs.
# 
# ### Load PSFs and instrument properties


# seeing specifics
if perfect_seeing:
    psf_type = 'NONE'
    psf_kernel = None
else:
    psf_type = 'PIXEL'
    if data_type == 'HST':
        psf_kernel = psf_util.get_HST_psf_kernel()
    else:
        raise NotImplementedError

# observational, instrumentation and numerical settings
kwargs_simulation, kwargs_numerics_simulation = fixed_param.get_simulation_kwargs(data_type, psf_type, psf_kernel)

# number of side pixels in cutout
delta_pix = kwargs_simulation['pixel_scale']
num_pix = int(fixed_param.cutout_size / delta_pix)
num_pix_source = int(num_pix*supersampling_factor_source)  # (maximum) number of side pixels for source plane


# ### Setup lens model, source position and magnitude

# lens model properties
lens_model_list, kwargs_lens = mass_util.get_lens_model_macro(fixed_param.sigma_v)

if shear_gamma1gamma2:
    # convert the parametrization of the shear from to ellipticities
    lens_model_list[1] = 'SHEAR'
    psi_ext, gamma_ext = kwargs_lens[1].pop('psi_ext'), kwargs_lens[1].pop('gamma_ext')
    gamma1, gamma2 = param_util.shear_polar2cartesian(psi_ext, gamma_ext)
    kwargs_lens[1]['gamma1'], kwargs_lens[1]['gamma2'] = gamma1, gamma2

# main source galaxy properties
ra_source, dec_source = fixed_param.ra_source, fixed_param.dec_source
mag_source = fixed_param.mag_source


# ### Visualize lens model

lens_model_class_plot = LensModel(lens_model_list=lens_model_list)

fig, ax = plt.subplots(1, 1)
ax.set_title("lens model (* is source position)")
lens_plot.lens_model_plot(ax, lens_model_class_plot, kwargs_lens, 
                          numPix=60, deltaPix=0.08,   # play with these to zoom in/out in the figure, or increase resolution
                          sourcePos_x=ra_source, sourcePos_y=dec_source,
                          point_source=True, with_caustics=True, coord_inverse=False)
fig.savefig(os.path.join("mock_lens_model_ss{}.png".format(supersampling_factor_source)), bbox_inches='tight')
plt.close()


# ### Setup the source surface brightness


if complex_source_type == 'highres-single':
    source_model_list, kwargs_source = source_util.prepare_highres_source(ra_source, dec_source, 
                                                                          galaxy_name=fixed_param.galaxy_name, 
                                                                          magnitude=fixed_param.mag_source)
elif complex_source_type == 'highres-group':
    source_model_list, kwargs_source = source_util.prepare_highres_source_multiple(fixed_param.pos_source_list, 
                                                                                   fixed_param.galaxy_name_list,
                                                                                   fixed_param.mag_source_list)


# ### Define the functions that simluate image data

def simulate_image(lens_model_list, kwargs_lens, source_model_list, kwargs_source, noise_seed=None):
    # wrap up the model components
    kwargs_model_sim = {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        #'lens_light_model_list': lens_light_model_list,
    }
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    source_model_class = LightModel(light_model_list=source_model_list)
    
    # instantiate the simulation class
    sim_class = SimAPI(num_pix, kwargs_simulation, kwargs_model_sim)

    # get the image model class
    imsim_class = sim_class.image_model_class(kwargs_numerics=kwargs_numerics_simulation)

    # get other useful classes
    data_class = sim_class.data_class
    psf_class = sim_class.psf_class

    if 'magnitude' in kwargs_source[0]:
        # convert magnitudes into internal amplitude units based on instrumental settings
        #print(kwargs_source[0]['magnitude'])
        _, kwargs_source, _ = sim_class.magnitude2amplitude(kwargs_source_mag=kwargs_source)
        #print(kwargs_source[0]['amp'])
        
    # simulate the noise-free image
    image_sim_no_noise = imsim_class.image(kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None)

    # add realistic noise
    seed = SEED if noise_seed is None else noise_seed
    image_sim = image_sim_no_noise + sim_class.noise_for_model(model=image_sim_no_noise, seed=seed)

    # update the data class with the simulated image
    data_class.update_data(image_sim)

    # one can also get the source light at the data resolution
    source_sim_data_res = imsim_class.source_surface_brightness(kwargs_source, de_lensed=True, unconvolved=True)
    
    # extract coordinates properties for orientation and pixel to angle conversion
    ra_at_xy_0, dec_at_xy_0 = data_class.radec_at_xy_0
    transform_pix2angle = data_class.transform_pix2angle

    # from these properties, create grid objects for image plane and source plane
    image_grid = RegularGrid(num_pix, num_pix, transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
    source_grid = RegularGrid(num_pix, num_pix, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, 
                              supersampling_factor=supersampling_factor_source)

    # extract 1D coordinates arrays for easy evaluation surface brightness profiles
    ra_grid_hd, dec_grid_hd = source_grid.coordinates_evaluate

    source_sim = source_model_class.surface_brightness(ra_grid_hd, dec_grid_hd, kwargs_source)
    source_sim = lenstro_util.array2image(source_sim)

    # flux normalization of true source for proper comparison with pixel-based reconstruction
    source_sim_comp = source_sim * delta_pix**2
    
    class_list = (imsim_class, data_class, psf_class, lens_model_class, source_model_class)
    return class_list, kwargs_source, source_sim_comp, source_sim_data_res
                         


# ### Simulate imaging data

# In[9]:


class_list, kwargs_source, source_sim_comp, source_sim_data_res = simulate_image(lens_model_list, kwargs_lens, source_model_list, kwargs_source)
imsim_class, data_class, psf_class, _, _ = class_list
image_sim = data_class.data


# ### Visualise products

fig = plt.figure(figsize=(18, 4))

ax = plt.subplot(1, 5, 1)
ax.set_title("image, convolved")
ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
im = ax.imshow(data_class.data, origin='lower', cmap=cmap_flux, norm=LogNorm(1e-2, 1e1))
nice_colorbar(im)

ax = plt.subplot(1, 5, 2)
ax.set_title("source, convolved")
ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
source_sim_data_res_conv = imsim_class.source_surface_brightness(kwargs_source, de_lensed=True, unconvolved=False)
im = ax.imshow(source_sim_data_res_conv, origin='lower', cmap=cmap_flux, norm=LogNorm(1e-2, 1e1))
nice_colorbar(im)

ax = plt.subplot(1, 5, 3)
ax.set_title("source, unconvolved, high-res")
ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
im = ax.imshow(source_sim_comp, origin='lower', cmap=cmap_flux, norm=LogNorm(1e-2, 1e1))
nice_colorbar(im)

ax = plt.subplot(1, 5, 4)
ax.set_title("noise map")
ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
im = ax.imshow(np.sqrt(data_class.C_D), origin='lower', cmap=cmap_default)
nice_colorbar(im)

ax = plt.subplot(1, 5, 5)
ax.set_title("PSF kernel")
ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
im = ax.imshow(psf_class.kernel_point_source, origin='lower', 
               cmap=cmap_default, norm=LogNorm(1e-8))
nice_colorbar(im)

fig.savefig(os.path.join("mock_data_ss{}.png".format(supersampling_factor_source)), bbox_inches='tight')
plt.close()


# ## Reconstruction of the source galaxy
# 
# We perform the reconstruction using two methods:
# - "pixel-based" using the new `'SLIT_STARLETS'` profile
# - "analytical" using the `'SHAPELETS'` and optionally `'SERSIC_ELLIPSE'` profiles

# ### Optionally setup a mask for exclude pixels from the likelihood


# extract 1D and 2D coordinates arrays for easy evaluation surface brightness profiles
ra_grid, dec_grid = imsim_class.ImageNumerics.coordinates_evaluate
ra_grid_2d, dec_grid_2d = lenstro_util.array2image(ra_grid), lenstro_util.array2image(dec_grid)

# you can set any mask on imaging data here (binary array, 0s are masked pixels)
if data_type == 'HST':
    image_mask = np.ones((num_pix, num_pix))
else:
    raise NotImplementedError


# #### Results will be saved in these dictionnaries

all_results = {}
all_timings = {}


# ### Pixel-based starlets reconstruction
# 
# #### Define a function that runs the pixel-based reconstruction

def corner_add_values_indic(fig, values, color='green', linewidth=1):
    # Extract the axes
    ndim = len(values)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        if values[i] is not None:
            ax.axvline(values[i], color=color, linewidth=linewidth)
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            if values[xi] is not None:
                ax.axvline(values[xi], color=color, linewidth=linewidth)
            if values[yi] is not None:
                ax.axhline(values[yi], color=color, linewidth=linewidth)
            if values[xi] is not None and values[yi] is not None:
                ax.plot(values[xi], values[yi], color=color, marker='s')


def run_pixelbased_sampling(class_list, kwargs_lens, ss_factor_source, 
                             num_starlet_scales=6, show_results=True, verbose=True, 
                             noise_seed=None):

    _, data_class, psf_class, lens_model_class, _ = class_list
    
    # setup sparse modelling
    source_model_list_sparsefit = ['SLIT_STARLETS']
    source_model_class_sparsefit = LightModel(light_model_list=source_model_list_sparsefit)

    kwargs_model = {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list_sparsefit,
    }
    
    # we fixed the mass to truth
    kwargs_lens_init = kwargs_lens
    if shear_gamma1gamma2:
        kwargs_lens_sigma = [{'theta_E': 0.01, 'center_x': 0.01, 'center_y': 0.01, 'e1': 0.01, 'e2': 0.01}, {'gamma1': 0.01, 'gamma2': 0.01}]
        kwargs_lens_lower = [{'theta_E': kwargs_lens[0]['theta_E']-0.1, 'center_x': kwargs_lens[0]['center_x']-0.1, 'center_y': kwargs_lens[0]['center_y']-0.1, 'e1': kwargs_lens[0]['e1']-0.1, 'e2': kwargs_lens[0]['e2']-0.1}, {'gamma1': kwargs_lens[1]['gamma1']-0.1, 'gamma2': kwargs_lens[1]['gamma2']-0.1}]
        kwargs_lens_upper = [{'theta_E': kwargs_lens[0]['theta_E']+0.1, 'center_x': kwargs_lens[0]['center_x']+0.1, 'center_y': kwargs_lens[0]['center_y']+0.1, 'e1': kwargs_lens[0]['e1']+0.1, 'e2': kwargs_lens[0]['e2']+0.1}, {'gamma1': kwargs_lens[1]['gamma1']+0.1, 'gamma2': kwargs_lens[1]['gamma2']+0.1}]
    else:
        kwargs_lens_sigma = [{'theta_E': 0.01, 'center_x': 0.01, 'center_y': 0.01, 'e1': 0.01, 'e2': 0.01}, {'gamma_ext': 0.01, 'psi_ext': 0.01}]
        kwargs_lens_lower = [{'theta_E': kwargs_lens[0]['theta_E']-0.1, 'center_x': kwargs_lens[0]['center_x']-0.1, 'center_y': kwargs_lens[0]['center_y']-0.1, 'e1': kwargs_lens[0]['e1']-0.1, 'e2': kwargs_lens[0]['e2']-0.1}, {'gamma_ext': kwargs_lens[1]['gamma_ext']-0.02, 'psi_ext': kwargs_lens[1]['psi_ext']-0.1}]
        kwargs_lens_upper = [{'theta_E': kwargs_lens[0]['theta_E']+0.1, 'center_x': kwargs_lens[0]['center_x']+0.1, 'center_y': kwargs_lens[0]['center_y']+0.1, 'e1': kwargs_lens[0]['e1']+0.1, 'e2': kwargs_lens[0]['e2']+0.1}, {'gamma_ext': kwargs_lens[1]['gamma_ext']+0.02, 'psi_ext': kwargs_lens[1]['psi_ext']+0.1}]
    kwargs_lens_fixed = [{}, {'ra_0': 0, 'dec_0': 0}]

    kwargs_source_init  = [{}]
    kwargs_source_sigma = [{}]
    kwargs_source_lower = [{}]
    kwargs_source_upper = [{}]
    kwargs_source_fixed = [
        {
            'n_scales': num_starlet_scales, 
            'n_pixels': data_class.data.size*ss_factor_source**2,
            'scale': 1, 'center_x': 0, 'center_y': 0,
        }
    ]

    if source_grid_offset:
        offset_bound = delta_pix/2.
        kwargs_special_init = {'delta_x_source_grid': 0, 'delta_y_source_grid': 0}
        kwargs_special_sigma = {'delta_x_source_grid': 0.01, 'delta_y_source_grid': 0.01}
        kwargs_special_lower = {'delta_x_source_grid': -offset_bound, 'delta_y_source_grid': -offset_bound}
        kwargs_special_upper = {'delta_x_source_grid': offset_bound, 'delta_y_source_grid': offset_bound}
        kwargs_special_fixed = {}
    else:
        kwargs_special_init = {}
        kwargs_special_sigma = {}
        kwargs_special_lower = {}
        kwargs_special_upper = {}
        kwargs_special_fixed = {}
    print("kwargs_special lower & upper", kwargs_special_lower, kwargs_special_upper)

    kwargs_params = {
        'lens_model': [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lens_lower, kwargs_lens_upper],
        'source_model': [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_source_lower, kwargs_source_upper],
        'special': [kwargs_special_init, kwargs_special_sigma, kwargs_special_fixed, kwargs_special_lower, kwargs_special_upper],
    }
    
    kwargs_sparse_solver = {
        'supersampling_factor_source': ss_factor_source,
        'minimal_source_plane': True,
        'min_threshold': 3,
        'threshold_increment_high_freq': 1,
        'threshold_decrease_type': 'exponential',
        'num_iter_source': 20,
        'num_iter_weights': 2,  # 3
        'verbose': verbose,
        'thread_count': thread_count_slit,  # number of processors
    }

    ra_at_xy_0, dec_at_xy_0 = data_class.radec_at_xy_0
    transform_pix2angle = data_class.transform_pix2angle
    kwargs_data_fit = {
        'image_data': data_class.data,
        'background_rms': data_class.background_rms,
        'exposure_time': data_class.exposure_map,
        'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
        'transform_pix2angle': transform_pix2angle,
    }
    kwargs_psf_fit = {
        'psf_type': psf_type,
        'kernel_point_source': psf_kernel,
    }

    kwargs_numerics_sparsefit = {'supersampling_factor': 1}
    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data_fit, kwargs_psf_fit, kwargs_numerics_sparsefit]],
        'multi_band_type': 'single-band',
    }
    kwargs_constraints = {
        'solver_type': 'NONE',
        'source_grid_offset': source_grid_offset,
    }
    kwargs_likelihood = {
        'image_position_likelihood': True,
        'image_likelihood_mask_list': [image_mask],
        'check_bounds': True,
        'kwargs_pixelbased': kwargs_sparse_solver,
    }

    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, 
                                  kwargs_likelihood, kwargs_params, verbose=False, mpi=False)

    param_class = fitting_seq.param_class
    param_class.print_setting()

    fitting_list = [
        ['MCMC', {
            'n_burn': n_burn, 
            'n_run': n_run,
            'walkerRatio': walkerRatio, 
            'sigma_scale': 1, 
            'threadCount': thread_count_sampling,
            'sampler_type': 'EMCEE',
        }],
    ]

    # launch solver
    start_time = time.time()
    chain_list_sparsefit = fitting_seq.fit_sequence(fitting_list)
    end_time = time.time()
    timing = end_time-start_time
    if verbose:
        print("Runtime : {:.3f} s".format(timing))
    
    # get best parameters from max logL
    kwargs_result_sparsefit = fitting_seq.best_fit()

    bestfit_values = param_class.kwargs2args(kwargs_lens=kwargs_result_sparsefit['kwargs_lens'], 
                                             kwargs_source=[{}],
                                             kwargs_special=kwargs_result_sparsefit['kwargs_special'])
    truth_values = param_class.kwargs2args(kwargs_lens=kwargs_lens, 
                                           kwargs_source=[{}],
                                           kwargs_special=kwargs_special_init)

    _, mcmc_samples, param_names, logL_chain = chain_list_sparsefit[-1]
    num_samples, num_params_nonlinear = mcmc_samples.shape
    print("(num samples, num params) :", mcmc_samples.shape)

    fig = plt.figure()
    plt.plot(logL_chain)
    fig.savefig(os.path.join("mcmc_logL_ss{}.png".format(supersampling_factor_source)), bbox_inches='tight')
    plt.close()

    for i in range(len(chain_list_sparsefit)):
        fig, _ = chain_plot.plot_chain_list(chain_list_sparsefit, i, 
                                            num_average=walkerRatio*num_params_nonlinear)
    fig.savefig(os.path.join("mcmc_convergence_simple_ss{}.png".format(supersampling_factor_source)), bbox_inches='tight')
    plt.close()

    # convergence by walkers
    [fig] = plot_convergence_by_walker(mcmc_samples, param_names, walkerRatio, verbose=True)
    fig.savefig(os.path.join("mcmc_convergence_walkers_ss{}.png".format(supersampling_factor_source)), bbox_inches='tight')
    plt.close()

    # posterior distributions
    if mcmc_samples.shape[0] > burnin_corner:
        mcmc_samples_burn = mcmc_samples[-burnin_corner:, :]
        print("Samples after burnin (for corner plot)", burnin_corner)
    else:
        mcmc_samples_burn = mcmc_samples
    fig = corner.corner(mcmc_samples_burn, labels=param_names, show_titles=True, 
                        quantiles=[0.5], smooth=0.8, smooth1d=0.8)
    corner_add_values_indic(fig, truth_values, color='green', linewidth=3)
    corner_add_values_indic(fig, bestfit_values, color='red', linewidth=1)
    fig.savefig(os.path.join("mcmc_posteriors_ss{}.png".format(supersampling_factor_source)), bbox_inches='tight')
    plt.close()

    print("Best fit parameters :", kwargs_result_sparsefit)


    # launch solver to get source and image models from best-fit parameters
    imageFit = ImageLinearFit(data_class, psf_class=psf_class, 
                                lens_model_class=lens_model_class,
                                source_model_class=source_model_class_sparsefit,
                                likelihood_mask=image_mask,
                                kwargs_numerics=kwargs_numerics_sparsefit,
                                kwargs_pixelbased=kwargs_sparse_solver)
    kwargs_result_sparsefit_copy = copy.deepcopy(kwargs_result_sparsefit)
    solver_results = imageFit.image_linear_solve(kwargs_lens=kwargs_result_sparsefit_copy['kwargs_lens'], 
                                                 kwargs_source=kwargs_result_sparsefit_copy['kwargs_source'],
                                                 kwargs_special=kwargs_result_sparsefit_copy['kwargs_special'])
    sparseSolver = imageFit.PixelSolver
    fig = sparseSolver.plot_results(log_scale=True, fontsize=16,
                                    vmin_res=-6, vmax_res=6,
                                    cmap_image=cmap_flux, 
                                    cmap_source=cmap_flux_with_neg,
                                    vmin_image=1e-2, vmax_image=1e1, 
                                    vmin_source=1e-2, vmax_source=1e1)
    fig.savefig(os.path.join("sparse_solver_result_ss{}.png".format(supersampling_factor_source)), bbox_inches='tight')
    plt.close()

    # get model maps
    image_model = sparseSolver.image_model(unconvolved=False)
    if kwargs_sparse_solver['minimal_source_plane'] is True:
        ra_at_xy_0, dec_at_xy_0 = data_class.radec_at_xy_0
        source_grid = RegularGrid(num_pix, num_pix, data_class.transform_pix2angle, 
                                  ra_at_xy_0, dec_at_xy_0, 
                                  supersampling_factor=ss_factor_source)
        ra_grid_hd, dec_grid_hd = source_grid.coordinates_evaluate
        source_model = source_model_class_sparsefit.surface_brightness(ra_grid_hd, dec_grid_hd, 
                                                                       kwargs_result_sparsefit_copy['kwargs_source'])
        source_model = lenstro_util.array2image(source_model)
    else:
        source_model = sparseSolver.source_model

    return kwargs_result_sparsefit, chain_list_sparsefit, data_class.data, image_model, source_model, timing



# #### Runs the reconstruction or load it from backup file


model_type = 'starlets'
backup_dir = 'backups_ss{}'.format(supersampling_factor_source)
if not os.path.exists(backup_dir):
    os.mkdir(backup_dir)
path_backup = os.path.join(backup_dir, 'backup-mass-sampling_data-{}_mocksource-{}_model-{}_ssres-{}.pkl'
                           .format(data_type, complex_source_type, model_type, supersampling_factor_source))
    
# run the computation
data_backup = run_pixelbased_sampling(class_list, kwargs_lens, supersampling_factor_source,
                                      num_starlet_scales=n_scales_source, verbose=False)
# pickle for backup
with open(path_backup, 'wb') as handle:
    pkl.dump(data_backup, handle)

print("=== Success. ===")

