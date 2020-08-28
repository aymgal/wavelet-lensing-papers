__author__ = 'aymgal'

# this files groups utility functions to define correct parameters for a substructure depending on the mass and redshift of the lens, given a cosmology

import copy
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import fixed_param


default_cosmo = FlatLambdaCDM(H0=fixed_param.H0, Om0=fixed_param.Om0, Ob0=0.0)
lens_cosmo = LensCosmo(fixed_param.z_lens, fixed_param.z_source, cosmo=default_cosmo)


def mass_in_radius(r):
    """returns mass enclosed in r in solar mass"""
    M_r = lens_cosmo.mass_in_theta_E(r)
    return M_r


def mass_in_DM_halo(theta_E):
    """assuming SIE, uses the correlation by Zahid et al. 2018 to computes M_200 of host DM halo given theta_E of total mass"""
    sigma_v = lens_cosmo.sis_theta_E2sigma_v(theta_E)
    #print("sigma_v", sigma_v)
    M_200 = 10**(0.09 + 3.48 * np.log10(sigma_v / 100)) * 1e12
    return M_200


def mass2theta_E(mass):
    """returns mass enclosed in r in solar mass"""
    theta_E = lens_cosmo.phys2arcsec_lens(np.sqrt(mass / np.pi / lens_cosmo.sigma_crit))
    return theta_E


def get_substruct_concentraion_Correr(substruct_mass):
    """C. A. Correa, J. S. B. Wyithe, J. Schaye, and A. R. Duffy, 2015"""
    return (0.0279*(substruct_mass/1e16)**0.36 + 1.942e-5*(substruct_mass / 3e-3)**0.0879)**(-0.3333)


def get_substruct_concentraion_Duffy(substruct_mass):
    """A. R. Duffy, J. Schaye, S. T. Kay and C. D. Vecchia, 2008"""
    return 5.22 * (substruct_mass / 2e12)**(-0.072) * (1 + fixed_param.z_lens)**(-0.42)


#def get_substruct_concentration_SanchezConde1(M_200):
#    """Miguel A. Sanchez-Conde1 and Francisco Prada, 2014"""
    #scaled_M = M_200


def get_NFW_param(mass, concentration):
    Rs, alpha_Rs = lens_cosmo.nfw_physical2angle(mass, concentration)
    return Rs, alpha_Rs


def get_TNFW_param(mass, concentration, truncation_ratio=5):
    Rs, alpha_Rs = get_NFW_param(mass, concentration)
    r_trunc = Rs * truncation_ratio
    return Rs, alpha_Rs, r_trunc


def get_lens_model_macro(sigma_v):
    theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v)
    lens_model_list = ['SIE', 'SHEAR_GAMMA_PSI']
    kwargs_pemd = {'theta_E': theta_E, 'center_x': fixed_param.ra_lens, 'center_y': fixed_param.dec_lens, 'e1': fixed_param.lens_e1, 'e2': fixed_param.lens_e2}
    kwargs_shear = {'gamma_ext': fixed_param.gamma_ext, 'psi_ext': fixed_param.psi_ext, 'ra_0': 0, 'dec_0': 0}
    kwargs_lens = [kwargs_pemd, kwargs_shear]
    return lens_model_list, kwargs_lens


def update_kwargs_with_substruct(lens_model_list, kwargs_lens, substruct_mass, center_x, center_y, profile='TFNW', 
                                 concentration_relation='Correr', truncation_ratio=5, fixed_concentration=None):
    lens_model_list_updated = copy.deepcopy(lens_model_list)
    kwargs_lens_updated = copy.deepcopy(kwargs_lens)
    if substruct_mass == 0:
        return lens_model_list_updated, kwargs_lens_updated, np.nan
    if profile in ['NFW', 'TNFW']:
        if fixed_concentration is not None:
            concentration = fixed_concentration
        elif concentration_relation == 'Correr':
            concentration = get_substruct_concentraion_Correr(substruct_mass)
        elif concentration_relation == 'Duffy':
            concentration = get_substruct_concentraion_Duffy(substruct_mass)
    else:
        concentration = None
    if profile == 'NFW':
        Rs, alpha_Rs = get_NFW_param(substruct_mass, concentration)
        kwargs_lens_updated.append({'alpha_Rs': alpha_Rs, 'Rs': Rs, 'center_x': center_x, 'center_y': center_y})
    elif profile == 'TNFW':
        Rs, alpha_Rs, r_trunc = get_TNFW_param(substruct_mass, concentration, truncation_ratio=truncation_ratio)
        kwargs_lens_updated.append({'alpha_Rs': alpha_Rs, 'Rs': Rs, 'r_trunc': r_trunc, 'center_x': center_x, 'center_y': center_y})
    elif profile in ['SIS', 'POINT_MASS']:
        theta_E = mass2theta_E(substruct_mass)
        kwargs_lens_updated.append({'theta_E': theta_E, 'center_x': center_x, 'center_y': center_y})
    elif profile == 'SIS_TRUNCATED':
        theta_E = mass2theta_E(substruct_mass)
        r_trunc = theta_E * truncation_ratio
        kwargs_lens_updated.append({'theta_E': theta_E, 'r_trunc': r_trunc, 'center_x': center_x, 'center_y': center_y})
    lens_model_list_updated.append(profile)
    return lens_model_list_updated, kwargs_lens_updated, concentration