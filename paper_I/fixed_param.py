__author__ = 'aymgal'

# here we set the global parameters used for the simulation of mock data (HST and E-ELT)


##### LENS MODEL SETTINGS #####
lens_model_list = ['SIE', 'SHEAR_GAMMA_PSI']
ra_lens, dec_lens = 0, 0
kwargs_pemd = {'theta_E': 1.8, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1}
kwargs_shear = {'gamma_ext': 0.05, 'psi_ext': 0, 'ra_0': 0, 'dec_0': 0}
kwargs_lens = [kwargs_pemd, kwargs_shear]


##### MASS SUBSTRUCTURES #####
profile_substruct = 'TNFW'


##### MASK #####
mask_outer_radius = 3.6
mask_inner_radius = None


##### SOURCE SETTINGS #####
# name of the galaxy (for the 'high-res' sources) 
galaxy_name = 'NGC1309'

# position of the source galaxy
ra_source, dec_source = 0.1, -0.1

# magnification of the source galaxy
mag_source = 22
