
class Settings():

	def __init__(self, settings):

	    #io parameters
		self.OUTPUT_DIR = settings['output_dir']

	    #spatial parameters (observation volume)
		self.X_SZ = settings['x_size']
		self.Y_SZ = settings['y_size']
		self.Z_SZ = settings['z_size']
		self.N_DIM = settings['n_dim']

	    #temporal parameters
		self.ACQ_TIME = settings['acq_time']
		self.EXPOSURE_TIME = settings['exposure_time']
		self.FRAME_RATE = settings['frame_rate']
		self.SAMPLE_PERIOD = 1/self.FRAME_RATE

	    #emitter parameters
		self.NPARTICLES = settings['nparticles']
		self.LAT_DIM = settings['lat_dim']
		self.DEG_FREEDOM = settings['deg_freedom']
		self.EX_WAVELENGTH = settings['ex_wavelength']
		self.EM_WAVELENGTH = settings['em_wavelength']
		self.PHOTON_RATE = settings['photon_rate']
		self.PARTICLE_RADIUS = settings['particle_radius']
		self.PARTICLE_MASS = settings['particle mass']

	    #camera parameters
		self.PIXEL_SIZE = settings['pixel_size']
		self.QUANT_EFF = settings['quant_eff']
		self.SENSITIVITY = settings['sensitivity']
		self.SIGMA_DARK = settings['sigma_dark']
		self.BIT_DEPTH = settings['bit_depth']
		self.BASELINE = settings['baseline']
		self.DYN_RANGE = settings['dynamic_range']

	    #microscopy parameters
		self.NUM_APERTURE = settings['num_aperture']
		self.REFR_INDEX = settings['refr_index']
		self.PINHOLE_RAD = settings['pinhole_rad']
		self.PINHOLE_SHAPE = settings['pinhole_shape']

	    #environmental parameters
		self.ENV_TEMP = settings['env_temp']
		self.ENV_VISC = settings['env_visc']

		self.NFRAMES = int(round(self.ACQ_TIME*self.FRAME_RATE))
		self.NCHAN = 1
