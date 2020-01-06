import pandas as pd
import os

class Config():

	def __init__(self, config):

		self.INPUT_PATH = config['Meta input_path']
		self.OUTPUT_PATH = config['Meta output_path']
		self.ROOT_NAME = self.INPUT_PATH.split('/')[-1].split('.')[0]

		self.START_FRAME = config['Meta start_frame']
		self.CHECK_FRAME = config['Meta check_frame']
		self.TRANGE = range(config['Meta start_frame'], config['Meta end_frame'])

		#REGISTRATION SETTINGS
		self.REF_IND_NUM = config['Regi ref_ind_num']
		self.SIG_MASK = config['Regi sig_mask']
		self.THRES_REL = config['Regi thres_rel']
		self.POLY_DEG = config['Regi poly_deg']
		self.ROTATION_MULTIPLIER = config['Regi rotation_multiplier']
		self.TRANSLATION_MULTIPLIER = config['Regi translation_multiplier']

		#SEGMENTATION SETTINGS
		self.MASK_SIG = config['Segm mask_sig']
		self.MASK_THRES = config['Segm mask_thres']
		self.MASK_MIN_SIZE = config['Segm min_size']

		#DENOISE SETTINGS
		self.BOXCAR_RADIUS = config['Deno boxcar_radius']
		self.GAUS_BLUR_SIG = config['Deno gaus_blur_sig']

		#DETECTION SETTINGS
		self.THRESHOLD = config['Det blob_threshold']
		self.MIN_SIGMA = config['Det blob_min_sigma']
		self.MAX_SIGMA = config['Det blob_max_sigma']
		self.NUM_SIGMA = config['Det blob_num_sigma']
		self.PEAK_THRESH_REL = config['Det pk_thresh_rel']
		self.PLOT_R = config['Det plot_r']

		#FITTING SETTINGS
		self.PATCH_SIZE = config['Fitt r_to_sigraw']

	    #TRACKING SETTINGS
		self.SEARCH_RANGE = config['Trak search_range']
		self.MEMORY = config['Trak memory']
		self.FRAME_RATE = config['Meta frame_rate']
		self.PIXEL_SIZE = config['Meta pixel_size']
		self.DIVIDE_NUM = config['Trak divide_num']

		#TRACKING FILTERING SETTINGS
		self.DO_FILTER = config['Filt do_filter']
		self.FILTERS = {

		'MAX_DIST_ERROR': config['Filt max_dist_err'],
		'SIG_TO_SIGRAW' : config['Filt max_sig_to_sigraw'],
		'MAX_DELTA_AREA': config['Filt max_delta_area'],
		'TRAJ_LEN_THRES': config['Filt traj_length_thres']

		}


		#SORTING SETTINGS
		self.DO_SORT = config['Sort do_sort']
		self.SORTERS = {

		'DIST_TO_BOUNDARY': config['Sort dist_to_boundary'],
		'DIST_TO_53BP1' : config['Sort dist_to_53bp1'],

		}

		self.DICT = config

	def clean_dir(self):

		flist = [f for f in os.listdir(self.OUTPUT_PATH)]
		for f in flist:
		    os.remove(os.path.join(self.OUTPUT_PATH, f))

	def save_config(self):

		path = self.OUTPUT_PATH + self.ROOT_NAME + '-analMeta.csv'
		config_df = pd.DataFrame.from_dict(data=self.DICT, orient='index')
		config_df = config_df.drop(['IO input_path', 'IO output_path',
		'Det plot_r', 'Filt do_filter', 'Diag diagnostic',
		'Diag pltshow', 'Check frame index'])
		config_df.to_csv(path, header=False)
