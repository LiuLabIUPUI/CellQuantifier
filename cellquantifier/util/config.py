#==================================================================================================
# config.py - ingests the config dictionary and implements a configuration object using it
#==================================================================================================
# Author:  Clayton Seitz <cwseitz@iu.edu>
#          06/01/2019
#==================================================================================================

import pandas as pd
import os

def get_file_name(path):

	filename = path.split('/')[-1]
	filename = filename.split('-')[:-1]
	filename = '-'.join(filename)

	return filename

class Config():

	def __init__(self, config):

		#I/O
		self.INPUT_PATH = config['IO input_path']
		self.OUTPUT_PATH = config['IO output_path']
		self.ROOT_NAME = config['Raw data file']
		self.TRANGE = range(config['Start frame index'], config['End frame index'])

		#SEGMENTATION SETTINGS
		self.ROI_SIZE = config['Segm size_range']
		self.SEGM_THRESHOLD = config['Segm threshold']

		#REGISTRATION_SETTINGS
		self.WINDOW_SIZE = config['Regi window_size']
		self.REF_INDEX_NUM = config['Regi ref_frame_index']

		#IMAGE PROCESSING SETTINGS
		self.BOXCAR_SIZE = config['Deno boxcar_size']
		self.GAUSS_BLUR_SIGMA = config['Deno gauss_blur_sigma']

		#DETECTION SETTINGS
		self.THRESHOLD = config['Det blob_threshold']
		self.MIN_SIGMA = config['Det blob_min_sigma']
		self.MAX_SIGMA = config['Det blob_max_sigma']
		self.NUM_SIGMA = config['Det blob_num_sigma']
		self.PEAK_THRESH_REL = config['Det pk_thresh_rel']

		#FITTING SETTINGS
		self.PATCH_SIZE = config['Fitt r_to_sigraw']

	    #TRACKING SETTINGS
		self.SEARCH_RANGE = config['Trak search_range']
		self.MEMORY = config['Trak memory']
		self.MIN_TRAJ_LENGTH = config['Trak min_traj_length']
		self.FRAME_RATE = config['Trak framerate']
		self.RESOLUTION = config['Trak pixelsize']
		self.DIVIDE_NUM = config['Trak divide_num']

		#FITTING FILTERING SETTINGS
		self.FROM_CSV = config['Filt from_csv']
		self.MAX_DIST_ERROR = config['Filt max_dist_err']
		self.SIG_TO_SIGRAW = config['Filt sig_to_sigraw']
		self.MAX_DELTA_AREA = config['Filt max_delta_area']
		self.TRAJ_LEN_THRES = config['Filt traj_length_thres']

		#DIAGNOSTIC
		self.SHOW_REG = config['Diag show_reg']
		self.SHOW_MASK = config['Diag show_mask']
		self.SCATTER_DETECTION = config['Diag scatter_det']
		self.SHOW_FIT = config['Diag show_fit']

		self.DICT = config

		if not self.FROM_CSV:
			self.clean_dir()
		self.save_config()

	def save_config(self):

		path = self.OUTPUT_PATH + self.ROOT_NAME + '-analMeta.csv'
		config_df = pd.DataFrame.from_dict(data=self.DICT, orient='index')
		config_df = config_df.drop(['IO input_path', 'IO output_path', 'Filt from_csv', 'Diag show_reg', 'Diag scatter_det', 'Diag show_fit', 'Diag show_mask'])
		config_df.to_csv(path, header=False)

	def clean_dir(self):

		flist = [f for f in os.listdir(self.OUTPUT_PATH)]
		for f in flist:
		    os.remove(os.path.join(self.OUTPUT_PATH, f))
