import pandas as pd; import numpy as np
import pims
import os
from datetime import date, datetime
import glob
import sys

from ..smt.detect import detect_blobs, detect_blobs_batch

class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

	def save_config(self):
		settings_df = pd.DataFrame.from_dict(data=self.settings, orient='index')
		settings_df = settings_df.drop(['Input path', 'Output path'])
		settings_df.to_csv(self.settings['Output path'] + self.root_name + \
				'-config-detect.csv', header=False)

	def check_detect(self):

		print("######################################")
		print("Check detection")
		print("######################################")

		check_frame_ind = [0, 50, 100, 150, 200]

		frames = pims.open(self.settings['Input path'] + self.root_name + \
									'.tif')

		for ind in check_frame_ind:
			blobs_df, det_plt_array = detect_blobs(frames[ind],
										min_sig=self.settings['Blob_min_sigma'],
										max_sig=self.settings['Blob_max_sigma'],
										num_sig=self.settings['Blob_num_sigma'],
										blob_thres_rel=self.settings['Blob_thres_rel'],
										overlap=0,

										peak_thres_rel=self.settings['Blob_pk_thresh_rel'],
										r_to_sigraw=1,
										pixel_size=self.settings['Pixel size'],

										diagnostic=True,
										pltshow=True,
										plot_r=True,
										truth_df=None)

	def detect(self):

		print("######################################")
		print("Detect")
		print("######################################")

		frames = pims.open(self.config.OUTPUT_PATH + \
				self.config.ROOT_NAME + '-raw.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames,
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres_rel=self.config.THRESHOLD,
									overlap=0,

									peak_thres_rel=self.config.PEAK_THRESH_REL,
									r_to_sigraw=1,
									pixel_size=self.config.PIXEL_SIZE,

									diagnostic=False,
									pltshow=False,
									plot_r=False,
									truth_df=None)

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-detData.csv', index=False)


		det_plt_array = anim_blob(blobs_df, frames,
									pixel_size=self.config.PIXEL_SIZE,
									blob_markersize=5,
									)
		try:
			imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detVideo.tif', det_plt_array)
		except:
			pass

		self.config.save_config()



def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('.')]
		root_name_list.append(root_name)

		for exclude_str in settings['Strs not in filename']:
			if exclude_str in root_name:
				root_name_list.remove(root_name)

	return np.array(sorted(root_name_list))


def pipe_batch(settings_dict, control_list):

	root_name_list = get_root_name_list(settings_dict)

	print("######################################")
	print("Total data num to be processed: %d" % len(root_name_list))
	print(root_name_list)
	print("######################################")

	ind = 0
	tot = len(root_name_list)
	for root_name in root_name_list:
		ind = ind + 1
		print("\n")
		print("Processing (%d/%d): %s" % (ind, tot, root_name))

		pipe = Pipe(settings_dict, control_list, root_name)
		for func in control_list:
			getattr(pipe, func)()
