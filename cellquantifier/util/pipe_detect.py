import pandas as pd; import numpy as np
import pims
import os
from datetime import date, datetime
import glob
import sys
import matplotlib.pyplot as plt

from ..smt.detect import detect_blobs, detect_blobs_batch
from ..smt.fit_psf import fit_psf, fit_psf_batch
from ..plot.plotutil import anno_blob
from ..deno import filter, filter_batch
from ..video import anim_blob

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
				'-config-detect-frame' + str(self.settings['Frame number'])+ '.csv',
				header=False)

	# def check_detect(self):
	#
	# 	print("######################################")
	# 	print("Check detection")
	# 	print("######################################")
	#
	# 	check_frame_ind = [0, 50, 100, 150, 200]
	#
	# 	frames = pims.open(self.settings['Input path'] + self.root_name + \
	# 								'.tif')
	#
	# 	for ind in check_frame_ind:
	# 		blobs_df, det_plt_array = detect_blobs(frames[ind],
	# 									min_sig=self.settings['Blob_min_sigma'],
	# 									max_sig=self.settings['Blob_max_sigma'],
	# 									num_sig=self.settings['Blob_num_sigma'],
	# 									blob_thres_rel=self.settings['Blob_thres_rel'],
	# 									overlap=0,
	#
	# 									peak_thres_rel=self.settings['Blob_pk_thresh_rel'],
	# 									r_to_sigraw=1,
	# 									pixel_size=self.settings['Pixel size'],
	#
	# 									diagnostic=True,
	# 									pltshow=True,
	# 									plot_r=True,
	# 									truth_df=None)

	def detect(self):

		print("######################################")
		print("Detect")
		print("######################################")

		frames = pims.open(self.settings['Output path'] + self.root_name + \
				'.tif')
		ind = self.settings['Frame number']

		# for ind in self.settings['Frame number']:
		frame = frames[ind]
		blobs_df, det_plt_array = detect_blobs(frame,
					min_sig=self.settings['Blob_min_sigma'],
					max_sig=self.settings['Blob_max_sigma'],
					num_sig=self.settings['Blob_num_sigma'],
					blob_thres_rel=self.settings['Blob_thres_rel'],
					overlap=0.5,

					peak_thres_rel=self.settings['Blob_pk_thresh_rel'],
					r_to_sigraw=1,
					pixel_size=self.settings['Pixel size'],

					diagnostic=False,
					pltshow=True,
					plot_r=True,
					truth_df=None)

		if self.settings['If_fit']:
			blobs_df, fit_plt_array = fit_psf(frame,
		            blobs_df,
		            diagnostic=False,
		            pltshow=True,
		            diag_max_dist_err=self.settings['max_dist_err'],
		            diag_max_sig_to_sigraw=self.settings['max_sig_to_sigraw'],
					diag_min_slope=self.settings['min_slope'],
					diag_min_mass=self.settings['min_mass'],
		            truth_df=None,
		            segm_df=blobs_df)

			blobs_df = blobs_df[ blobs_df['dist_err']<self.settings['max_dist_err'] ]
			blobs_df = blobs_df[ blobs_df['sigx_to_sigraw']<self.settings['max_sig_to_sigraw'] ]
			blobs_df = blobs_df[ blobs_df['sigy_to_sigraw']<self.settings['max_sig_to_sigraw'] ]
			blobs_df = blobs_df[ blobs_df['slope']>self.settings['min_slope'] ]
			blobs_df = blobs_df[ blobs_df['mass']>self.settings['min_mass'] ]
			blobs_df = blobs_df[ blobs_df['r']<=self.settings['Blob_max_sigma'] ]
			blobs_df['r'] = (blobs_df['sig_x'] + blobs_df['sig_y']) / 2

		fig, ax = plt.subplots(figsize=(6,6))
		ax.imshow(frame, cmap="gray", aspect='equal')
		anno_blob(ax, blobs_df, marker='^', markersize=10,
				plot_r=True, color=(0,0,1,0.8))
		plt.show()

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df.round(3).to_csv(self.settings['Output path'] + self.root_name + \
				'-detData-frame' + str(ind) + '.csv',
				index=False)

		# self.save_config()


	def detect_batch(self):

		print("######################################")
		print("Detect")
		print("######################################")

		frames = pims.open(self.settings['Output path'] + self.root_name + \
				'.tif')
		ind = self.settings['Frame number']

		blobs_df, det_plt_array = detect_blobs_batch(frames,
					min_sig=self.settings['Blob_min_sigma'],
					max_sig=self.settings['Blob_max_sigma'],
					num_sig=self.settings['Blob_num_sigma'],
					blob_thres_rel=self.settings['Blob_thres_rel'],
					overlap=0.5,

					peak_thres_rel=self.settings['Blob_pk_thresh_rel'],
					r_to_sigraw=1,
					pixel_size=self.settings['Pixel size'],

					diagnostic=False,
					pltshow=True,
					plot_r=True,
					truth_df=None)

		if self.settings['If_fit']:
			blobs_df, fit_plt_array = fit_psf_batch(frames,
		            blobs_df,
		            diagnostic=False,
		            pltshow=False,
		            diag_max_dist_err=self.settings['max_dist_err'],
		            diag_max_sig_to_sigraw=self.settings['max_sig_to_sigraw'],
					diag_min_slope=self.settings['min_slope'],
					diag_min_mass=self.settings['min_mass'],
		            truth_df=None,
		            segm_df=blobs_df)

			blobs_df = blobs_df[ blobs_df['dist_err']<self.settings['max_dist_err'] ]
			blobs_df = blobs_df[ blobs_df['sigx_to_sigraw']<self.settings['max_sig_to_sigraw'] ]
			blobs_df = blobs_df[ blobs_df['sigy_to_sigraw']<self.settings['max_sig_to_sigraw'] ]
			blobs_df = blobs_df[ blobs_df['slope']>self.settings['min_slope'] ]
			blobs_df = blobs_df[ blobs_df['mass']>self.settings['min_mass'] ]
			blobs_df = blobs_df[ blobs_df['r']<=self.settings['Blob_max_sigma'] ]
			blobs_df['r'] = (blobs_df['sig_x'] + blobs_df['sig_y']) / 2

		det_plt_array = anim_blob(blobs_df, frames,
									pixel_size=self.settings['Pixel size'],
									blob_markersize=10,
									)
		try:
			imsave(self.settings['Output path'] + self.root_name + '-detVideo.tif', det_plt_array)
		except:
			pass

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df.round(3).to_csv(self.settings['Output path'] + self.root_name + \
				'-detData' + '.csv',
				index=False)

		self.save_config()



def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob.glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('.')]
		root_name_list.append(root_name)

		for exclude_str in settings['Strs not in filename']:
			if exclude_str in root_name:
				root_name_list.remove(root_name)

	return np.array(sorted(root_name_list))

def analMeta_to_dict(analMeta_path):
	df = pd.read_csv(analMeta_path, header=None, index_col=0, na_filter=False)
	df = df.rename(columns={1:'value'})
	srs = df['value']

	dict = {}
	for key in srs.index:
		try: dict[key] = ast.literal_eval(srs[key])
		except: dict[key] = srs[key]
	return dict

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
