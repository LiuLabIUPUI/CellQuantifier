import pandas as pd; import numpy as np
import pims
import os
from datetime import date, datetime
import glob
import sys
import matplotlib.pyplot as plt

from skimage.io import imread, imsave

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
				'-config-fit' + '.csv',
				header=False)

	def fit_batch(self):

		print("######################################")
		print("Fit")
		print("######################################")

		frames = pims.open(self.settings['Output path'] + self.root_name + \
				'.tif')

		try:
			frames_deno = pims.open(self.settings['Output path'] + self.root_name + \
					'-deno.tif')
		except:
			frames_deno = frames

		blobs_df = pd.read_csv(self.settings['Output path'] + self.root_name + \
				'-detData.csv')

		blobs_df, fit_plt_array = fit_psf_batch(frames_deno,
	            blobs_df,
	            diagnostic=False,
	            pltshow=False,
	            diag_max_dist_err=self.settings['max_dist_err'],
	            diag_max_sig_to_sigraw=self.settings['max_sig_to_sigraw'],
				diag_min_slope=self.settings['min_slope'],
				diag_min_mass=self.settings['min_mass'],
	            truth_df=None,
	            segm_df=blobs_df)

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df['slope'] = blobs_df['A'] / (9 * np.pi * blobs_df['sig_x'] * blobs_df['sig_y'])
		blobs_df.round(3).to_csv(self.settings['Output path'] + self.root_name + \
				'-fitData' + '.csv', index=False)

		self.save_config()

	def anim_fitData(self):

		print("######################################")
		print("Anim fitData")
		print("######################################")

		frames = pims.open(self.settings['Output path'] + self.root_name + \
				'.tif')
		blobs_df = pd.read_csv(self.settings['Output path'] + self.root_name + \
				'-fitData.csv')

		blobs_df = blobs_df[ blobs_df['dist_err']<self.settings['max_dist_err'] ]
		blobs_df = blobs_df[ blobs_df['sigx_to_sigraw']<self.settings['max_sig_to_sigraw'] ]
		blobs_df = blobs_df[ blobs_df['sigy_to_sigraw']<self.settings['max_sig_to_sigraw'] ]
		blobs_df = blobs_df[ blobs_df['slope']>self.settings['min_slope'] ]
		blobs_df = blobs_df[ blobs_df['mass']>self.settings['min_mass'] ]

		fit_plt_array = anim_blob(blobs_df, frames,
									pixel_size=self.settings['Pixel size'],
									blob_markersize=10,
									)
		imsave(self.settings['Output path'] + self.root_name + '-fittVideo.tif', fit_plt_array)


def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob.glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('.tif')]
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
