import pims; import pandas as pd; import numpy as np
import os
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
import glob
import sys

import trackpy as tp


class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

	def mergedf_to_msddata(self):
		df = pd.read_csv(self.settings['Input path'] + self.root_name + '.csv')
		exp_labels = df['exp_label'].drop_duplicates().sort_values()
		figdata_dfs = []

		if 'alpha' in df:
			df = df[ df['alpha']>0 ]

		if 'traj_length' in df:
			df = df[ df['traj_length']>self.settings['traj_length_thres'] ]

		for exp_label in exp_labels:
			exp_df = df.loc[ df['exp_label']==exp_label, \
							['x', 'y', 'frame', 'particle'] ]

			im = tp.imsd(exp_df,
					mpp=self.settings['pixel_size'],
					fps=self.settings['frame_rate'],
					max_lagtime=np.inf,
					)

			n = len(im.index) #for use in stand err calculation
			m = int(round(len(im.index)/5))
			im = im.head(m)
			im = im*1e6

			imsd_mean = im.mean(axis=1)
			imsd_std = im.std(axis=1, ddof=0)
			x = imsd_mean.index.to_numpy()
			y = imsd_mean.to_numpy()
			n_data_pts = np.sqrt(np.linspace(n-1, n-m, m))
			yerr = np.divide(imsd_std.to_numpy(), n_data_pts)

			tmp_df = pd.DataFrame({
		            exp_label + '_t': x,
		            exp_label + '_MSD': y,
		            exp_label + '_err': yerr,
		            })

			figdata_dfs.append(tmp_df)

		df_figdata = pd.concat(figdata_dfs, axis=1)

		df_figdata.round(3).to_csv(self.settings['Output path'] + \
			self.root_name + '-msd' + \
			'.csv', index=False)

def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob.glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('.')]
		if root_name not in root_name_list:
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
