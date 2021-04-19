import pims; import pandas as pd; import numpy as np
import os
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
import glob
import sys

from cellquantifier.phys import *


class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

	def mergedf_to_figdata(self):
		df = pd.read_csv(self.settings['Input path'] + self.root_name + '.csv')
		exp_labels = df['exp_label'].drop_duplicates().sort_values()
		figdata_dfs = []

		if 'traj_length' in df:
			df = df[ df['traj_length']>self.settings['traj_length_thres'] ]

		if self.settings['add_stepsize']:
			print("######################################")
			print("add_stepsize...")
			print("######################################")
			df = add_stepsize(df, pixel_size=self.settings['pixel_size'])

		if self.settings['add_constrain_length']:
			print("######################################")
			print("add_constrain_length...")
			print("######################################")
			df = add_constrain_length(df, pixel_size=self.settings['pixel_size'])

		if self.settings['fit_D_alpha_with_C']:
			print("######################################")
			print("fit_D_alpha_with_C...")
			print("######################################")
			df = add_D_alpha(df, pixel_size=self.settings['pixel_size'],
				frame_rate=self.settings['frame_rate'],
				divide_num=5,
				fit_method='fit_msd2')

		if 'alpha' in df:
			df = df[ df['alpha']>0 ]

		if self.settings['drop_duplicates']:
			df = df.drop_duplicates('particle')

		for exp_label in exp_labels:
			tmp_figdata = df.loc[ df['exp_label']==exp_label, \
					self.settings['figData col name'] ].to_numpy()
			tmp_df = pd.DataFrame(tmp_figdata, columns=[exp_label])
			figdata_dfs.append(tmp_df)

		df_figdata = pd.concat(figdata_dfs, axis=1)

		df_figdata.round(3).to_csv(self.settings['Output path'] + \
			self.root_name + '-' + self.settings['figData col name'] + \
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
