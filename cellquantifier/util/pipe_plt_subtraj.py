import pandas as pd; import numpy as np
import os
from datetime import date, datetime
import pims
import glob
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from ..plot.plotutil import *

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
				'-config-plt3dfoci.csv', header=False)

	def plot_subtraj(self):
		frames = pims.open(self.settings['Input path'] + self.root_name + \
			'.tif')
		phys_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
			'-physData.csv')

		# """
		# ~~~~~~~~filters~~~~~~~~
		# """
		if self.settings['Subtraj type']:
			phys_df = phys_df[ phys_df['subparticle_type']== \
				self.settings['Subtraj type'] ]

		if self.settings['Subtraj length thres']:
			phys_df = phys_df[ phys_df['subparticle_traj_length']>= \
				self.settings['Subtraj length thres'] ]

		if self.settings['Subtraj travel min distance']:
			phys_df = phys_df[ phys_df['subparticle_travel_dist']>= \
				self.settings['Subtraj travel min distance'] ]

		if self.settings['Subtraj travel max distance']:
			phys_df = phys_df[ phys_df['subparticle_travel_dist']<= \
				self.settings['Subtraj travel max distance'] ]

		phys_df = phys_df.rename(columns={
				'D':'original_D',
				'alpha':'original_alpha',
				'traj_length':'orig_traj_length',
				'particle':'original_particle',
				'subparticle': 'particle',
				'subparticle_D': 'D',
				})


		# """
		# ~~~~~~~~Optimize the colorbar format~~~~~~~~
		# """
		if len(phys_df.drop_duplicates('particle')) > 1:
			D_max = phys_df['D'].quantile(0.9)
			D_min = phys_df['D'].quantile(0.1)
			D_range = D_max - D_min
			cb_min=D_min
			cb_max=D_max
			cb_major_ticker=round(0.2*D_range)
			cb_minor_ticker=round(0.2*D_range)
		else:
			cb_min, cb_max, cb_major_ticker, cb_minor_ticker = None, None, None, None


		fig, ax = plt.subplots()
		anno_traj(ax, phys_df,

					show_image=True,
					image = frames[0],

					show_scalebar=False,

					show_colorbar=True,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

		            show_traj_num=False,

					show_traj_end=False,

					show_particle_label=False,
					)
		plt.show()
		# fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-' \
		# 			+ subtype + '-subtrajs.pdf', dpi=300)
		# plt.clf(); plt.close()


def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		root_name = path.split('/')[-1]
		root_name = root_name[:root_name.find('.tif')]
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
