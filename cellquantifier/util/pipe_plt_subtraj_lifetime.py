import pandas as pd; import numpy as np
import os
from datetime import date, datetime
import glob
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from ..plot.plotutil import *
from ..segm import *

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

	def plot_subtraj_lifetime(self):

		phys_df = pd.read_csv(self.settings['Input path']+self.root_name)

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

		phys_df = phys_df.drop_duplicates('subparticle')
		fig, ax = plt.subplots(figsize=(6,4.5))
		plt.hist(x=phys_df['subparticle_traj_length'])
		ax.set_xlabel(self.settings['Subtraj type'] + '_subparticle_lifetime')
		ax.set_ylabel('count')
		plt.show()

		filename = self.root_name
		root_name = filename[:filename.find('.')]
		fig.savefig(self.settings['Output path'] + root_name + '-' + \
			self.settings['Subtraj type'] + '-subtraj-lifetime.pdf', dpi=600)
		plt.clf(); plt.close()



def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		root_name = path.split('/')[-1]
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
