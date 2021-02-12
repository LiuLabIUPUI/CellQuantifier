import pandas as pd; import numpy as np
import os
from datetime import date, datetime
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

	def plot_3d_foci(self):
		Ch1_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				self.settings['Ch1 detData label'])
		Ch2_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				self.settings['Ch2 detData label'])

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(Ch1_df['x'], Ch1_df['y'], Ch1_df['frame'], marker='o',
				s=Ch1_df['r']**2, c=[[0,1,0]], alpha=1)
		ax.scatter(Ch2_df['x'], Ch2_df['y'], Ch2_df['frame'], marker='o',
				s=Ch2_df['r']**2, c=[[1,0,0]], alpha=1)
		# ax.axis("off")
		ax.grid(False)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])
		plt.show()

		# # """
		# # ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~
		# # """
		# all_figures.savefig('/home/linhua/Desktop/Figure_1.pdf', dpi=600)
		# plt.clf(); plt.close()
		# sys.exit()



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
