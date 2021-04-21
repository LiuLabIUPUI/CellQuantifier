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

	def plot_3d_foci_1ch(self):
		Ch1_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				self.settings['Ch1 detData label'])
		min_f = self.settings['Min frame number']
		Ch1_df = Ch1_df[ Ch1_df['frame']>=min_f ]

		select_frames = Ch1_df.drop_duplicates('frame')['frame'].sort_values().to_numpy()

		if self.settings['If plot even layer only']:
			select_frames = select_frames[::2]
			Ch1_df = Ch1_df[ Ch1_df['frame'].isin(select_frames)]

		print(select_frames)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# """
		# ~~~~Add foci plot~~~~
		# """
		ax.scatter(Ch1_df['x']*self.settings['Pixel size'],
				Ch1_df['y']*self.settings['Pixel size'],
				Ch1_df['frame']*self.settings['Z stack size'],
				marker='o',
				s=Ch1_df['r']**2, c=[[0,0,0]], alpha=0.5)

		# """
		# ~~~~Add boundary plot~~~~
		# """
		if self.settings['If_plot_boundary']:
			bdr_mask = imread(self.settings['Input path'] + \
				self.root_name + self.settings['Boundary label'])
			for i in range(len(bdr_mask)):
				frame_no = i

				if frame_no >= min_f and frame_no in select_frames:
					mask = bdr_mask[i]
					if mask.ndim == 3:
						mask = mask[0]

					X, Y, Z = mask_to_3d_coord(mask)
					X = X * self.settings['Pixel size']
					Y = Y * self.settings['Pixel size']
					Z = Z * self.settings['Z stack size'] * frame_no
					ax.scatter(X, Y, Z, c=[[0.12,0.56,1]], s=0.1, alpha=0.5)

		# # """
		# # ~~~~By cell, add text~~~~
		# # """
		# colocal_df = pd.read_csv(self.settings['Input path']+self.root_name + \
		#  		self.settings['Colocal df label'])
		# bdr_focinum_df = pd.read_csv(self.settings['Input path'] + \
		# 		self.root_name + self.settings['Boudary focinum df label'])
		#
		# colocal_df = colocal_df[ colocal_df['frame'].isin(select_frames)]
		# bdr_focinum_df = bdr_focinum_df[ bdr_focinum_df['frame'].isin(select_frames)]
		#
		# ova_overlap_ratio = colocal_df['ova overlap ratio'].median()
		# mhc1_overlap_ratio = colocal_df['mhc1 overlap ratio'].median()
		# ova_bdr_ratio = bdr_focinum_df['ova_bdr_num'].sum() / \
		# (bdr_focinum_df['ova_bdr_num'].sum() + bdr_focinum_df['ova_itl_num'].sum())
		# mhc1_bdr_ratio = bdr_focinum_df['mhc1_bdr_num'].sum() / \
		# (bdr_focinum_df['mhc1_bdr_num'].sum() + bdr_focinum_df['mhc1_itl_num'].sum())
		#
		# ax.text2D(1,
		# 		0.9,
		# 		"OVA overlap ratio: %.2f\nMHC1 overlap ratio: %.2f\nOVA bdr ratio: %.2f\nMHC1 bdr ratio: %.2f\n" \
		# 		%(ova_overlap_ratio, mhc1_overlap_ratio, ova_bdr_ratio, mhc1_bdr_ratio),
		# 		horizontalalignment='right',
		# 		verticalalignment='bottom',
		# 		fontsize = 12,
		# 		color = (0,0,0, 0.8),
		# 		transform=ax.transAxes,
		# 		weight = 'bold',
		# 		)
		#
		# plt_data_df = pd.DataFrame({
		# 	'ova overlap ratio': ova_overlap_ratio,
		# 	'mhc1 overlap ratio': mhc1_overlap_ratio,
		# 	'mhc1_bdr_ratio': mhc1_bdr_ratio,
		# 	'ova_bdr_ratio': ova_bdr_ratio,
		# 	}, index=[0], dtype=float)
		# plt_data_df.round(3).to_csv(self.settings['Output path'] \
		# 	+ self.root_name + '-ratio-pltData.csv', index=False)


		# # """
		# # ~~~~By layer, no text added~~~~
		# # """
		# colocal_df = pd.read_csv(self.settings['Input path']+self.root_name + \
		#  		self.settings['Colocal df label'])
		# bdr_focinum_df = pd.read_csv(self.settings['Input path'] + \
		# 		self.root_name + self.settings['Boudary focinum df label'])
		#
		# colocal_df = colocal_df[ colocal_df['frame'].isin(select_frames)]
		# bdr_focinum_df = bdr_focinum_df[ bdr_focinum_df['frame'].isin(select_frames)]
		#
		# colocal_df = colocal_df[ colocal_df['frame']!=len(bdr_mask)-1]
		# bdr_focinum_df = bdr_focinum_df[ bdr_focinum_df['frame']!=len(bdr_mask)-1]
		#
		# plt_data_df = pd.DataFrame([])
		# plt_data_df['frame'] = colocal_df['frame']
		# plt_data_df['mhc1 overlap ratio'] = colocal_df['mhc1 overlap ratio']
		# plt_data_df['ova overlap ratio'] = colocal_df['ova overlap ratio']
		# plt_data_df['mhc1_bdr_num'] = bdr_focinum_df['mhc1_bdr_num']
		# plt_data_df['mhc1_itl_num'] = bdr_focinum_df['mhc1_itl_num']
		# plt_data_df['ova_bdr_num'] = bdr_focinum_df['ova_bdr_num']
		# plt_data_df['ova_itl_num'] = bdr_focinum_df['ova_itl_num']
		#
		# plt_data_df['mhc1_bdr_ratio'] = plt_data_df['mhc1_bdr_num'] / \
		# 	(plt_data_df['mhc1_bdr_num'] + plt_data_df['mhc1_itl_num'])
		# plt_data_df['mhc1_itl_ratio'] = plt_data_df['mhc1_itl_num'] / \
		# 	(plt_data_df['mhc1_bdr_num'] + plt_data_df['mhc1_itl_num'])
		# plt_data_df['ova_bdr_ratio'] = plt_data_df['ova_bdr_num'] / \
		# 	(plt_data_df['ova_bdr_num'] + plt_data_df['ova_itl_num'])
		# plt_data_df['ova_itl_ratio'] = plt_data_df['ova_itl_num'] / \
		# 	(plt_data_df['ova_bdr_num'] + plt_data_df['ova_itl_num'])
		#
		# plt_data_df.round(3).to_csv(self.settings['Output path'] \
		# 	+ self.root_name + '-ratio-pltData.csv', index=False)

		# """
		# ~~~~Format and save~~~~
		# """
		# ax.axis("off")
		ax.grid(True)
		# ax.set_xticks([])
		# ax.set_yticks([])
		# ax.set_zticks([])
		plt.show()

		# fig.savefig('/home/linhua/Desktop/Figure_1.pdf', dpi=600)
		# plt.clf(); plt.close()
		# sys.exit()

		self.save_config()



def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find(settings['Str in filename'])]
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
