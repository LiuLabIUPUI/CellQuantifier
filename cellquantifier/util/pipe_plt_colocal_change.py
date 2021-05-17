import pims; import pandas as pd; import numpy as np
import os
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
from skimage.morphology import binary_dilation, binary_erosion, disk
import glob
import sys
import matplotlib.pyplot as plt

import seaborn as sns

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
				'-config-colocal.csv', header=False)

	def plt_colocal_change(self):

		if self.settings['Ch1 label'] in self.root_name:
			# Get Ch1_blob_mask and Ch1_blob_df
			Ch1_blob_mask = imread(self.settings['Input path'] + \
				self.root_name + '-blobMask.tif')
			if self.settings['Ch1 mask dilation pixel']:
				dila_disk = disk(self.settings['Ch1 mask dilation pixel'])
				for i in range(len(Ch1_blob_mask)):
					Ch1_blob_mask[i] = binary_dilation(Ch1_blob_mask[i],
							selem=dila_disk)
			Ch1_blob_df = pd.read_csv(self.settings['Input path'] + \
				self.root_name + '-detData.csv')

			# Get Ch2_blob_mask and Ch2_blob_df
			Ch2_label = self.root_name
			Ch2_label = Ch2_label[:-len(self.settings['Ch1 label'])]
			Ch2_label = Ch2_label + self.settings['Ch2 label']
			Ch2_blob_mask = imread(self.settings['Input path'] + Ch2_label + \
					'-blobMask.tif')
			Ch2_blob_df = pd.read_csv(self.settings['Input path'] + \
				Ch2_label + '-detData.csv')

			# Generate foci colocal data
			colocal_arr = []

			for i in range(len(Ch1_blob_mask)):
				curr_Ch1_blobdf = Ch1_blob_df[ Ch1_blob_df['frame']==i ]
				curr_Ch2_blobdf = Ch2_blob_df[ Ch2_blob_df['frame']==i ]
				curr_colocal_foci_num = 0

				for index in curr_Ch2_blobdf.index:
					r = int(curr_Ch2_blobdf.loc[index, 'x'])
					c = int(curr_Ch2_blobdf.loc[index, 'y'])
					if Ch1_blob_mask[i, r, c]>0:
						curr_colocal_foci_num = curr_colocal_foci_num + 1

				try:
					curr_Ch1_overlap_ratio = curr_colocal_foci_num / len(curr_Ch1_blobdf)
				except:
					curr_Ch1_overlap_ratio = None

				try:
					curr_Ch2_overlap_ratio = curr_colocal_foci_num / len(curr_Ch2_blobdf)
				except:
					curr_Ch2_overlap_ratio = None

				colocal_arr.append([i, curr_colocal_foci_num,
					curr_Ch1_overlap_ratio, curr_Ch2_overlap_ratio])

				colocal_df = pd.DataFrame(colocal_arr, columns=['frame',
					'colocal_foci_num', 'Ch1_colocal_ratio', 'Ch2_colocal_ratio'])


			# plot colocal ratio change
			fig, ax = plt.subplots(figsize=(6,4.5))
			sns.lineplot(x="frame", y="colocal_foci_num", data=colocal_df, ax=ax)
			fig.savefig(self.settings['Output path'] + \
				self.root_name + '-colocal-change.pdf', dpi=600)
			plt.clf(); plt.close()


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
