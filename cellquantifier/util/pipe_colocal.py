import pims; import pandas as pd; import numpy as np
import os
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
from skimage.morphology import binary_dilation, binary_erosion, disk
import glob
import sys
import matplotlib.pyplot as plt

from ..io import *
from ..segm import *
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
				'-config-colocal.csv', header=False)

	def dilate_Ch1_mask(self):
		if self.settings['Ch1 label'] in self.root_name:
			frames = imread(self.settings['Input path'] + self.root_name + \
					'.tif')

			# If only 1 frame available, duplicate it to total_frames_num
			total_frames_num = 1
			if frames.ndim==2:
				dup_frames = np.zeros((total_frames_num, \
					frames.shape[0], frames.shape[1]), dtype=frames.dtype)
				for i in range(total_frames_num):
					dup_frames[i] = frames
				frames = dup_frames

			Ch1_blob_df = pd.read_csv(self.settings['Input path'] + \
				self.root_name + '-detData.csv')
			Ch1_blob_mask = blobs_df_to_mask(frames, Ch1_blob_df)

			dila_disk = disk(self.settings['Ch1 mask dilation pixel'])
			for i in range(len(Ch1_blob_mask)):
				Ch1_blob_mask[i] = binary_dilation(Ch1_blob_mask[i],
													selem=dila_disk)

			Ch1_blob_mask_255 = np.rint(Ch1_blob_mask / \
								Ch1_blob_mask.max() * 255).astype(np.uint8)
			imsave(self.settings['Output path'] + self.root_name + \
				'-blobMask-dilate.tif', Ch1_blob_mask_255)

	def generate_colocalmap(self):
		if self.settings['Ch1 label'] in self.root_name:
			Ch1_dilate = imread(self.settings['Input path'] + self.root_name + \
					'-blobMask-dilate.tif')

			Ch2_label = self.root_name
			Ch2_label = Ch2_label
			Ch2_label = Ch2_label[:-len(self.settings['Ch1 label'])]
			Ch2_label = Ch2_label + self.settings['Ch2 label']
			Ch2_mask = imread(self.settings['Input path'] + Ch2_label + \
					'-blobMask.tif')

			frames = imread(self.settings['Input path'] + self.root_name + \
					'.tif')
			colocalmap = np.zeros((frames.shape[0], frames.shape[1], 3))
			colocalmap[:, :, 0] = Ch2_mask / Ch2_mask.max()
			colocalmap[:, :, 1] = Ch1_dilate / Ch1_dilate.max()

			Ch2_blob_df = pd.read_csv(self.settings['Input path'] + \
				Ch2_label + '-detData.csv')

			Ch2_overlap_num = 0
			for index in Ch2_blob_df.index:
				r = int(Ch2_blob_df.loc[index, 'x'])
				c = int(Ch2_blob_df.loc[index, 'y'])
				if Ch1_dilate[0, r, c]>0:
					Ch2_overlap_num = Ch2_overlap_num + 1

			Ch2_overlap_ratio = Ch2_overlap_num / len(Ch2_blob_df)

			fig, ax = plt.subplots(figsize=(6,6))
			ax.imshow(colocalmap)
			ax.text(0.95,
					0.05,
					"OVA overlap ratio: %.2f" %(Ch2_overlap_ratio),
					horizontalalignment='right',
					verticalalignment='bottom',
					fontsize = 12,
					color = (0.5, 0.5, 0.5, 0.5),
					transform=ax.transAxes,
					weight = 'bold',
					)
			# plt.show()
			fig.savefig(self.settings['Output path'] + self.root_name + \
				'-colocalmap.pdf')
			plt.clf(); plt.close()

			# colocalmap = np.zeros((3, frames.shape[0], frames.shape[1]), dtype=frames.dtype)
			# colocalmap[0] = Ch2_mask
			# colocalmap[1] = Ch1_dilate
			# imsave(self.settings['Output path'] + self.root_name + \
			# 	'-colocalmap.tif', colocalmap)


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
