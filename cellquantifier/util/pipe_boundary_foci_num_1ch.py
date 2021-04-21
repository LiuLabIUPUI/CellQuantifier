import pandas as pd; import numpy as np
import os
from datetime import date, datetime
import glob
import sys
from skimage.io import imread, imsave

from ..phys import count_boundary_foci_num
from ..segm import blobs_df_to_mask

class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

	def get_boundary_focinum_df(self):
		Ch1_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				self.settings['Ch1 detData label'])
		Ch2_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				self.settings['Ch2 detData label'])
		Boundary_mask = imread(self.settings['Input path'] + self.root_name + \
				self.settings['Boundary mask label'])
		Boundary_mask = Boundary_mask // 255

		bdr_focinum_df, bdr_mask = count_boundary_foci_num(
				Boundary_mask=Boundary_mask,
				Boundary_outer_thickness=self.settings['Boundary outer thickness'],
				Boundary_inner_thickness=self.settings['Boundary inner thickness'],
				Ch1_df=Ch1_df,
				Ch2_df=Ch2_df,
				Ch1_label=self.settings['Ch1 foci label'],
				Ch2_label=self.settings['Ch2 foci label'],
				)

		bdr_focinum_df.to_csv(self.settings['Output path'] + \
			self.root_name + '-bdr-foci-num.csv', index=False)

		bdr_mask_255 = np.rint(bdr_mask / \
							bdr_mask.max() * 255).astype(np.uint8)
		imsave(self.settings['Output path'] + \
			self.root_name + '-bdr-ring.tif', bdr_mask_255)

	def generate_boundary_foci_map(self):
		Ch1_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				self.settings['Ch1 detData label'])
		Ch2_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				self.settings['Ch2 detData label'])
		bdr_mask = imread(self.settings['Input path'] + self.root_name + \
				'-bdr-ring.tif')

		colocalmap = np.zeros((bdr_mask.shape[0], bdr_mask.shape[1],
				bdr_mask.shape[2], 3), dtype=bdr_mask.dtype)
		colocalmap[:,:,:,1] = blobs_df_to_mask(bdr_mask, Ch1_df)*255
		colocalmap[:,:,:,0] = blobs_df_to_mask(bdr_mask, Ch2_df)*255
		colocalmap[:,:,:,2] = bdr_mask
		imsave(self.settings['Output path'] + self.root_name + \
			'-bdr-focinum-map.tif', colocalmap)


def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob.glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('-')]
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
