import pims; import pandas as pd; import numpy as np
import os; import os.path as osp; import ast
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
import glob
import sys

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
				'-config-thresMask.csv', header=False)

	def get_thres_mask(self):
		frames = imread(self.settings['Input path'] + self.root_name + '.tif')
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)

		# If only 1 frame available, duplicate it to total_frames_num
		total_frames_num = 1
		if frames.ndim==2:
			dup_frames = np.zeros((total_frames_num, \
				frames.shape[0], frames.shape[1]), dtype=frames.dtype)
			for i in range(total_frames_num):
				dup_frames[i] = frames
			frames = dup_frames

		# Get mask file and save it using 255 and 0
		masks_thres = get_thres_mask_batch(frames,
			sig=self.settings['Mask sig'],
			thres_rel=self.settings['Mask thres_rel'],
			)

		self.save_config()

		return masks_thres


	def generate_thres_mask(self):
		print("######################################")
		print("Generate thres mask")
		print("######################################")
		masks_thres = self.get_thres_mask()
		masks_thres_255 = np.rint(masks_thres / \
							masks_thres.max() * 255).astype(np.uint8)
		imsave(self.settings['Output path'] + self.root_name + '-thresMask.tif',
				masks_thres_255)

		print("######################################")
		print("Generate dist2thresMask")
		print("######################################")
		dist_masks = img_as_int(get_dist2boundary_mask_batch(masks_thres))
		imsave(self.settings['Output path'] + self.root_name + \
			'-dist2thresMask.tif', dist_masks)


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

def analMeta_to_dict(analMeta_path):
	df = pd.read_csv(analMeta_path, header=None, index_col=0, na_filter=False)
	df = df.rename(columns={1:'value'})
	srs = df['value']

	dict = {}
	for key in srs.index:
		try: dict[key] = ast.literal_eval(srs[key])
		except: dict[key] = srs[key]
	return dict

def pipe_batch(settings_dict, control_list, load_configFile=False):

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

		# If load_config==True, then load existing config file
		if load_configFile:
			existing_settings = analMeta_to_dict(settings_dict['Input path'] + \
							root_name + '-config-blobMask.csv')
			existing_settings['Input path']= settings_dict['Input path']
			existing_settings['Output path'] = settings_dict['Output path']
			settings_dict = existing_settings
			print(settings_dict)

		pipe = Pipe(settings_dict, control_list, root_name)
		for func in control_list:
			getattr(pipe, func)()
