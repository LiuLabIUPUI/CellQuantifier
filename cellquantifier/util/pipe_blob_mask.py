import pims; import pandas as pd; import numpy as np
import os; import os.path as osp; import ast
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
import glob
import sys

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
				'-config-blobMask.csv', header=False)

	def get_blob_mask(self):
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
		imsave(self.settings['Output path'] + self.root_name + '-tempFile.tif',
				frames)
		pims_frames = pims.open(self.settings['Output path'] + self.root_name +
								'-tempFile.tif')

		blobs_df, det_plt_array = detect_blobs(pims_frames[0],
				min_sig=self.settings['Mask blob_min_sigma'],
				max_sig=self.settings['Mask blob_max_sigma'],
				num_sig=self.settings['Mask blob_num_sigma'],
				blob_thres_rel=self.settings['Mask blob_thres_rel'],
				peak_thres_rel=self.settings['Mask blob_pk_thresh_rel'],
				r_to_sigraw=1.4,
				show_scalebar=False,
				pixel_size=10**9,
				diagnostic=True,
				pltshow=False,
				plot_r=True,
				truth_df=None)

		blobs_df, det_plt_array = detect_blobs_batch(pims_frames,
				min_sig=self.settings['Mask blob_min_sigma'],
				max_sig=self.settings['Mask blob_max_sigma'],
				num_sig=self.settings['Mask blob_num_sigma'],
				blob_thres_rel=self.settings['Mask blob_thres_rel'],
				peak_thres_rel=self.settings['Mask blob_pk_thresh_rel'],
				r_to_sigraw=1.4,
				show_scalebar=False,
				pixel_size=10**9,
				diagnostic=False,
				pltshow=False,
				plot_r=False,
				truth_df=None)

		blobs_df.round(3).to_csv(self.settings['Output path'] + self.root_name + \
						'-detData.csv', index=False)

		masks_blob = blobs_df_to_mask(frames, blobs_df)

		os.remove(self.settings['Output path'] + \
			self.root_name + '-tempFile.tif')
		self.save_config()

		return masks_blob


	def generate_blob_mask(self):
		print("######################################")
		print("Generate blob mask")
		print("######################################")
		masks_blob = self.get_blob_mask()
		masks_blob_255 = np.rint(masks_blob / \
							masks_blob.max() * 255).astype(np.uint8)
		imsave(self.settings['Output path'] + self.root_name + '-blobMask.tif',
				masks_blob_255)

		print("######################################")
		print("Generate dist2blob_mask")
		print("######################################")
		dist2blob_masks = img_as_int(get_dist2boundary_mask_batch(masks_blob))
		imsave(self.settings['Output path'] + self.root_name + \
			'-dist2blobMask.tif', dist2blob_masks)


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
