import pims; import pandas as pd; import numpy as np
import os
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
import glob
import sys


class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

	def images_to_stack(self):
		images_list = np.array(sorted(glob.glob(self.settings['Input path'] + \
			'*' + self.root_name + '*' + self.settings['Str in filename'])))

		tmp_img = imread(images_list[0])
		if tmp_img.ndim == 3:
			tmp_img = tmp_img[0]
		shape = tmp_img.shape
		stack = np.zeros((len(images_list), shape[0], shape[1]), \
				dtype=tmp_img.dtype)

		for img in images_list:
			m = img.find('frame')
			n = img.find(self.settings['Postfix label'])
			frame_no = int(img[m+5:n])

			curr_img = imread(img)
			stack[frame_no] = curr_img

		imsave(self.settings['Output path'] + self.root_name + \
			self.settings['Postfix label'] + '.tif', stack)



		# frames = imread(self.settings['Input path'] + self.root_name + \
		# 		'.tif')
		# if frames.ndim == 3:
		# 	for i in range(len(frames)):
		# 		imsave(self.settings['Output path'] + self.root_name + \
		# 			'-frame' + str(i) + '.tif', frames[i])


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
