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

	def save_config(self):
		settings_df = pd.DataFrame.from_dict(data=self.settings, orient='index')
		settings_df = settings_df.drop(['Input path', 'Output path'])
		settings_df.to_csv(self.settings['Output path'] + self.root_name + \
				'-config-colocal.csv', header=False)

	def extract_channels(self):
		period = self.settings['Period (frames)']

		frames = imread(self.settings['Input path'] + self.root_name + \
				'.tif')
		Ch1 = np.zeros((len(frames)//period, \
			frames.shape[1], frames.shape[2]))
		Ch2 = np.zeros((len(frames)//period, \
			frames.shape[1], frames.shape[2]))

		for i in range(len(frames)):
			if i%period == self.settings['Ch1 index']:
				Ch1[i//period] = frames[i] / frames[i].max()
			if i%period == self.settings['Ch2 index']:
				Ch2[i//period] = frames[i] / frames[i].max()

		# Ch1 = Ch1 / Ch1.max()
		Ch1 = img_as_ubyte(Ch1)
		# Ch2 = Ch2 / Ch2.max()
		Ch2 = img_as_ubyte(Ch2)
		imsave(self.settings['Output path'] + self.root_name + \
			self.settings['Ch1 label'] + '.tif', Ch1)
		imsave(self.settings['Output path'] + self.root_name + \
			self.settings['Ch2 label'] + '.tif', Ch2)

		self.save_config()


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
