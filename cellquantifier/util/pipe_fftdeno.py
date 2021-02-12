import pims; import pandas as pd; import numpy as np
import os
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_int
import glob
import sys
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

from ..io import *


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

	def fft_denoise(self):
		frames = imread(self.settings['Input path'] + self.root_name + \
				'.tif')

		if frames.ndim==2:
			a = fftshift(fft2(frames))
			b = ifft2(ifftshift(a))

			filter = np.zeros((frames.shape[0], frames.shape[1]))
			row_center = frames.shape[0] // 2
			col_center = frames.shape[1] // 2

			f_r = 150
			filter[row_center-f_r:row_center+f_r, col_center-f_r:col_center+f_r]=1

			a_f = a * filter
			b_f = ifft2(ifftshift(a_f))

			fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
			ax = ax.ravel()

			ax[0].imshow(np.abs(b), cmap='gray')
			ax[1].imshow(np.log(np.abs(a)), cmap='magma')
			ax[2].imshow(np.log(np.abs(a_f)), cmap='magma')
			ax[3].imshow(np.abs(b_f), cmap='gray')

			ax[0].set_title("Original image")
			ax[1].set_title("Original FFT")
			ax[2].set_title("Filtered FFT")
			ax[3].set_title("Filtered image")

			plt.show()

			# frs = [50, 75, 100, 125, 150]
			# bfs = []
			# for f_r in frs:
			# 	filter = np.zeros((frames.shape[0], frames.shape[1]))
			# 	filter[row_center-f_r:row_center+f_r, col_center-f_r:col_center+f_r]=1
			# 	af = a * filter
			# 	bf = np.abs(ifft2(ifftshift(af)))
			# 	bfs.append(bf)
			#
			# bfs.append(frames)
			#
			# fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
			# ax = ax.ravel()
			# for i in range(len(bfs)):
			# 	ax[i].imshow(bfs[i], cmap='gray')
			# plt.show()






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
