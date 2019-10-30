from core import io
from skimage.io import imread, imsave
import pandas as pd

from improc import proc
from registration import regi
from segmentation import segm
from sct import cell_tracker
from smt import tracker


#==================================================================================================
#pipeline.py - the main interface between the subpackages and user-space
#==================================================================================================
# Author:  Clayton Seitz <cwseitz@iu.edu>
#          06/01/2019
#==================================================================================================

class Movie():

	def __init__(self, config):

		self.channels = io.read(config)

class Pipeline():

	def __init__(self, config):

		self.movie = Movie(config)
		self.config = config

	def segmentation(self, channel_id, method):

		print("######################################")
		print("Segmenting Image Stacks")
		print("######################################")

		im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-active.tif')
		mask, centroid = segm.segm_batch(self.config, im, method)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-mask.tif', mask)

		return centroid

	def sct(self, channel_id):

		print("######################################")
		print("Single Cell Tracking")
		print("######################################")

		mask = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-mask.tif')
		im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-active.tif')

		mask = cell_tracker.track(im, mask, self.config)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-mask.tif', mask)

	def smt(self, channel_id):

		print("######################################")
		print("Single Molecule Tracking")
		print("######################################")

		im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-raw.tif')
		im_deno = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-deno.tif')
		D, alpha = tracker.track(im, im_deno, self.config) #use raw channel for detection, denoised for fitting

		meta_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + "-analMeta.csv")
		meta_df.loc[len(meta_df)] = ['MSD Curve D (nm^2/s)', D]
		meta_df.loc[len(meta_df)] = ['MSD Curve alpha', alpha]
		meta_df.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + "-analMeta.csv", index=False)

	def improc(self, channel_id):

		print("######################################")
		print("Filtering Image Stacks")
		print("######################################")

		im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-active.tif')
		proc_im = proc.run(im, self.config)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-deno.tif', proc_im)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_id) + '-active.tif', proc_im)

	def register(self, mask_channel, channel_to_register):

		print("######################################")
		print("Registering Images")
		print("######################################")

		im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_to_register) + '-raw.tif')
		mask = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(mask_channel) + '-mask.tif')

		registered_im = regi.run(im, mask, self.config)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-ch' + str(channel_to_register) + '-regi.tif', registered_im)

	def ml(self, path):

		data = pd.read_csv(path)
		ml.run(data)
