import pims; import pandas as pd; import numpy as np
import trackpy as tp
import os.path as osp; import os; import ast
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_uint, img_as_int
from skimage.measure import regionprops
import warnings
import glob
import sys

from ..deno import filter_batch
from ..io import *
from ..segm import *
from ..video import *
from ..plot.plotutil import *
from ..regi import get_regi_params, apply_regi_params

from ..smt.detect import detect_blobs, detect_blobs_batch
from ..smt.fit_psf import fit_psf, fit_psf_batch
from ..smt.track import track_blobs
from ..smt.msd import plot_msd_batch, get_sorter_list
from ..smt import *
from ..phys import *
from ..plot import *
from ..plot import plot_phys_1 as plot_merged
from ..phys.physutil import relabel_particles, merge_physdfs

from ..publish._fig_quick_merge3 import *


class Config():

	def __init__(self, config):

		#I/O
		self.INPUT_PATH = config['IO input_path']
		self.OUTPUT_PATH = config['IO output_path']
		self.ROOT_NAME = ''
		self.PIXEL_SIZE = config['Pixel_size']
		self.FRAME_RATE = config['Frame_rate']

		#Mask SETTINGS
		self.DIST2BOUNDARY_MASK_NAME = ''
		self.MASK_SIG_BOUNDARY = config['Mask boundary_mask sig']
		self.MASK_THRES_BOUNDARY = config['Mask boundary_mask thres_rel']
		self.DIST253BP1_MASK_NAME = ''
		self.MASK_SIG_53BP1 = config['Mask 53bp1_mask sig']
		self.MASK_THRES_53BP1 = config['Mask 53bp1_mask thres_rel']
		self.MASK_53BP1_BLOB_NAME = ''
		self.MASK_53BP1_BLOB_THRES = config['Mask 53bp1_blob_threshold']
		self.MASK_53BP1_BLOB_MINSIG = config['Mask 53bp1_blob_min_sigma']
		self.MASK_53BP1_BLOB_MAXSIG = config['Mask 53bp1_blob_max_sigma']
		self.MASK_53BP1_BLOB_NUMSIG = config['Mask 53bp1_blob_num_sigma']
		self.MASK_53BP1_BLOB_PKTHRES_REL = config['Mask 53bp1_blob_pk_thresh_rel']
		self.MASK_53BP1_BLOB_SEARCH_RANGE = config['Mask 53bp1_blob_search_range']
		self.MASK_53BP1_BLOB_MEMORY = config['Mask 53bp1_blob_memory']
		self.MASK_53BP1_BLOB_TRAJ_LENGTH_THRES = config['Mask 53bp1_blob_traj_length_thres']

		#DENOISE SETTINGS
		self.BOXCAR_RADIUS = config['Foci deno boxcar_radius']
		self.GAUS_BLUR_SIG = config['Foci deno gaus_blur_sig']

		#DETECTION SETTINGS
		self.THRESHOLD = config['Foci det blob_thres_rel']
		self.MIN_SIGMA = config['Foci det blob_min_sigma']
		self.MAX_SIGMA = config['Foci det blob_max_sigma']
		self.NUM_SIGMA = config['Foci det blob_num_sigma']
		self.PEAK_MIN = 0
		self.NUM_PEAKS = 1
		self.PEAK_THRESH_REL = config['Foci det pk_thres_rel']
		self.MASS_THRESH_REL = 0
		self.PEAK_R_REL = 0
		self.MASS_R_REL = 0

		#TRACKING SETTINGS
		self.SEARCH_RANGE = config['Trak search_range']
		self.MEMORY = config['Trak memory']
		self.DIVIDE_NUM = config['Trak divide_num']

		#TRACKING FILTERING SETTINGS
		if (config['Foci filt max_dist_err']=='') & \
			(config['Foci filt max_sig_to_sigraw']=='') & \
			(config['Foci filt max_delta_area']==''):
			self.DO_FILTER = False
		else:
			self.DO_FILTER = True

		self.FILTERS = {

		'MAX_DIST_ERROR': config['Foci filt max_dist_err'],
		'SIG_TO_SIGRAW' : config['Foci filt max_sig_to_sigraw'],
		'MAX_DELTA_AREA': config['Foci filt max_delta_area'],
		'TRAJ_LEN_THRES': config['Foci filt traj_length_thres'],

		}

		#DICT
		self.DICT = config.copy()


	def save_config(self):

		path = self.OUTPUT_PATH + self.ROOT_NAME + '-analMeta.csv'
		config_df = pd.DataFrame.from_dict(data=self.DICT, orient='index')
		config_df = config_df.drop(['IO input_path', 'IO output_path',
									'Processed By:'])
		config_df.to_csv(path, header=False)

	def clean_dir(self):

		flist = [f for f in os.listdir(self.OUTPUT_PATH)]
		for f in flist:
		    os.remove(os.path.join(self.OUTPUT_PATH, f))

def nonempty_exists_then_copy(input_path, output_path, filename):
	not_empty = len(filename)!=0
	exists_in_input = osp.exists(input_path + filename)

	if not_empty and exists_in_input:
		frames = imread(input_path + filename)
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		imsave(output_path + filename, frames)


def file1_exists_or_pimsopen_file2(head_str, tail_str1, tail_str2):
	if osp.exists(head_str + tail_str1):
		frames = pims.open(head_str + tail_str1)
	else:
		frames = pims.open(head_str + tail_str2)
	return frames


def nonempty_openfile1_or_openfile2(path, filename1, filename2):
	if filename1 and osp.exists(path + filename1): # if not empty and exists
		frames = imread(path + filename1)
	else:
		frames = imread(path + filename2)
	return frames


class Pipeline3():

	def __init__(self, config):
		self.config = config

	def clean_dir(self):
		self.config.clean_dir()

	def load(self):
		# load data file
		if osp.exists(self.config.INPUT_PATH + self.config.ROOT_NAME + '.tif'):
			frames = imread(self.config.INPUT_PATH + self.config.ROOT_NAME + '.tif')
		else:
			frames = imread(self.config.INPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif', frames)

		print('\tMask_dist2boundary_file: [%s]' % self.config.DIST2BOUNDARY_MASK_NAME)
		print('\tMask_dist253bp1_file: [%s]' % self.config.DIST253BP1_MASK_NAME)
		print('\tMask_dist253bp1_file: [%s]' % self.config.MASK_53BP1_BLOB_NAME)
		# load reference files
		nonempty_exists_then_copy(self.config.INPUT_PATH, self.config.OUTPUT_PATH, self.config.DIST2BOUNDARY_MASK_NAME)
		nonempty_exists_then_copy(self.config.INPUT_PATH, self.config.OUTPUT_PATH, self.config.DIST253BP1_MASK_NAME)
		nonempty_exists_then_copy(self.config.INPUT_PATH, self.config.OUTPUT_PATH, self.config.MASK_53BP1_BLOB_NAME)

	def get_boundary_mask(self):
		# If no mask ref file, use raw file automatically
		frames = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.DIST2BOUNDARY_MASK_NAME,
					self.config.ROOT_NAME+'-raw.tif')

		# If only 1 frame available, duplicate it to enough frames_num.
		tot_frame_num = len(imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif'))
		if frames.ndim==2:
			dup_frames = np.zeros((tot_frame_num, frames.shape[0], frames.shape[1]),
									dtype=frames.dtype)
			for i in range(tot_frame_num):
				dup_frames[i] = frames
			frames = dup_frames

		boundary_masks = get_thres_mask_batch(frames,
					self.config.MASK_SIG_BOUNDARY, self.config.MASK_THRES_BOUNDARY)

		return boundary_masks


	def get_53bp1_mask(self):
		# If no mask ref file, use raw file automatically
		frames = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.DIST253BP1_MASK_NAME,
					self.config.ROOT_NAME+'-raw.tif')

		# If only 1 frame available, duplicate it to enough frames_num.
		tot_frame_num = len(imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif'))
		if frames.ndim==2:
			dup_frames = np.zeros((tot_frame_num, frames.shape[0], frames.shape[1]),
									dtype=frames.dtype)
			for i in range(tot_frame_num):
				dup_frames[i] = frames
			frames = dup_frames

		# Get mask file and save it using 255 and 0
		masks_53bp1 = get_thres_mask_batch(frames,
							self.config.MASK_SIG_53BP1, self.config.MASK_THRES_53BP1)

		return masks_53bp1


	def get_53bp1_blob_mask(self):
		# If no mask ref file, use raw file automatically
		frames = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.MASK_53BP1_BLOB_NAME,
					self.config.ROOT_NAME+'-raw.tif')

		# If only 1 frame available, duplicate it to enough frames_num.
		tot_frame_num = len(imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif'))
		if frames.ndim==2:
			dup_frames = np.zeros((tot_frame_num, frames.shape[0], frames.shape[1]),
									dtype=frames.dtype)
			for i in range(tot_frame_num):
				dup_frames[i] = frames
			frames = dup_frames

		# Get mask file and save it using 255 and 0
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-tempFile.tif',
				frames)
		pims_frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME +
								'-tempFile.tif')

		blobs_df, det_plt_array = detect_blobs(pims_frames[0],
									min_sig=self.config.MASK_53BP1_BLOB_MINSIG,
									max_sig=self.config.MASK_53BP1_BLOB_MAXSIG,
									num_sig=self.config.MASK_53BP1_BLOB_NUMSIG,
									blob_thres=self.config.MASK_53BP1_BLOB_THRES,
									peak_thres_rel=self.config.MASK_53BP1_BLOB_PKTHRES_REL,
									r_to_sigraw=1.4,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=True,
									pltshow=True,
									plot_r=False,
									truth_df=None)

		blobs_df, det_plt_array = detect_blobs_batch(pims_frames,
									min_sig=self.config.MASK_53BP1_BLOB_MINSIG,
									max_sig=self.config.MASK_53BP1_BLOB_MAXSIG,
									num_sig=self.config.MASK_53BP1_BLOB_NUMSIG,
									blob_thres=self.config.MASK_53BP1_BLOB_THRES,
									peak_thres_rel=self.config.MASK_53BP1_BLOB_PKTHRES_REL,
									r_to_sigraw=1.4,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=False,
									pltshow=False,
									plot_r=False,
									truth_df=None)


		blobs_df = tp.link_df(blobs_df,
									search_range=self.config.MASK_53BP1_BLOB_SEARCH_RANGE,
									memory=self.config.MASK_53BP1_BLOB_MEMORY)
		blobs_df = tp.filter_stubs(blobs_df, self.config.MASK_53BP1_BLOB_TRAJ_LENGTH_THRES)
		blobs_df = blobs_df.reset_index(drop=True)

		masks_53bp1_blob = blobs_df_to_mask(frames, blobs_df)

		os.remove(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-tempFile.tif')

		return masks_53bp1_blob

	def mask_boundary(self):
		print("######################################")
		print("Generate mask_boundary")
		print("######################################")
		boundary_masks = self.get_boundary_mask()
		# Save it using 255 and 0
		boundary_masks_255 = np.rint(boundary_masks / \
							boundary_masks.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif',
				boundary_masks_255)

		# print("######################################")
		# print("Generate dist2boundary_mask")
		# print("######################################")
		# dist2boundary_masks = img_as_int(get_dist2boundary_mask_batch(boundary_masks))
		# imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist2boundaryMask.tif',
		# 		dist2boundary_masks)


	def mask_53bp1(self):
		print("######################################")
		print("Generate mask_53bp1")
		print("######################################")
		masks_53bp1 = self.get_53bp1_mask()
		# Save it using 255 and 0
		masks_53bp1_255 = np.rint(masks_53bp1 / \
							masks_53bp1.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1Mask.tif',
				masks_53bp1_255)

		print("######################################")
		print("Generate dist253bp1_mask")
		print("######################################")
		dist253bp1_masks = img_as_int(get_dist2boundary_mask_batch(masks_53bp1))
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist253bp1Mask.tif',
				dist253bp1_masks)


	def mask_53bp1_blob(self):
		print("######################################")
		print("Generate mask_53bp1_blob")
		print("######################################")
		masks_53bp1_blob = self.get_53bp1_blob_mask()
		# Save it using 255 and 0
		masks_53bp1_blob_255 = np.rint(masks_53bp1_blob / \
							masks_53bp1_blob.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1BlobMask.tif',
				masks_53bp1_blob_255)

		print("######################################")
		print("Generate dist253bp1blob_mask")
		print("######################################")
		dist253bp1blob_masks = img_as_int(get_dist2boundary_mask_batch(masks_53bp1_blob))
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist253bp1blobMask.tif',
				dist253bp1blob_masks)



	def foci_denoise(self):

		print("######################################")
		print('Applying Boxcar Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')
		frames = img_as_ubyte(frames)
		filtered = filter_batch(frames, method='boxcar', arg=self.config.BOXCAR_RADIUS)
		filtered = filter_batch(filtered, method='gaussian', arg=self.config.GAUS_BLUR_SIG)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fociDeno.tif', filtered)


	def check_foci_detection(self):

		print("######################################")
		print("Check foci detection")
		print("######################################")

		check_frame_ind = [0, 50, 100, 144]

		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')

		for ind in check_frame_ind:
			blobs_df, det_plt_array = detect_blobs(frames[ind],
										min_sig=self.config.MIN_SIGMA,
										max_sig=self.config.MAX_SIGMA,
										num_sig=self.config.NUM_SIGMA,
										blob_thres_rel=self.config.THRESHOLD,
										peak_min=self.config.PEAK_MIN,
										num_peaks=self.config.NUM_PEAKS,
										peak_thres_rel=self.config.PEAK_THRESH_REL,
										mass_thres_rel=self.config.MASS_THRESH_REL,
										peak_r_rel=self.config.PEAK_R_REL,
										mass_r_rel=self.config.MASS_R_REL,
										r_to_sigraw=1,
										pixel_size=self.config.PIXEL_SIZE,
										diagnostic=True,
										pltshow=True,
										blob_markersize=5,
										plot_r=False,
										truth_df=None)

	def detect_foci(self):

		print("######################################")
		print("Detect foci")
		print("######################################")

		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames,
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres_rel=self.config.THRESHOLD,
									peak_min=self.config.PEAK_MIN,
									num_peaks=self.config.NUM_PEAKS,
									peak_thres_rel=self.config.PEAK_THRESH_REL,
									mass_thres_rel=self.config.MASS_THRESH_REL,
									peak_r_rel=self.config.PEAK_R_REL,
									mass_r_rel=self.config.MASS_R_REL,
									r_to_sigraw=1,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=False,
									pltshow=False,
									blob_markersize=5,
									plot_r=False,
									truth_df=None)

		det_plt_array = anim_blob(blobs_df, frames,
									pixel_size=self.config.PIXEL_SIZE,
									blob_markersize=5,
									)
		try:
			imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detVideo.tif', det_plt_array)
		except:
			pass

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-detData.csv', index=False)

		self.config.save_config()


	def fit(self):

		print("######################################")
		print("Fit")
		print("######################################")

		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detData.csv')

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fociDeno.tif')

		psf_df, fit_plt_array = fit_psf_batch(frames_deno,
		            blobs_df,
		            diagnostic=False,
		            pltshow=False,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=None)

		psf_df = psf_df.apply(pd.to_numeric)
		psf_df['slope'] = psf_df['A'] / (9 * np.pi * psf_df['sig_x'] * psf_df['sig_y'])
		psf_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-fittData.csv', index=False)


	# helper function for filt and track()
	def track_blobs_twice(self):
		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')
		psf_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fittData.csv')


		blobs_df, im = track_blobs(psf_df,
								    search_range=self.config.SEARCH_RANGE,
									memory=self.config.MEMORY,
									pixel_size=self.config.PIXEL_SIZE,
									frame_rate=self.config.FRAME_RATE,
									divide_num=self.config.DIVIDE_NUM,
									filters=None,
									do_filter=False)

		if self.config.DO_FILTER:
			blobs_df, im = track_blobs(blobs_df,
									    search_range=self.config.SEARCH_RANGE,
										memory=self.config.MEMORY,
										pixel_size=self.config.PIXEL_SIZE,
										frame_rate=self.config.FRAME_RATE,
										divide_num=self.config.DIVIDE_NUM,
										filters=self.config.FILTERS,
										do_filter=True)

		# Add 'traj_length' column and save physData before traj_length_thres filter
		blobs_df = add_traj_length(blobs_df)

		return blobs_df


	# helper function for filt and track()
	def print_filt_traj_num(self, blobs_df):
		traj_num_before = blobs_df['particle'].nunique()
		after_filter_df = blobs_df [blobs_df['traj_length'] >= self.config.FILTERS['TRAJ_LEN_THRES']]
		print("######################################")
		print("Trajectory number before filters: \t%d" % traj_num_before)
		print("Trajectory number after filters: \t%d" % after_filter_df['particle'].nunique())
		print("######################################")


	# helper function for filt and track()
	def filt_phys_df(self, phys_df):

		df = phys_df.copy()
		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in df:
			df = df[ df['traj_length']>=self.config.FILTERS['TRAJ_LEN_THRES'] ]

		return df

	def filt_track(self):

		print("######################################")
		print("Filter and Linking")
		print("######################################")

		check_search_range = isinstance(self.config.SEARCH_RANGE, list)
		if check_search_range:
			param_list = self.config.SEARCH_RANGE
			particle_num_list = []
			phys_dfs = []
			mean_D_list = []
			mean_alpha_list = []
			for search_range in param_list:
				self.config.SEARCH_RANGE = search_range
				phys_df = self.track_blobs_twice()
				self.print_filt_traj_num(phys_df)
				phys_df = self.filt_phys_df(phys_df)
				phys_df = phys_df.drop_duplicates('particle')
				phys_df['search_range'] = search_range
				phys_dfs.append(phys_df)
				particle_num_list.append(len(phys_df))
				mean_D_list.append(phys_df['D'].mean())
				mean_alpha_list.append(phys_df['alpha'].mean())
			phys_df_all = pd.concat(phys_dfs)
			sr_opt_fig = plot_track_param_opt(
							track_param_name='search_range',
							track_param_unit='pixel',
							track_param_list=param_list,
							particle_num_list=particle_num_list,
							df=phys_df_all,
							mean_D_list=mean_D_list,
							mean_alpha_list=mean_alpha_list,
							)
			sr_opt_fig.savefig(self.config.OUTPUT_PATH + \
							self.config.ROOT_NAME + '-opt-search-range.pdf')


		check_traj_len_thres = isinstance(self.config.FILTERS['TRAJ_LEN_THRES'], list)
		if check_traj_len_thres:
			param_list = self.config.FILTERS['TRAJ_LEN_THRES']
			particle_num_list = []
			phys_dfs = []
			mean_D_list = []
			mean_alpha_list = []

			if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv'):
				original_phys_df = pd.read_csv(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-physData.csv')
			else:
				original_phys_df = self.track_blobs_twice()

			for traj_len_thres in param_list:
				self.config.FILTERS['TRAJ_LEN_THRES'] = traj_len_thres
				self.print_filt_traj_num(original_phys_df)
				phys_df = self.filt_phys_df(original_phys_df)
				phys_df = phys_df.drop_duplicates('particle')
				phys_df['traj_len_thres'] = traj_len_thres
				phys_dfs.append(phys_df)
				particle_num_list.append(len(phys_df))
				mean_D_list.append(phys_df['D'].mean())
				mean_alpha_list.append(phys_df['alpha'].mean())
			phys_df_all = pd.concat(phys_dfs)
			sr_opt_fig = plot_track_param_opt(
							track_param_name='traj_len_thres',
							track_param_unit='frame',
							track_param_list=param_list,
							particle_num_list=particle_num_list,
							df=phys_df_all,
							mean_D_list=mean_D_list,
							mean_alpha_list=mean_alpha_list,
							)
			sr_opt_fig.savefig(self.config.OUTPUT_PATH + \
							self.config.ROOT_NAME + '-opt-traj-len-thres.pdf')

		else:
			blobs_df = self.track_blobs_twice()
			self.print_filt_traj_num(blobs_df)
			blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
										'-physData.csv', index=False)

		self.config.save_config()

	def plot_traj(self):
		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.FILTERS['TRAJ_LEN_THRES']!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.FILTERS['TRAJ_LEN_THRES'] ]


		# """
		# ~~~~~~~~Optimize the colorbar format~~~~~~~~
		# """
		if len(phys_df.drop_duplicates('particle')) > 1:
			D_max = phys_df['D'].quantile(0.9)
			D_min = phys_df['D'].quantile(0.1)
			D_range = D_max - D_min
			cb_min=D_min
			cb_max=D_max
			cb_major_ticker=round(0.2*D_range)
			cb_minor_ticker=round(0.2*D_range)
		else:
			cb_min, cb_max, cb_major_ticker, cb_minor_ticker = None, None, None, None


		fig, ax = plt.subplots()
		anno_traj(ax, phys_df,

					show_image=True,
					image = frames[0],

					show_scalebar=True,
					pixel_size=self.config.PIXEL_SIZE,

					show_colorbar=True,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

		            show_traj_num=True,

					show_particle_label=False,

					# show_boundary=True,
					# boundary_mask=boundary_masks[0],
					# boundary_list=self.config.DICT['Sort dist_to_boundary'],
					)
		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-results.pdf')
		plt.clf(); plt.close()
		# plt.show()

		self.config.save_config()

	def phys(self):

		print("######################################")
		print("Add Physics Parameters")
		print("######################################")
		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		self.phys_dist2boundary()
		self.phys_dist253bp1()
		self.phys_dist253bp1_blob()

		# Save '-physData.csv'
		phys_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def phys_dist2boundary(self):
		print("######################################")
		print("Add Physics Param: dist_to_boundary")
		print("######################################")
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist2boundaryMask.tif'):
			dist2boundary_masks = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist2boundaryMask.tif')
		else:
			boundary_masks = self.get_boundary_mask()
			dist2boundary_masks = get_dist2boundary_mask_batch(boundary_masks)

		phys_df = add_dist_to_boundary_batch_2(phys_df, dist2boundary_masks)
		phys_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def phys_dist253bp1(self):
		print("######################################")
		print("Add Physics Param: dist_to_53bp1")
		print("######################################")
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist253bp1Mask.tif'):
			dist253bp1_masks = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist253bp1Mask.tif')
		else:
			masks_53bp1 = self.get_53bp1_mask()
			dist253bp1_masks = get_dist2boundary_mask_batch(masks_53bp1)

		phys_df = add_dist_to_boundary_batch_2(phys_df, dist253bp1_masks, col_name='dist_to_53bp1')
		phys_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def phys_dist253bp1_blob(self):
		print("######################################")
		print("Add Physics Param: dist_to_53bp1_blob")
		print("######################################")
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist253bp1blobMask.tif'):
			dist253bp1blob_masks = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist253bp1blobMask.tif')
		else:
			masks_53bp1blob = self.get_53bp1_blob_mask()
			dist253bp1blob_masks = get_dist2boundary_mask_batch(masks_53bp1blob)

		phys_df = add_dist_to_boundary_batch_2(phys_df, dist253bp1blob_masks, col_name='dist_to_53bp1blob')
		phys_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def merge_plot(self):

		today = str(date.today().strftime("%y%m%d"))

		print("######################################")
		print("Merge and PlotMSD")
		print("######################################")

		merged_files = np.array(sorted(glob(self.config.OUTPUT_PATH + '/*physDataMerged.csv')))
		print(merged_files)

		if len(merged_files) > 1:
			print("######################################")
			print("Found multiple physDataMerged file!!!")
			print("######################################")
			return

		if len(merged_files) == 1:
			phys_df = pd.read_csv(merged_files[0])

		else:
			phys_files = np.array(sorted(glob(self.config.OUTPUT_PATH + '/*physData.csv')))
			print("######################################")
			print("Total number of physData to be merged: %d" % len(phys_files))
			print("######################################")
			print(phys_files)

			if len(phys_files) > 1:
				ind = 1
				tot = len(phys_files)
				for file in phys_files:
					print("Updating fittData (%d/%d)" % (ind, tot))
					ind = ind + 1

					curr_physdf = pd.read_csv(file, index_col=False)
					if 'traj_length' not in curr_physdf:
						curr_physdf = add_traj_length(curr_physdf)
						curr_physdf.round(3).to_csv(file, index=False)

				phys_df = merge_physdfs(phys_files, mode='general')

			else:
				phys_df = pd.read_csv(phys_files[0])


			print("######################################")
			print("Rename particles...")
			print("######################################")
			phys_df['particle'] = phys_df['raw_data'] + phys_df['particle'].apply(str)
			phys_df.round(3).to_csv(self.config.OUTPUT_PATH + today + \
							'-physDataMerged.csv', index=False)

		# Apply traj_length_thres filter
		if 'traj_length' in phys_df:
			phys_df = phys_df[ phys_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES'] ]

		fig_quick_merge(phys_df)


		sys.exit()


def get_root_name_list(settings_dict):
	# Make a copy of settings_dict
	# Use '*%#@)9_@*#@_@' to substitute if the labels are empty
	settings = settings_dict.copy()
	if settings['Mask boundary_mask file label'] == '':
		settings['Mask boundary_mask file label'] = '*%#@)9_@*#@_@'
	if settings['Mask 53bp1_mask file label'] == '':
		settings['Mask 53bp1_mask file label'] = '*%#@)9_@*#@_@'
	if settings['Mask 53bp1_blob_mask file label'] == '':
		settings['Mask 53bp1_blob_mask file label'] = '*%#@)9_@*#@_@'

	root_name_list = []

	path_list = glob(settings['IO input_path'] + '/*-physData.csv')
	if len(path_list) != 0:
		for path in path_list:
			temp = path.split('/')[-1]
			temp = temp[:-len('-physData.csv')]
			root_name_list.append(temp)

	else:
		path_list = glob(settings['IO input_path'] + '/*-raw.tif')
		if len(path_list) != 0:
			for path in path_list:
				temp = path.split('/')[-1]
				temp = temp[:-4 - len('-raw')]
				root_name_list.append(temp)
		else:
			path_list = glob(settings['IO input_path'] + '/*.tif')
			for path in path_list:
				temp = path.split('/')[-1]
				temp = temp[:-4]
				if (settings['Mask boundary_mask file label'] not in temp+'.tif') and \
					(settings['Mask 53bp1_mask file label'] not in temp+'.tif') and \
					(settings['Mask 53bp1_blob_mask file label'] not in temp+'.tif'):
					root_name_list.append(temp)

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


def pipeline_batch(settings_dict, control_list):

	# """
	# ~~~~~~~~~~~~~~~~~1. Get root_name_list~~~~~~~~~~~~~~~~~
	# """
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

		# """
		# ~~~~~~~~~~~~~~~~~2. Update config~~~~~~~~~~~~~~~~~
		# """

		config = Config(settings_dict)

		# 2.0. If LOAD_ANALMETA==True, then load existing analMeta file, if there is one
		if osp.exists(settings_dict['IO input_path'] + root_name + '-analMeta.csv'):
			existing_settings = analMeta_to_dict(settings_dict['IO input_path'] + root_name + '-analMeta.csv')
			existing_settings['IO input_path']= settings_dict['IO input_path']
			existing_settings['IO output_path'] = settings_dict['IO output_path']
			existing_settings['Processed By:'] = settings_dict['Processed By:']
			existing_settings.pop('Processed by:', None)
			settings_dict = existing_settings
			config = Config(settings_dict)

		# 2.1. Update config.ROOT_NAME and config.DICT
		config.ROOT_NAME = root_name
		config.DICT['Raw data file'] = root_name + '.tif'
		config.DICT['Processed date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		config.DICT['Processed by:'] = settings_dict['Processed By:']

		if '-' in root_name and root_name.find('-')>0:
			key = root_name[0:root_name.find('-')]
		else:
			key = root_name


		# 2.3. Update config.DIST2BOUNDARY_MASK_NAME
		if settings_dict['Mask boundary_mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Mask boundary_mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DIST2BOUNDARY_MASK_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Mask boundary_mask file label'] = ''

		# 2.4. Update config.DIST253BP1_MASK_NAME
		if settings_dict['Mask 53bp1_mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Mask 53bp1_mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DIST253BP1_MASK_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Mask 53bp1_mask file label'] = ''

		# 2.5. Update config.MASK_53BP1_BLOB_NAME
		if settings_dict['Mask 53bp1_blob_mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Mask 53bp1_blob_mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.MASK_53BP1_BLOB_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Mask 53bp1_blob_mask file label'] = ''

		# """
		# ~~~~~~~~~~~~~~~~~3. Setup pipe and run~~~~~~~~~~~~~~~~~
		# """
		pipe = Pipeline3(config)
		for func in control_list:
			getattr(pipe, func)()
