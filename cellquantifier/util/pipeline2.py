import pims; import pandas as pd; import numpy as np
import trackpy as tp
import os.path as osp; import os
from datetime import date
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
import warnings
import glob

from ..deno import filter_batch
from ..segm import get_mask_batch
from ..segm import get_thres_mask_batch
from ..regi import get_regi_params, apply_regi_params

from ..smt.detect import detect_blobs, detect_blobs_batch
from ..smt.fit_psf import fit_psf, fit_psf_batch
from ..smt.track import track_blobs
from ..smt.msd import plot_msd_batch, get_sorter_list
from ..phys import *
from ..util.config import Config
from ..plot import plot_phys_1 as plot_merged
from ..phys.physutil import relabel_particles, merge_physdfs

class Pipeline2():

	def __init__(self, config):
		self.config = config


	def clean_dir(self):
		self.config.clean_dir()


	def load(self):
		frames = imread(self.config.INPUT_PATH + self.config.DICT['Raw data file'])
		frames = frames[list(self.config.TRANGE),:,:]
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif', frames)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', frames)


	def mask(self):

		print("######################################")
		print("Generate dist2boundary_thres_masks")
		print("######################################")
		dist2boundary_tif = imread(self.config.INPUT_PATH + \
							self.config.DIST2BOUNDARY_MASK_NAME) \
							[list(self.config.TRANGE),:,:]

		dist2boundary_thres_masks = get_thres_mask_batch(dist2boundary_tif,
							self.config.MASK_SIG_BOUNDARY, self.config.MASK_THRES_BOUNDARY)
		dist2boundary_thres_masks = np.rint(dist2boundary_thres_masks / \
							dist2boundary_thres_masks.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif',
				dist2boundary_thres_masks)

		print("######################################")
		print("Generate dist253bp1_thres_masks")
		print("######################################")
		dist253bp1_tif = imread(self.config.INPUT_PATH + \
							self.config.DIST253BP1_MASK_NAME) \
							[list(self.config.TRANGE),:,:]

		dist253bp1_thres_masks = get_thres_mask_batch(dist253bp1_tif,
							self.config.MASK_SIG_53BP1, self.config.MASK_THRES_53BP1)
		dist253bp1_thres_masks = np.rint(dist253bp1_thres_masks / \
							dist253bp1_thres_masks.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1Mask.tif',
				dist253bp1_thres_masks)

		print("######################################")
		print("Mask generation are done!")
		print("######################################")


	def segmentation(self, method):

		print("######################################")
		print("Segmenting Image Stack")
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		mask, centroid = get_mask_batch(frames, method, min_size=self.config.MIN_SIZE,show_mask=self.config.PLTSHOW)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-mask.tif', mask)

		return centroid


	def regi(self):

		print("######################################")
		print("Registering Image Stack")
		print("######################################")

		im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')

		regi_params_array_2d = get_regi_params(im,
		              ref_ind_num=self.config.REF_IND_NUM,
		              sig_mask=self.config.SIG_MASK,
		              thres_rel=self.config.THRES_REL,
		              poly_deg=self.config.POLY_DEG,
		              rotation_multplier=self.config.ROTATION_MULTIPLIER,
		              translation_multiplier=self.config.TRANSLATION_MULTIPLIER,
		              diagnostic=False)

		registered = apply_regi_params(im, regi_params_array_2d)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif', registered)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', registered)


	def deno_gaus(self):

		print("######################################")
		print('Applying Gaussian Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		filtered = filter_batch(frames, method='gaussian', arg=self.config.GAUS_BLUR_SIG)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', filtered)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-deno.tif', filtered)

	def deno_box(self):

		print("######################################")
		print('Applying Boxcar Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		filtered = filter_batch(frames, method='boxcar', arg=self.config.BOXCAR_RADIUS)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', filtered)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-deno.tif', filtered)


	def check(self):

		print("######################################")
		print("Check detection and fitting")
		print("######################################")

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')

		blobs_df, det_plt_array = detect_blobs(frames[self.config.CHECK_FRAME],
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres=self.config.THRESHOLD,
									peak_thres_rel=self.config.PEAK_THRESH_REL,
									r_to_sigraw=self.config.PATCH_SIZE,
									pixel_size = self.config.PIXEL_SIZE,
									diagnostic=True,
									pltshow=True,
									plot_r=self.config.PLOT_R,
									truth_df=None)

		psf_df, fit_plt_array = fit_psf(frames_deno[self.config.CHECK_FRAME],
		            blobs_df,
		            diagnostic=True,
		            pltshow=True,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=blobs_df)


	def detect_fit(self):

		print("######################################")
		print("Detect, Fit")
		print("######################################")

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames,
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres=self.config.THRESHOLD,
									peak_thres_rel=self.config.PEAK_THRESH_REL,
									r_to_sigraw=self.config.PATCH_SIZE,
									pixel_size = self.config.PIXEL_SIZE,
									diagnostic=True,
									pltshow=False,
									plot_r=False,
									truth_df=None)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detVideo.tif', det_plt_array)

		psf_df, fit_plt_array = fit_psf_batch(frames_deno,
		            blobs_df,
		            diagnostic=False,
		            pltshow=False,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=None,
					output_path=self.config.OUTPUT_PATH,
					root_name=self.config.ROOT_NAME,
					save_csv=True)

	def filt_track(self):

		print("######################################")
		print("Filter and Linking")
		print("######################################")

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		psf_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fittData.csv')

		blobs_df, im = track_blobs(psf_df,
								    search_range=self.config.SEARCH_RANGE,
									memory=self.config.MEMORY,
									pixel_size=self.config.PIXEL_SIZE,
									frame_rate=self.config.FRAME_RATE,
									divide_num=self.config.DIVIDE_NUM,
									filters=None,
									do_filter=False)

		traj_num_before = blobs_df['particle'].nunique()

		if self.config.DO_FILTER:
			blobs_df, im = track_blobs(blobs_df,
									    search_range=self.config.SEARCH_RANGE,
										memory=self.config.MEMORY,
										pixel_size=self.config.PIXEL_SIZE,
										frame_rate=self.config.FRAME_RATE,
										divide_num=self.config.DIVIDE_NUM,
										filters=self.config.FILTERS,
										do_filter=True)

		blobs_df.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		print("######################################")
		print("Trajectory number before filters: \t%d" % traj_num_before)
		print("Trajectory number after filters: \t%d" % blobs_df['particle'].nunique())
		print("######################################")


	def phys(self):

		print("######################################")
		print("Add Physics Parameters")
		print("######################################")

		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		# add 'dist_to_boundary' column
		print("######################################")
		print("Add 'dist_to_boundary'")
		print("######################################")
		dist2boundary_tif = imread(self.config.INPUT_PATH + \
							self.config.DIST2BOUNDARY_MASK_NAME) \
							[list(self.config.TRANGE),:,:]

		dist2boundary_thres_masks = get_thres_mask_batch(dist2boundary_tif,
							self.config.MASK_SIG_BOUNDARY, self.config.MASK_THRES_BOUNDARY)
		phys_df = add_dist_to_boundary_batch(blobs_df, dist2boundary_thres_masks)

		# add 'dist_to_boundary' column
		print("######################################")
		print("Add 'dist_to_53bp1'")
		print("######################################")
		dist253bp1_tif = imread(self.config.INPUT_PATH + \
							self.config.DIST253BP1_MASK_NAME) \
							[list(self.config.TRANGE),:,:]

		dist253bp1_thres_masks = get_thres_mask_batch(dist253bp1_tif,
							self.config.MASK_SIG_53BP1, self.config.MASK_THRES_53BP1)
		phys_df = add_dist_to_53bp1_batch(blobs_df, dist253bp1_thres_masks)

		# Save '-physData.csv'
		phys_df.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def sort_plot(self):

		print("######################################")
		print("Sort and PlotMSD")
		print("######################################")

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		if self.config.DO_SORT:
			phys_df = sort_phys(phys_df, self.config.SORTERS)
			phys_df.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)
		else:
			sorter_list = get_sorter_list(phys_df)
			phys_df = phys_df.drop(columns=sorter_list[1:-1])

		plot_msd_batch(phys_df,
					 image=frames[0],
					 output_path=self.config.OUTPUT_PATH,
					 root_name=self.config.ROOT_NAME,
					 pixel_size=self.config.PIXEL_SIZE,
					 frame_rate=self.config.FRAME_RATE,
					 divide_num=self.config.DIVIDE_NUM,
					 plot_without_sorter=False,
					 show_fig=False,
					 save_pdf=True,
					 open_pdf=False)

		self.config.save_config()

	def merge_plot(self):

		start_ind = self.config.ROOT_NAME.find('_')
		end_ind = self.config.ROOT_NAME.find('_', start_ind+1)
		today = str(date.today().strftime("%y%m%d"))
		merged_name = today + self.config.ROOT_NAME[start_ind:end_ind]

		print("######################################")
		print("Merge and PlotMSD")
		print("######################################")

		merged_files = np.array(sorted(glob.glob(self.config.OUTPUT_PATH + '/*physDataMerged.csv')))
		print(merged_files)
		if len(merged_files) > 1:
			print("######################################")
			print("Found multiple physDataMerged file!!!")
			print("######################################")
			return

		if len(merged_files) == 1:
			phys_df = pd.read_csv(merged_files[0])

		else:
			phys_files = np.array(sorted(glob.glob(self.config.OUTPUT_PATH + '/*physData.csv')))
			print(phys_files)

			if len(phys_files) > 1:
				phys_df = merge_physdfs(phys_files)
				phys_df = relabel_particles(phys_df)
			else:
				phys_df = pd.read_csv(phys_files[0])

			phys_df.to_csv(self.config.OUTPUT_PATH + merged_name + \
							'-physDataMerged.csv', index=False)

		# phys_df = phys_df.loc[phys_df['exp_label'] == 'BLM']
		fig = plot_merged(phys_df, 'exp_label',
						pixel_size=self.config.PIXEL_SIZE,
						frame_rate=self.config.FRAME_RATE,
						divide_num=self.config.DIVIDE_NUM,
						RGBA_alpha=0.5,
						do_gmm=False)

		fig.savefig(self.config.OUTPUT_PATH + merged_name + '-mergedResults.pdf')


def pipeline_control2(settings_dict, control_list):
	config = Config(settings_dict)
	pipe = Pipeline2(config)

	for func in control_list:
		getattr(pipe, func)()
