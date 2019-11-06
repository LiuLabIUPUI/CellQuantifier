import pims
import os.path as osp; import os
from skimage.io import imread, imsave
import pandas as pd

from ..deno import filter_batch
from ..segm import get_mask_batch
from ..regi import get_regi_params, apply_regi_params

from ..smt.detect import detect_blobs, detect_blobs_batch
from ..smt.fit_psf import fit_psf, fit_psf_batch
from ..smt.track import track_blobs
from ..smt.msd import plot_msd

class Pipeline():

	def __init__(self, config):

			frames = imread(config.INPUT_PATH)
			frames = frames[list(config.TRANGE),:,:]

			a = config.OUTPUT_PATH + config.ROOT_NAME + '-raw.tif'
			b = config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif'

			imsave(a, frames)
			imsave(b, frames)

			self.config = config


	def segmentation(self, method):

		print("######################################")
		print("Segmenting Image Stack")
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		mask, centroid = get_mask_batch(frames, method, min_size=self.config.MIN_SIZE,show_mask=self.config.PLTSHOW)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-mask.tif', mask)

		return centroid


	def register(self):

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
		              diagnostic=True)

		registered = apply_regi_params(im, regi_params_array_2d)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif', registered)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', registered)


	def deno(self, method, arg):

		print("######################################")
		print('Applying ' + method.capitalize() + ' Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		filtered = filter_batch(frames, method=method, arg=arg)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', filtered)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-deno.tif', filtered)


	def check_start_frame(self):

		print("######################################")
		print("Check Start Frame")
		print("######################################")

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')

		blobs_df, det_plt_array = detect_blobs(frames[self.config.START_FRAME],
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

		psf_df, fit_plt_array = fit_psf(frames_deno[self.config.START_FRAME],
		            blobs_df,
		            diagnostic=True,
		            pltshow=True,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=blobs_df,
					centroid=None)


	def detect_fit_link(self):

		print("######################################")
		print("Detect, Fit, Linking")
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
									diagnostic=False,
									pltshow=False,
									plot_r=self.config.PLOT_R,
									truth_df=None)

		psf_df, fit_plt_array = fit_psf_batch(frames_deno,
		            blobs_df,
		            diagnostic=False,
		            pltshow=False,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=None,
					centroid=None)

		blobs_df, im = track_blobs(psf_df,
								    search_range=self.config.SEARCH_RANGE,
									memory=self.config.MEMORY,
									min_traj_length=self.config.MIN_TRAJ_LENGTH,
									filters=self.config.FILTERS,
									pixel_size=self.config.PIXEL_SIZE,
									frame_rate=self.config.FRAME_RATE,
									divide_num=self.config.DIVIDE_NUM,
									do_filter=False,
									output_path=self.config.OUTPUT_PATH,
									root_name=self.config.ROOT_NAME)


	def filter_and_plotmsd(self):

		print("######################################")
		print("Filter and PlotMSD")
		print("######################################")

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		psf_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fittData.csv')

		blobs_df, im = track_blobs(psf_df,
								    search_range=self.config.SEARCH_RANGE,
									memory=self.config.MEMORY,
									min_traj_length=self.config.MIN_TRAJ_LENGTH,
									filters=self.config.FILTERS,
									pixel_size=self.config.PIXEL_SIZE,
									frame_rate=self.config.FRAME_RATE,
									divide_num=self.config.DIVIDE_NUM,
									do_filter=True,
									output_path=self.config.OUTPUT_PATH,
									root_name=self.config.ROOT_NAME,
									save_csv=False)
		d, alpha = plot_msd(im,
		            		 blobs_df,
		            		 image=frames[0],
		            		 output_path=self.config.OUTPUT_PATH,
		            		 root_name=self.config.ROOT_NAME,
		            		 pixel_size=self.config.PIXEL_SIZE,
		            		 divide_num=self.config.DIVIDE_NUM)

		self.config.save_config()
		os.remove(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')


	def smt(self, centroid=None):

		print("######################################")
		print("Single Molecule Tracking")
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
									diagnostic=self.config.DIAGNOSTIC,
									pltshow=self.config.PLTSHOW,
									plot_r=self.config.PLOT_R,
									truth_df=None)

		psf_df, fit_plt_array = fit_psf_batch(frames_deno,
		            blobs_df,
		            diagnostic=self.config.DIAGNOSTIC,
		            pltshow=self.config.PLTSHOW,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=None,
					centroid=centroid)

		blobs_df, im = track_blobs(psf_df,
								    search_range=self.config.SEARCH_RANGE,
									memory=self.config.MEMORY,
									min_traj_length=self.config.MIN_TRAJ_LENGTH,
									filters=self.config.FILTERS,
									pixel_size=self.config.PIXEL_SIZE,
									frame_rate=self.config.FRAME_RATE,
									divide_num=self.config.DIVIDE_NUM,
									do_filter=self.config.DO_FILTER,
									output_path=self.config.OUTPUT_PATH,
									root_name=self.config.ROOT_NAME)

		d, alpha = plot_msd(im,
		            		 blobs_df,
		            		 image=frames[0],
		            		 output_path=self.config.OUTPUT_PATH,
		            		 root_name=self.config.ROOT_NAME,
		            		 pixel_size=self.config.PIXEL_SIZE,
		            		 divide_num=self.config.DIVIDE_NUM)

		self.config.save_config()
		os.remove(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
