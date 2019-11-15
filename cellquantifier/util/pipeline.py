import pims
import trackpy as tp
import os.path as osp; import os
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
import pandas as pd
import warnings

from ..deno import filter_batch
from ..segm import get_mask_batch
from ..segm import get_thres_mask_batch
from ..regi import get_regi_params, apply_regi_params

from ..smt.detect import detect_blobs, detect_blobs_batch
from ..smt.fit_psf import fit_psf, fit_psf_batch
from ..smt.track import track_blobs
from ..smt.msd import plot_msd_batch
from ..phys import *
from ..util.config import Config


class Pipeline():

	def __init__(self, config, is_new=True):

		self.config = config

		if is_new:
			self.config.clean_dir()

			frames = imread(config.INPUT_PATH)
			frames = frames[list(config.TRANGE),:,:]

			a = config.OUTPUT_PATH + config.ROOT_NAME + '-raw.tif'
			b = config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif'

			imsave(a, frames)
			imsave(b, frames)


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
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
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


	def detect_fit(self, detect_video=False, fit_psf_video=False):

		print("######################################")
		print("Detect, Fit")
		print("######################################")

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')

		if detect_video:
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
		else:
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
					output_path=self.config.OUTPUT_PATH,
					root_name=self.config.ROOT_NAME,
					save_csv=True)

	def filter_and_track(self):

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
									pixel_size=self.config.PIXEL_SIZE,
									frame_rate=self.config.FRAME_RATE,
									divide_num=self.config.DIVIDE_NUM,
									filters=None,
									do_filter=False,
									output_path=self.config.OUTPUT_PATH,
									root_name=self.config.ROOT_NAME,
									save_csv=False)

		if self.config.DO_FILTER:
			blobs_df, im = track_blobs(blobs_df,
									    search_range=self.config.SEARCH_RANGE,
										memory=self.config.MEMORY,
										pixel_size=self.config.PIXEL_SIZE,
										frame_rate=self.config.FRAME_RATE,
										divide_num=self.config.DIVIDE_NUM,
										filters=self.config.FILTERS,
										do_filter=True,
										output_path=self.config.OUTPUT_PATH,
										root_name=self.config.ROOT_NAME,
										save_csv=False)
		else:
			blobs_df.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-filtTrakData.csv')


	def phys(self):

		print("######################################")
		print("Add Physics Parameters")
		print("######################################")

		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-filtTrakData.csv')

		# add 'dist_to_boundary' column
		print("######################################")
		print("Add 'dist_to_boundary'")
		print("######################################")
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			dist2boundary_tif = imread(self.config.OUTPUT_PATH + \
							self.config.DIST2BOUNDARY_MASK_NAME + \
							'-regi.tif')[list(self.config.TRANGE),:,:]
		else:
			dist2boundary_tif = imread(self.config.OUTPUT_PATH + \
							self.config.DIST2BOUNDARY_MASK_NAME + \
							'-raw.tif')[list(self.config.TRANGE),:,:]

		dist2boundary_thres_masks = get_thres_mask_batch(dist2boundary_tif,
							self.config.MASK_SIG_BOUNDARY, self.config.MASK_THRES_BOUNDARY)
		phys_df = add_dist_to_boundary_batch(blobs_df, dist2boundary_thres_masks)

		# add 'dist_to_boundary' column
		print("######################################")
		print("Add 'dist_to_53bp1'")
		print("######################################")
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif'):
			dist253bp1_tif = imread(self.config.OUTPUT_PATH + \
							self.config.DIST253BP1_MASK_NAME + \
							'-regi.tif')[list(self.config.TRANGE),:,:]
		else:
			dist253bp1_tif = imread(self.config.OUTPUT_PATH + \
							self.config.DIST253BP1_MASK_NAME + \
							'-raw.tif')[list(self.config.TRANGE),:,:]

		dist253bp1_thres_masks = get_thres_mask_batch(dist253bp1_tif,
							self.config.MASK_SIG_53BP1, self.config.MASK_THRES_53BP1)
		phys_df = add_dist_to_53bp1_batch(blobs_df, dist253bp1_thres_masks)

		# Save '-physData.csv'
		phys_df.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')


	def sort_and_plot(self):

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
		phys_df.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-sortPhysData.csv')

		plot_msd_batch(phys_df,
					 image=frames[0],
					 output_path=self.config.OUTPUT_PATH,
					 root_name=self.config.ROOT_NAME,
					 pixel_size=self.config.PIXEL_SIZE,
					 frame_rate=self.config.FRAME_RATE,
					 divide_num=self.config.DIVIDE_NUM,
					 pltshow=False)

		self.config.save_config()
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif'):
			os.remove(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')


def pipeline_control(settings_dict, control_dict):
	warnings.filterwarnings("ignore")
	config = Config(settings_dict)
	pipe = Pipeline(config, is_new=False)

	if control_dict['load']:
		config = Config(settings_dict)
		pipe = Pipeline(config)
	if control_dict['regi']:
		pipe.register()

	# If not registered, remove meaningless regi parameters
	if settings_dict['Regi rotation_multplier'] == 0 & \
					settings_dict['Regi rotation_multplier'] == 0:
		settings_dict['Regi ref_ind_num'] = 'NA'
		settings_dict['Regi sig_mask'] = 'NA'
		settings_dict['Regi thres_rel'] = 'NA'
		settings_dict['Regi poly_deg'] = 'NA'
		settings_dict['Regi rotation_multplier'] = 'NA'
		settings_dict['Regi translation_multiplier'] = 'NA'

	if control_dict['deno']:
		pipe.deno(method='boxcar', arg=settings_dict['Deno boxcar_radius'])
		pipe.deno(method='gaussian', arg=settings_dict['Deno gaus_blur_sig'])
	if control_dict['check']:
		pipe.check_start_frame()
	if control_dict['detect_fit']:
		pipe.detect_fit(detect_video=control_dict['video'])
	if control_dict['filt_trak']:
		pipe.filter_and_track()
	if control_dict['phys']:
		pipe.phys()
	if control_dict['sort_plot']:
		pipe.sort_and_plot()
