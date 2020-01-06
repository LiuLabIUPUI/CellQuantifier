import pims; import pandas as pd; import numpy as np
import trackpy as tp
import os.path as osp; import os
from datetime import date
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
import warnings
import glob

from cellquantifier.deno import filter_batch
from cellquantifier.segm import get_mask_batch
from cellquantifier.segm import get_thres_mask_batch
from cellquantifier.regi import get_regi_params, apply_regi_params

from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
from cellquantifier.smt.fit_psf import fit_psf, fit_psf_batch
from cellquantifier.smt.track import track_blobs
from cellquantifier.smt.msd import plot_msd_batch, get_sorter_list
from cellquantifier.phys import *
from cellquantifier.util.config import Config
from cellquantifier.plot import plot_phys_1 as plot_merged
from cellquantifier.phys.physutil import relabel_particles, merge_physdfs

class Pipeline():

	def __init__(self, config):

		self.config = config

	def load(self, config):

		frames = imread(config.INPUT_PATH)
		frames = frames[list(config.TRANGE),:,:]
		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-raw.tif', frames)
		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif', frames)

	def register(self, config):

		print("######################################")
		print("Registering Image Stack")
		print("######################################")

		im = pims.open(config.FILENAME)

		regi_params_array_2d = get_regi_params(im,
		              ref_ind_num=config.REF_IND_NUM,
		              sig_mask=config.SIG_MASK,
		              thres_rel=config.THRES_REL,
		              poly_deg=config.POLY_DEG,
		              rotation_multplier=config.ROTATION_MULTIPLIER,
		              translation_multiplier=config.TRANSLATION_MULTIPLIER,
		              diagnostic=True)

		registered = apply_regi_params(im, regi_params_array_2d)

		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif', registered)
		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif', registered)


	def deno(self, config, method, arg):

		print("######################################")
		print('Applying ' + method.capitalize() + ' Filter')
		print("######################################")

		frames = imread(config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif')
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		filtered = filter_batch(frames, method=method, arg=arg)

		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif', filtered)
		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-deno.tif', filtered)

		return filtered

	def check(self, config):

		print("######################################")
		print("Check detection and fitting")
		print("######################################")

		if osp.exists(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-raw.tif')

		frames_deno = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif')

		blobs_df, det_plt_array = detect_blobs(frames[config.CHECK_FRAME],
									min_sig=config.MIN_SIGMA,
									max_sig=config.MAX_SIGMA,
									num_sig=config.NUM_SIGMA,
									blob_thres=config.THRESHOLD,
									peak_thres_rel=config.PEAK_THRESH_REL,
									r_to_sigraw=config.PATCH_SIZE,
									pixel_size = config.PIXEL_SIZE,
									pltshow=True,
									plot_r=config.PLOT_R,
									truth_df=None)

	def detect_fit(self, config):

		print("######################################")
		print("Detect, Fit")
		print("######################################")

		if osp.exists(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-raw.tif')

		frames_deno = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames,
									min_sig=config.MIN_SIGMA,
									max_sig=config.MAX_SIGMA,
									num_sig=config.NUM_SIGMA,
									blob_thres=config.THRESHOLD,
									peak_thres_rel=config.PEAK_THRESH_REL,
									r_to_sigraw=config.PATCH_SIZE,
									pixel_size = config.PIXEL_SIZE,
									pltshow=True,
									plot_r=config.PLOT_R,
									truth_df=None)

		psf_df, fit_plt_array = fit_psf_batch(frames_deno,
				            blobs_df,
				            pltshow=True,
				            diag_max_dist_err=config.FILTERS['MAX_DIST_ERROR'],
				            diag_max_sig_to_sigraw = config.FILTERS['SIG_TO_SIGRAW'],
				            truth_df=None,
				            segm_df=None,
							output_path=config.OUTPUT_PATH,
							root_name=config.ROOT_NAME,
							save_csv=True)

		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-detVideo.tif', det_plt_array)
		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-fittVideo.tif', fit_plt_array)

		return det_plt_array, fit_plt_array


	def filter_and_track(self, config):

		print("######################################")
		print("Filter and Linking")
		print("######################################")

		if osp.exists(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-raw.tif')

		psf_df = pd.read_csv(config.OUTPUT_PATH + config.ROOT_NAME + '-fittData.csv')

		blobs_df, im = track_blobs(psf_df,
								    search_range=config.SEARCH_RANGE,
									memory=config.MEMORY,
									pixel_size=config.PIXEL_SIZE,
									frame_rate=config.FRAME_RATE,
									divide_num=config.DIVIDE_NUM,
									filters=None,
									do_filter=False)

		traj_num_before = blobs_df['particle'].nunique()

		if config.DO_FILTER:
			blobs_df, im = track_blobs(blobs_df,
									    search_range=config.SEARCH_RANGE,
										memory=config.MEMORY,
										pixel_size=config.PIXEL_SIZE,
										frame_rate=config.FRAME_RATE,
										divide_num=config.DIVIDE_NUM,
										filters=config.FILTERS,
										do_filter=True)

		blobs_df.to_csv(config.OUTPUT_PATH + config.ROOT_NAME + '-physData.csv')

		# plot_msd_batch(blobs_df,
		# 			 image=frames[0],
		# 			 output_path=config.OUTPUT_PATH,
		# 			 root_name=config.ROOT_NAME,
		# 			 pixel_size=config.PIXEL_SIZE,
		# 			 frame_rate=config.FRAME_RATE,
		# 			 divide_num=config.DIVIDE_NUM,
		# 			 pltshow=True,
		# 			 check_traj_msd=True)
		#
		# print("######################################")
		# print("Trajectory number before filters: \t%d" % traj_num_before)
		# print("Trajectory number after filters: \t%d" % blobs_df['particle'].nunique())
		# print("######################################")


	def segmentation(self, config, ax=None):

		print("######################################")
		print("Generate Masks")
		print("######################################")

		frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-active.tif')

		masks_array_3d, dist_array_3d, px_array_3d = get_thres_mask_batch(frames,
							config.MASK_SIG, config.MASK_THRES,
							config.MASK_MIN_SIZE)

		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-mask.tif',
				masks_array_3d)

		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-dist_mask.tif',
				dist_array_3d)

		imsave(config.OUTPUT_PATH + config.ROOT_NAME + '-px_mask.tif',
				px_array_3d)

		return masks_array_3d, dist_array_3d, px_array_3d


	def phys(self, config):

		print("######################################")
		print("Add Physics Parameters")
		print("######################################")

		blobs_df = pd.read_csv(config.OUTPUT_PATH + config.ROOT_NAME + '-physData.csv')

		# add 'dist_to_boundary' column
		print("######################################")
		print("Add 'dist_to_boundary'")
		print("######################################")
		dist2boundary_tif = imread(config.INPUT_PATH + \
							config.DIST2BOUNDARY_MASK_NAME) \
							[list(config.TRANGE),:,:]

		dist2boundary_thres_masks = get_thres_mask_batch(dist2boundary_tif,
							config.MASK_SIG_BOUNDARY, config.MASK_THRES_BOUNDARY)
		phys_df = add_dist_to_boundary_batch(blobs_df, dist2boundary_thres_masks)

		# add 'dist_to_boundary' column
		print("######################################")
		print("Add 'dist_to_53bp1'")
		print("######################################")
		dist253bp1_tif = imread(config.INPUT_PATH + \
							config.DIST253BP1_MASK_NAME) \
							[list(config.TRANGE),:,:]

		dist253bp1_thres_masks = get_thres_mask_batch(dist253bp1_tif,
							config.MASK_SIG_53BP1, config.MASK_THRES_53BP1)
		phys_df = add_dist_to_53bp1_batch(blobs_df, dist253bp1_thres_masks)

		# Save '-physData.csv'
		phys_df.to_csv(config.OUTPUT_PATH + config.ROOT_NAME + \
						'-physData.csv', index=False)


	def sort_and_plot(self, config):

		print("######################################")
		print("Sort and PlotMSD")
		print("######################################")

		if osp.exists(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif'):
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-regi.tif')
		else:
			frames = pims.open(config.OUTPUT_PATH + config.ROOT_NAME + '-raw.tif')

		phys_df = pd.read_csv(config.OUTPUT_PATH + config.ROOT_NAME + '-physData.csv')

		if config.DO_SORT:
			phys_df = sort_phys(phys_df, selfconfig.SORTERS)
			phys_df.to_csv(selfconfig.OUTPUT_PATH + selfconfig.ROOT_NAME + \
						'-physData.csv', index=False)
		else:
			sorter_list = get_sorter_list(phys_df)
			phys_df = phys_df.drop(columns=sorter_list[1:-1])

		plot_msd_batch(phys_df,
					 image=frames[0],
					 output_path=selfconfig.OUTPUT_PATH,
					 root_name=selfconfig.ROOT_NAME,
					 pixel_size=selfconfig.PIXEL_SIZE,
					 frame_rate=selfconfig.FRAME_RATE,
					 divide_num=selfconfig.DIVIDE_NUM,
					 pltshow=True)

		selfconfig.save_config()
		if osp.exists(selfconfig.OUTPUT_PATH + selfconfig.ROOT_NAME + '-active.tif'):
			os.remove(selfconfig.OUTPUT_PATH + selfconfig.ROOT_NAME + '-active.tif')
		if osp.exists(selfconfig.OUTPUT_PATH + selfconfig.ROOT_NAME + '-boundaryMask.tif'):
			os.remove(selfconfig.OUTPUT_PATH + selfconfig.ROOT_NAME + '-boundaryMask.tif')
		if osp.exists(selfconfig.OUTPUT_PATH + selfconfig.ROOT_NAME + '-53bp1Mask.tif'):
			os.remove(selfconfig.OUTPUT_PATH + selfconfig.ROOT_NAME + '-53bp1Mask.tif')

def merge_and_plot(self, config):

	start_ind = selfconfig.ROOT_NAME.find('_')
	end_ind = selfconfig.ROOT_NAME.find('_', start_ind+1)
	today = str(date.today().strftime("%y%m%d"))
	merged_name = today + selfconfig.ROOT_NAME[start_ind:end_ind]

	print("######################################")
	print("Merge and PlotMSD")
	print("######################################")

	merged_files = np.array(sorted(glob.glob(config.OUTPUT_PATH + '/*physDataMerged.csv')))
	print(merged_files)
	if len(merged_files) > 1:
		print("######################################")
		print("Found multiple physDataMerged file!!!")
		print("######################################")
		return

	if len(merged_files) == 1:
		phys_df = pd.read_csv(merged_files[0])

	else:
		phys_files = np.array(sorted(glob.glob(config.OUTPUT_PATH + '/*physData.csv')))
		print(phys_files)

		if len(phys_files) > 1:
			phys_df = merge_physdfs(phys_files)
			phys_df = relabel_particles(phys_df)
		else:
			phys_df = pd.read_csv(phys_files[0])

		phys_df.to_csv(selfconfig.OUTPUT_PATH + merged_name + \
						'-physDataMerged.csv', index=False)

	# phys_df = phys_df.loc[phys_df['exp_label'] == 'BLM']
	fig = plot_merged(phys_df, 'exp_label',
					pixel_size=selfconfig.PIXEL_SIZE,
					frame_rate=selfconfig.FRAME_RATE,
					divide_num=selfconfig.DIVIDE_NUM,
					RGBA_alpha=0.5,
					do_gmm=False)

	fig.savefig(self.config.OUTPUT_PATH + merged_name + '-mergedResults.pdf')

def pipeline_control(settings_dict, control_dict):

	warnings.filterwarnings("ignore")
	config = Config(settings_dict)
	pipe = Pipeline(config)

	if control_dict['load']:

		pipe.load(config)

	if control_dict['regi']:
		pipe.register()

	# If not registered, remove meaningless regi parameters
	if int(settings_dict['Regi rotation_multiplier'] * 1000) < 1 & \
					int(settings_dict['Regi rotation_multiplier'] * 1000) < 0.001:
		settings_dict['Regi ref_ind_num'] = 'NA'
		settings_dict['Regi sig_mask'] = 'NA'
		settings_dict['Regi thres_rel'] = 'NA'
		settings_dict['Regi poly_deg'] = 'NA'
		settings_dict['Regi rotation_multplier'] = 'NA'
		settings_dict['Regi translation_multiplier'] = 'NA'
	if control_dict['mask']:
		pipe.mask(config)
	if control_dict['deno']:
		pipe.deno(config, method='boxcar', arg=settings_dict['Deno boxcar_radius'])
		pipe.deno(config, method='gaussian', arg=settings_dict['Deno gaus_blur_sig'])
	if control_dict['check']:
		pipe.check(config)
	if control_dict['detect_fit']:
		pipe.detect_fit(config)
	if control_dict['filt_trak']:
		pipe.filter_and_track(config)
	if control_dict['phys']:
		pipe.phys(config)
	if control_dict['sort_plot']:
		pipe.sort_and_plot(config)
	if control_dict['merge_plot']:
		pipe.merge_and_plot(config)
