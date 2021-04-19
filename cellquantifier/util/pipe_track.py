import pandas as pd; import numpy as np
import pims
import os; import os.path as osp
from datetime import date, datetime
import glob
import sys
import matplotlib.pyplot as plt

from skimage.io import imread, imsave

from ..smt import *
from ..phys import *
from ..plot import *
from ..video import *

class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

		#I/O
		self.INPUT_PATH = settings_dict['Input path']
		self.OUTPUT_PATH = settings_dict['Output path']

		#TRACKING SETTINGS
		self.PIXEL_SIZE = settings_dict['Pixel size']
		self.FRAME_RATE = settings_dict['Frame rate']
		self.SEARCH_RANGE = settings_dict['Trak search_range']
		self.MEMORY = settings_dict['Trak memory']
		self.DIVIDE_NUM = settings_dict['Trak divide_num']

		#TRACKING FILTERING SETTINGS
		if (settings_dict['Foci filt max_dist_err']=='') & \
			(settings_dict['Foci filt max_sig_to_sigraw']=='') & \
			(settings_dict['Foci filt max_delta_area']==''):
			self.DO_FILTER = False
		else:
			self.DO_FILTER = True

		self.FILTERS = {

		'MAX_DIST_ERROR': settings_dict['Foci filt max_dist_err'],
		'SIG_TO_SIGRAW' : settings_dict['Foci filt max_sig_to_sigraw'],
		'MAX_DELTA_AREA': settings_dict['Foci filt max_delta_area'],
		'TRAJ_LEN_THRES': settings_dict['Foci filt traj_length_thres'],

		}

	def save_config(self):
		settings_df = pd.DataFrame.from_dict(data=self.settings, orient='index')
		settings_df = settings_df.drop(['Input path', 'Output path'])
		settings_df.to_csv(self.settings['Output path'] + self.root_name + \
				'-config-track' + '.csv', header=False)

	# helper function for filt and track()
	def track_blobs_twice(self):
		frames = imread(self.settings['Input path'] + self.root_name + \
				'.tif')
		psf_df = pd.read_csv(self.settings['Input path'] + self.root_name + \
				'-fittData.csv')

		blobs_df, im = track_blobs(psf_df,
			    search_range=self.SEARCH_RANGE,
				memory=self.MEMORY,
				pixel_size=self.PIXEL_SIZE,
				frame_rate=self.FRAME_RATE,
				divide_num=self.DIVIDE_NUM,
				filters=None,
				do_filter=False)

		if self.DO_FILTER:
			blobs_df, im = track_blobs(blobs_df,
			    search_range=self.SEARCH_RANGE,
				memory=self.MEMORY,
				pixel_size=self.PIXEL_SIZE,
				frame_rate=self.FRAME_RATE,
				divide_num=self.DIVIDE_NUM,
				filters=self.FILTERS,
				do_filter=True)

		# Add 'traj_length' column and save physData before traj_length_thres filter
		blobs_df = add_traj_length(blobs_df)

		return blobs_df


	# helper function for filt and track()
	def print_filt_traj_num(self, blobs_df):
		traj_num_before = blobs_df['particle'].nunique()
		after_filter_df = blobs_df [blobs_df['traj_length'] >= self.FILTERS['TRAJ_LEN_THRES']]
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
			df = df[ df['traj_length']>=self.FILTERS['TRAJ_LEN_THRES'] ]

		return df

	def filt_track(self):

		print("######################################")
		print("Filter and Linking")
		print("######################################")

		check_search_range = isinstance(self.SEARCH_RANGE, list)
		if check_search_range:
			param_list = self.SEARCH_RANGE
			particle_num_list = []
			phys_dfs = []
			mean_D_list = []
			mean_alpha_list = []
			for search_range in param_list:
				self.SEARCH_RANGE = search_range
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
			sr_opt_fig.savefig(self.OUTPUT_PATH + \
							self.root_name + '-opt-search-range.pdf')

		check_memory = isinstance(self.MEMORY, list)
		if check_memory:
			param_list = self.MEMORY
			particle_num_list = []
			phys_dfs = []
			mean_D_list = []
			mean_alpha_list = []
			for memory in param_list:
				self.MEMORY = memory
				phys_df = self.track_blobs_twice()
				self.print_filt_traj_num(phys_df)
				phys_df = self.filt_phys_df(phys_df)
				phys_df = phys_df.drop_duplicates('particle')
				phys_df['memory'] = memory
				phys_dfs.append(phys_df)
				particle_num_list.append(len(phys_df))
				mean_D_list.append(phys_df['D'].mean())
				mean_alpha_list.append(phys_df['alpha'].mean())
			phys_df_all = pd.concat(phys_dfs)
			sr_opt_fig = plot_track_param_opt(
							track_param_name='memory',
							track_param_unit='pixel',
							track_param_list=param_list,
							particle_num_list=particle_num_list,
							df=phys_df_all,
							mean_D_list=mean_D_list,
							mean_alpha_list=mean_alpha_list,
							)
			sr_opt_fig.savefig(self.OUTPUT_PATH + \
							self.root_name + '-opt-memory.pdf')

		check_traj_len_thres = isinstance(self.FILTERS['TRAJ_LEN_THRES'], list)
		if check_traj_len_thres:
			param_list = self.FILTERS['TRAJ_LEN_THRES']
			particle_num_list = []
			phys_dfs = []
			mean_D_list = []
			mean_alpha_list = []

			if osp.exists(self.OUTPUT_PATH + self.root_name + '-physData.csv'):
				original_phys_df = pd.read_csv(self.OUTPUT_PATH + \
					self.root_name + '-physData.csv')
			else:
				original_phys_df = self.track_blobs_twice()

			for traj_len_thres in param_list:
				self.FILTERS['TRAJ_LEN_THRES'] = traj_len_thres
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
			sr_opt_fig.savefig(self.OUTPUT_PATH + \
							self.root_name + '-opt-traj-len-thres.pdf')

		else:
			blobs_df = self.track_blobs_twice()
			self.print_filt_traj_num(blobs_df)
			blobs_df.round(3).to_csv(self.OUTPUT_PATH + self.root_name + \
										'-physData.csv', index=False)

		self.save_config()

	def plot_traj(self):
		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = imread(self.settings['Input path'] + self.root_name + \
				'.tif')
		phys_df = pd.read_csv(self.OUTPUT_PATH + self.root_name + '-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.FILTERS['TRAJ_LEN_THRES']!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.FILTERS['TRAJ_LEN_THRES'] ]


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
					pixel_size=self.PIXEL_SIZE,

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
		fig.savefig(self.OUTPUT_PATH + self.root_name + '-results.pdf')
		plt.clf(); plt.close()
		# plt.show()

		self.save_config()

	def anim_traj(self):

		print("######################################")
		print("Animating trajectories")
		print("######################################")

		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = imread(self.settings['Input path'] + self.root_name + \
				'.tif')
		phys_df = pd.read_csv(self.OUTPUT_PATH + self.root_name + '-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.FILTERS['TRAJ_LEN_THRES']!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.FILTERS['TRAJ_LEN_THRES'] ]


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

		# """
		# ~~~~~~~~Generate and save animation video~~~~~~~~
		# """
		anim_tif = anim_traj(phys_df, frames,

					show_image=True,

					show_scalebar=True,
					pixel_size=self.PIXEL_SIZE,

					show_colorbar=True,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

					show_traj_num=False,

		            show_tail=True,
					tail_length=500,

					# show_boundary=False,
					# boundary_masks=boundary_masks,

					dpi=100,
					)
		imsave(self.OUTPUT_PATH + self.root_name + '-animVideo.tif', anim_tif,
			check_contrast=False)


def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('.tif')]
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
