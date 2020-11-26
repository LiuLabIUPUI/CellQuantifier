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
from scipy import ndimage as ndi
import skimage
from skimage.segmentation import watershed, mark_boundaries
from skimage.morphology import binary_dilation, binary_erosion, disk, dilation
from skimage import color
import matplotlib.patches as patches

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

		#FILE SETTINGS
		self.DAPI_MASK_FILE = ''
		self.DAPI_MARKER_FILE = ''
		self.DAPI_FILE = ''
		self.INS_FILE = ''
		self.MIR146_FILE = ''
		self.MIR155_FILE = ''

		#DENOISE SETTINGS
		self.BOXCAR_RADIUS = 10
		self.GAUS_BLUR_SIG = 0.5

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

		#DICT
		self.DICT = config.copy()


	def save_config(self, rna_label='miR155'):

		path = self.OUTPUT_PATH + self.ROOT_NAME + '-analMeta-' + rna_label + '.csv'
		config_df = pd.DataFrame.from_dict(data=self.DICT, orient='index')
		config_df = config_df.drop(['IO input_path', 'IO output_path',
									'Processed By:'])
		config_df.to_csv(path, header=False)

	def clean_dir(self):

		flist = [f for f in os.listdir(self.OUTPUT_PATH)]
		for f in flist:
		    os.remove(os.path.join(self.OUTPUT_PATH, f))

def expand_labels(label_image, distance=1):
	distances, nearest_label_coords = ndi.distance_transform_edt(
        label_image == 0, return_indices=True)
	labels_out = np.zeros_like(label_image)
	dilate_mask = distances <= distance
	# build the coordinates to find nearest labels,
	# in contrast to [1] this implementation supports label arrays
	# of any dimension
	masked_nearest_label_coords = [
	    dimension_indices[dilate_mask]
	    for dimension_indices in nearest_label_coords
	]
	nearest_labels = label_image[tuple(masked_nearest_label_coords)]
	labels_out[dilate_mask] = nearest_labels
	return labels_out

def nonempty_exists_then_copy(input_path, output_path, filename):
	not_empty = len(filename)!=0
	exists_in_input = osp.exists(input_path + filename)

	if not_empty and exists_in_input:
		frames = imread(input_path + filename)
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
		pass

	def segm_cell(self):
		print("######################################")
		print("segment cells")
		print("######################################")

		print('\tDapi mask file: [%s]' % self.config.DAPI_MASK_FILE)
		print('\tDapi marker file: [%s]' % self.config.DAPI_MARKER_FILE)
		print('\tDapi file: [%s]' % self.config.DAPI_FILE)

		dapi = imread(self.config.OUTPUT_PATH + self.config.DAPI_FILE)
		dapi = dapi / dapi.max()
		dapi = img_as_ubyte(dapi)
		dapi_rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3), dtype=dapi.dtype)
		dapi_rgb[:,:,2] = dapi
		dapi = dapi_rgb

		dapi_mask = imread(self.config.OUTPUT_PATH + self.config.DAPI_MASK_FILE)
		dapi_mask = dapi_mask==255

		dapi_marker = pd.read_csv(self.config.OUTPUT_PATH + self.config.DAPI_MARKER_FILE,
		 				sep='\t', header=None)
		dapi_marker.columns = ['y', 'x']
		dapi_marker_mask = np.zeros_like(dapi_mask, dtype=bool)
		for idx in dapi_marker.index:
			r = int(dapi_marker.loc[idx, 'x'])
			c = int(dapi_marker.loc[idx, 'y'])
			dapi_marker_mask[r, c] = True
		markers = ndi.label(dapi_marker_mask)[0]

		distance = ndi.distance_transform_edt(dapi_mask)
		labels = watershed(-distance, markers, mask=dapi_mask)

		labels2 = labels.copy()
		while True:
			labels2 = expand_labels(labels2,distance=30)
			if np.count_nonzero(labels2==0) == 0:
				break

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-labels.tif', labels2)

		fig, ax = plt.subplots(1, 2)
		trace1 = mark_boundaries(dapi, labels)
		ax[0].imshow(trace1)
		plt.box(False)
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		ax[0].set_title("Watershed")
		trace2 = mark_boundaries(dapi, labels2)
		ax[1].imshow(trace2)
		plt.box(False)
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		ax[1].set_title("Expanded Watershed")

		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-segm.pdf',
                    dpi=300)
		plt.clf(); plt.close()

	def classify_cell(self):
		print("######################################")
		print("classify cells")
		print("######################################")
		print('\tINS file: [%s]' % self.config.INS_FILE)
		ins = imread(self.config.OUTPUT_PATH + self.config.INS_FILE)
		mask = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-labels.tif')

		df = pd.DataFrame([], columns=['raw_data', 'cell'])
		cells = np.unique(mask)
		df['cell'] = cells
		df['raw_data'] = self.config.ROOT_NAME
		for cell in cells:
			cell_array = ins[mask==cell]
			df.loc[ df['cell']==cell, 'ins_mean'] = cell_array.mean()
			df.loc[ df['cell']==cell, 'ins_sum'] = cell_array.sum()
		df['cell_type'] = 'low_ins'
		high_ins_bool = (df['ins_mean']>=df['ins_mean'].mean()) & \
						(df['ins_sum']>=df['ins_sum'].mean())
		df.loc[high_ins_bool, 'cell_type'] = 'high_ins'

		fig, ax = plt.subplots(1, 2, figsize=(8,4))
		ax[0].plot(df['ins_mean'], df['ins_sum'], 'o')
		rect = patches.Rectangle((df['ins_mean'].mean(),df['ins_sum'].mean()),
				df['ins_mean'].max()-df['ins_mean'].mean(),
				df['ins_sum'].max()-df['ins_sum'].mean(),
				linewidth=1,edgecolor='r',facecolor='none')
		ax[0].add_patch(rect)
		ax[0].set_xlabel('INS mean intensity', fontsize='large')
		ax[0].set_ylabel('INS total intensity', fontsize='large')

		ins_rgb = ins / ins.max()
		ins_rgb = img_as_ubyte(ins_rgb)
		ins_rgb = color.gray2rgb(ins_rgb)
		high_ins_cells = df[ df['cell_type']=='high_ins' ]['cell'].unique()
		for cell in high_ins_cells:
			ins_rgb[mask==cell] = ins_rgb[mask==cell]*[1,0,0]

		trace = mark_boundaries(ins_rgb, mask)
		ax[1].imshow(trace)
		plt.box(False)
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		ax[1].set_title("High INS cell")

		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-celltype.pdf',
                    dpi=300)
		plt.clf(); plt.close()
		df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def detect_miR(self, rna_label='miR155'):
		ins = imread(self.config.OUTPUT_PATH + self.config.INS_FILE)
		mask = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-labels.tif')
		df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		if rna_label=='miR146':
			print('\tmiR146 file: [%s]' % self.config.MIR146_FILE)
			rna = imread(self.config.OUTPUT_PATH + self.config.MIR146_FILE)
			frames = imread(self.config.OUTPUT_PATH + self.config.MIR146_FILE)
		elif rna_label=='miR155':
			print('\tmiR155 file: [%s]' % self.config.MIR155_FILE)
			rna = imread(self.config.OUTPUT_PATH + self.config.MIR155_FILE)
			frames = imread(self.config.OUTPUT_PATH + self.config.MIR155_FILE)
		else:
			sys.exit()

		# If only 1 frame available, duplicate it to enough frames_num.
		tot_frame_num = 5
		if frames.ndim==2:
			dup_frames = np.zeros((tot_frame_num, frames.shape[0], frames.shape[1]),
									dtype=frames.dtype)
			for i in range(tot_frame_num):
				dup_frames[i] = frames
			frames = dup_frames

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-tempFile.tif', frames)
		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-tempFile.tif')
		blobs_df, det_plt_array = detect_blobs(frames[0],
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
									pltshow=True,
									blob_markersize=5,
									plot_r=True,
									truth_df=None)

		os.remove(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-tempFile.tif')

		fig, ax = plt.subplots(1, 2, figsize=(8,4))
		ax[0].imshow(rna, cmap='gray')
		anno_blob(ax[0], blobs_df,
		            marker='^',
		            markersize=10,
		            plot_r=True,
		            color=(0,0,1,1))
		ax[0].spines['right'].set_visible(False)
		ax[0].spines['top'].set_visible(False)
		ax[0].spines['left'].set_visible(False)
		ax[0].spines['bottom'].set_visible(False)
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		ax[0].set_title(rna_label)


		rna_rgb = rna / rna.max()
		rna_rgb = img_as_ubyte(rna_rgb)
		rna_rgb = color.gray2rgb(rna_rgb)
		high_ins_cells = df[ df['cell_type']=='high_ins' ]['cell'].unique()
		for cell in high_ins_cells:
			rna_rgb[mask==cell] = rna_rgb[mask==cell]*[1,0,0]

		trace = mark_boundaries(rna_rgb, mask)
		ax[1].imshow(trace)
		plt.box(False)
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		ax[1].set_title("High_INS + " + rna_label)
		anno_blob(ax[1], blobs_df,
		            marker='^',
		            markersize=10,
		            plot_r=True,
		            color=(0,0,1,1))

		plt.show()

		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
				'-' + rna_label + '-detection.pdf',
                    dpi=300)

		# """
	    # ~~~~rna_mask~~~~
	    # """
		rna_mask = np.zeros_like(rna, dtype=bool)
		for idx in blobs_df.index:
			r = int(blobs_df.loc[idx, 'x'])
			c = int(blobs_df.loc[idx, 'y'])
			rna_mask[r, c] = True

		cells = df['cell'].unique()
		col_name = rna_label + '_num'
		if col_name in df:
			df = df.drop(col_name, axis=1)

		for cell in cells:
			curr_rna = rna_mask[mask==cell]
			df.loc[ df['cell']==cell, col_name] = np.count_nonzero(curr_rna)

		blobs_df.drop('frame', axis=1)
		blobs_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-' + rna_label + '-detData.csv', index=False)
		df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)
		self.config.save_config(rna_label=rna_label)

	def detect_miR146(self):
		print("######################################")
		print("miR146 detection")
		print("######################################")
		self.detect_miR(rna_label='miR146')

	def detect_miR155(self):
		print("######################################")
		print("miR155 detection")
		print("######################################")
		self.detect_miR(rna_label='miR155')


	def merge_plot(self):
		pass

		# today = str(date.today().strftime("%y%m%d"))
		#
		# print("######################################")
		# print("Merge and PlotMSD")
		# print("######################################")
		#
		# merged_files = np.array(sorted(glob(self.config.OUTPUT_PATH + '/*physDataMerged.csv')))
		# print(merged_files)
		#
		# if len(merged_files) > 1:
		# 	print("######################################")
		# 	print("Found multiple physDataMerged file!!!")
		# 	print("######################################")
		# 	return
		#
		# if len(merged_files) == 1:
		# 	phys_df = pd.read_csv(merged_files[0])
		#
		# else:
		# 	phys_files = np.array(sorted(glob(self.config.OUTPUT_PATH + '/*physData.csv')))
		# 	print("######################################")
		# 	print("Total number of physData to be merged: %d" % len(phys_files))
		# 	print("######################################")
		# 	print(phys_files)
		#
		# 	if len(phys_files) > 1:
		# 		ind = 1
		# 		tot = len(phys_files)
		# 		for file in phys_files:
		# 			print("Updating fittData (%d/%d)" % (ind, tot))
		# 			ind = ind + 1
		#
		# 			curr_physdf = pd.read_csv(file, index_col=False)
		# 			if 'traj_length' not in curr_physdf:
		# 				curr_physdf = add_traj_length(curr_physdf)
		# 				curr_physdf.round(3).to_csv(file, index=False)
		#
		# 		phys_df = merge_physdfs(phys_files, mode='general')
		#
		# 	else:
		# 		phys_df = pd.read_csv(phys_files[0])
		#
		#
		# 	print("######################################")
		# 	print("Rename particles...")
		# 	print("######################################")
		# 	phys_df['particle'] = phys_df['raw_data'] + phys_df['particle'].apply(str)
		# 	phys_df.round(3).to_csv(self.config.OUTPUT_PATH + today + \
		# 					'-physDataMerged.csv', index=False)
		#
		# # Apply traj_length_thres filter
		# if 'traj_length' in phys_df:
		# 	phys_df = phys_df[ phys_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES'] ]
		#
		# fig_quick_merge(phys_df)
		#
		#
		# sys.exit()


def get_root_name_list(settings_dict):
	# Make a copy of settings_dict
	# Use '*%#@)9_@*#@_@' to substitute if the labels are empty
	settings = settings_dict.copy()
	if settings['Dapi mask file label'] == '':
		settings['Dapi mask file label'] = '*%#@)9_@*#@_@'
	if settings['Dapi marker file label'] == '':
		settings['Dapi marker file label'] = '*%#@)9_@*#@_@'
	if settings['Dapi channel file label'] == '':
		settings['Dapi channel file label'] = '*%#@)9_@*#@_@'
	if settings['INS channel file label'] == '':
		settings['INS channel file label'] = '*%#@)9_@*#@_@'
	if settings['miR146 channel file label'] == '':
		settings['miR146 channel file label'] = '*%#@)9_@*#@_@'
	if settings['miR155 channel file label'] == '':
		settings['miR155 channel file label'] = '*%#@)9_@*#@_@'

	root_name_list = []

	path_list = glob(settings['IO input_path'] + '/*-physData.csv')
	if len(path_list) != 0:
		for path in path_list:
			temp = path.split('/')[-1]
			temp = temp[:-len('-physData.csv')]
			root_name_list.append(temp)

	else:
		path_list = glob(settings['IO input_path'] + '/*-dapi-mask.tif')
		if len(path_list) != 0:
			for path in path_list:
				temp = path.split('/')[-1]
				temp = temp[:-4 - len('-dapi-mask')]
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

		# 2.2. Update file labels
		if settings_dict['Dapi mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Dapi mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DAPI_MASK_FILE = file_list[0].split('/')[-1]
			else:
				config.DICT['Dapi mask file label'] = ''

		if settings_dict['Dapi marker file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Dapi marker file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DAPI_MARKER_FILE = file_list[0].split('/')[-1]
			else:
				config.DICT['Dapi marker file label'] = ''

		if settings_dict['Dapi channel file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' +
					'*' + settings_dict['Dapi channel file label'] + key + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DAPI_FILE = file_list[0].split('/')[-1]
			else:
				config.DICT['Dapi channel file label'] = ''

		if settings_dict['INS channel file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' +
					'*' + settings_dict['INS channel file label'] + key + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.INS_FILE = file_list[0].split('/')[-1]
			else:
				config.DICT['INS channel file label'] = ''

		if settings_dict['miR146 channel file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' +
					'*' + settings_dict['miR146 channel file label'] + key + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.MIR146_FILE = file_list[0].split('/')[-1]
			else:
				config.DICT['miR146 channel file label'] = ''

		if settings_dict['miR155 channel file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' +
					'*' + settings_dict['miR155 channel file label'] + key + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.MIR155_FILE = file_list[0].split('/')[-1]
			else:
				config.DICT['miR155 channel file label'] = ''

		# """
		# ~~~~~~~~~~~~~~~~~3. Setup pipe and run~~~~~~~~~~~~~~~~~
		# """
		pipe = Pipeline3(config)
		for func in control_list:
			getattr(pipe, func)()
