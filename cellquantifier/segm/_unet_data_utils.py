import os
import numpy as np
import requests
import random
import warnings
from os.path import join
from glob import glob
from skimage.io import imread, imsave
from skimage.morphology import *
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte
from zipfile import ZipFile


def get_bbbc_training_data(parent_dir, get_metadata=True, preprocess=True):

	"""
	Utility function for retrieving the BBBC039 U2OS Nuclei
	dataset for network training. The function builds the following
	directory structure under /parent_dir:

	/parent_dir
		/bbbc039
			/raw_images
			/raw_masks
			/metadata
			/proc_images
			/proc_masks

	Pseudo code
	----------
    1. Request dataset from data.broadinstitute.org
    2.

	Parameters
	----------
    parent_dir: parent_dir,
        The directory where the all folders and files should be stored

    preprocess: bool,
        Whether or not to normalize images and preprocess masks
		See normalize_iamges() and preprocess_masks() for more info.

	get_metadata: bool,
		Whether or not to request image  metatdata from data.broadinstitute.org

	"""

	# """
	# ~~~~~~~~~~Initialize file tree~~~~~~~~~~~~~~
	# """

	bbbc_dir = parent_dir + '/bbbc039'
	os.makedirs(bbbc_dir, exist_ok=True)

	# """
	# ~~~~~~~~~~Retrieve the dataset~~~~~~~~~~~~~~
	# """

	ext_im_url = 'https://data.broadinstitute.org/bbbc/BBBC039/images.zip'
	ext_mask_url = 'https://data.broadinstitute.org/bbbc/BBBC039/masks.zip'
	ext_meta_url = 'https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip'

	request_and_extract(ext_im_url, bbbc_dir, name='images')
	request_and_extract(ext_mask_url, bbbc_dir, name='masks')

	if get_metadata:
		request_and_extract(ext_meta_url, bbbc_dir, name='metadata')

	# """
	# ~~~~~~~~~~Preprocessing~~~~~~~~~~~~~~
	# """

	if preprocess:

		os.makedirs(bbbc_dir + '/proc_images', exist_ok=True)
		os.makedirs(bbbc_dir + '/proc_masks', exist_ok=True)

		normalize_bbbc_images(bbbc_dir + '/images', bbbc_dir + '/proc_images')
		preprocess_bbbc_masks(input_dir=bbbc_dir + '/masks',
							  proc_dir=bbbc_dir + '/proc_masks',
							  valid_dir=bbbc_dir + '/val_masks')

def request_and_extract(url, save_dir, name='images'):

	"""

	Utility function for obtaining training data that is stored
	at an external resource i.g. data.broadinstitute.org

	Pseudo code
	----------
	1. Download the .zip file to save_dir
	2. Extract the zip file in save_dir

	Parameters
	----------
	url : str
		url to the data
	save_dir : str
		local path where the data should be stored

	"""

	zip = requests.get(url)
	path_to_zip = save_dir + '/%s.zip' % (name)
	with open(path_to_zip, 'wb') as f:
		f.write(zip.content)

	with ZipFile(path_to_zip, 'r') as zipObj:
	   zipObj.extractall(save_dir + '/%s' % (name))

def normalize_images(input_dir, output_dir, clean=False):

	"""
	Normalizes images stored at input_dir and outputs to output_dir

	Pseudo code
	----------
	1. Clean output_dir
	2. Find images recursively under input_dir
	3. Normalize images to range [0, 1]

	Parameters
	----------

	"""

	warnings.filterwarnings("ignore")
	os.makedirs(output_dir, exist_ok=True)

	# """
	# ~~~~~~~~~~~Clean output_dir~~~~~~~~~~~~~~
	# """

	if clean:
		for f in os.listdir(output_dir):
			os.remove(join(output_dir, f))

	# """
	# ~~~~~~~~~~~Find images recursively~~~~~~~~~~~~~~
	# """

	files = glob(input_dir + '/**/*.tif', recursive=True)
	print('Found ' + str(len(files)) + ' images at ' + input_dir)

	# """
	# ~~~~~~~~~~~Normalize images~~~~~~~~~~~~~~
	# """

	for file in files:
		filename = file.split('/')[-1]
		print('Normalizing image: ' + filename)
		im = img_as_ubyte(imread(file))
		im = im/im.max()
		imsave(output_dir + '/' + filename, im)

def make_bbbc_masks_binary(input_dir, output_dir):

	"""
	Utility function for converting bbbc masks to binary

	Pseudo code
	----------
	1. Clean output_dir
	2. Find masks recursively under input_dir
	3. Convert masks to binary

	Parameters
	----------

	"""

	os.makedirs(output_dir, exist_ok=True)

	# """
	# ~~~~~~~~~~~Clean output_dir~~~~~~~~~~~~~~
	# """

	for f in os.listdir(output_dir):
		os.remove(join(output_dir, f))

	# """
	# ~~~~~~~~~~~Find masks recursively~~~~~~~~~~~~~~
	# """

	files = glob(input_dir + '/**/*.png', recursive=True)
	print('Found ' + str(len(files)) + ' masks at ' + input_dir)

	# """
	# ~~~~~~~~~~~Convert to binary~~~~~~~~~~~~~~
	# """

	for file in files:
		filename = file.split('/')[-1]
		mask = imread(file)
		mask = img_as_ubyte(mask[:,:,0] > 0)
		imsave(join(output_dir, filename), mask)


def preprocess_bbbc_masks(input_dir, proc_dir, valid_dir,
						  min_size=100, boundary_size=2):

	"""
	Removes objects smaller than min_size from mask and saves
	as a 3-channel image: interior, boundary, and background

	Pseudo code
	----------
	1. Clean output_dir
	2. Find masks recursively under input_dir
	3. Expand masks to 3-channels: interior, boundary, and background

	Parameters
	----------
	input_dir: str,
		directory containing the raw masks
	proc_dir: str,
		directory to store the masks in binary format
	valid_dir: str,
		directory to store the 3-channel masks (validation masks)
	min_size: int,
		smallest object to keep
	boundary_size: int,
		width of the object boundary in pixels

	"""

	warnings.filterwarnings("ignore")

	os.makedirs(proc_dir, exist_ok=True)
	os.makedirs(valid_dir, exist_ok=True)

	# """
	# ~~~~~~~~~~~Clean proc_dir~~~~~~~~~~~~~~
	# """

	for f in os.listdir(proc_dir):
		os.remove(join(proc_dir, f))


	# """
	# ~~~~~~~~~~~Clean valid_dir~~~~~~~~~~~~~~
	# """

	for f in os.listdir(valid_dir):
		os.remove(join(valid_dir, f))


	# """
	# ~~~~~~~~~~~Find masks recursively~~~~~~~~~~~~~~
	# """

	files = glob(input_dir + '/**/*.png', recursive=True)
	print('Found ' + str(len(files)) + ' masks at ' + input_dir)

	# """
	# ~~~~~~~~~~~Expand masks to 3-channels~~~~~~~~~~~~~~
	# """

	for file in files:

		filename = file.split('/')[-1]
		print('Preprocesing mask: ' + filename)

		raw_mask = imread(file)
		raw_mask = remove_small_objects(raw_mask[:,:,0], min_size=min_size)
		binary_mask = img_as_ubyte(raw_mask > 0)
		boundaries = find_boundaries(raw_mask)

		proc_mask = np.zeros((raw_mask.shape + (3,)))
		proc_mask[(raw_mask == 0) & (boundaries == 0), 0] = 1
		proc_mask[(raw_mask != 0) & (boundaries == 0), 1] = 1
		proc_mask[boundaries == 1, 2] = 1

		imsave(valid_dir + '/' + filename, img_as_ubyte(proc_mask))
		imsave(proc_dir + '/' + filename, binary_mask)


def partition_files(file_list, fraction_train=0.5):

	"""

	Finds all files within input_dir and paritions them into training
	and validation subsets

	Pseudo code
	----------
	1. Ensure fraction_train does not exceed 1
	2. Get images names and shuffle them
	3. Split names into training and testing sets

	Parameters
	----------

	input_dir: str,
		directory where files are stored
	fraction_train: float,
		fraction of images to use for training purposes

	"""

	# """
	# ~~~~~~~~~~~Error Check~~~~~~~~~~~~~~
	# """

	if fraction_train > 1:
		print("fraction_train + fraction_validation is > 1!")
		print("setting fraction_train = 0.5")
		fraction_train = 0.5

	# """
	# ~~~~~~~~~~~Get image names and shuffle~~~~~~~~~~~~~~
	# """

	fraction_test = 1 - fraction_train
	file_list = [x.split('.')[0] for x in file_list if x.endswith("tif") ]
	random.shuffle(file_list)

	# """
	# ~~~~~~~~~~~Split into training and testing~~~~~~~~~~~~~~
	# """

	ind = int(len(file_list)*fraction_train)
	train = file_list[:ind]; val = file_list[ind:]

	return (train, val)

def write_path_files(input_dir, file_list, name='files'):

	"""
	Utility function for writing lists of training and test data to txt files

	Parameters
	----------
	input_dir: str,
		directory where files are stored
	file_list: list,
		list of files
	name: str,
		name of the group of files in file_list

	"""

	path = join(input_dir, name + '.txt')
	with open(path, 'w') as f:
		for line in file_list:
			f.write(line + '\n')


def get_random_crop(input, target, crop_size):

	"""
	Takes an input and target image and crops them both to crop_size.
	The same patch is extracted from both input and target

	Pseudo code
	----------
	1. Unpack crop_size
	2. Extract patch from input and target

	Parameters
	----------
	input: ndarray,
		raw image
	target: ndarray,
		raw mask
	crop_size: tuple,
		dimensions for input_patch and target_patch

	"""

	dim1,dim2 = crop_size
	delta1 = np.random.randint(low=0, high=input.shape[0] - dim1)
	delta2 = np.random.randint(low=0, high=input.shape[1] - dim2)
	input_patch = input[delta1:delta1 + dim1, delta2:delta2 + dim2]
	target_patch = target[delta1:delta1 + dim1, delta2:delta2 + dim2]

	return input_patch, target_patch


def read_train_files(train_dir, crop_size):


	"""

	Pseudo code
	----------
	1. Create buffer for input and target tensors
	2. Populate input and target buffers

	Parameters
	----------
	input_dir: str,
		directory where images are stored
	target_dir: str,
		directory where masks are stored
	train_files: str,
		list of files to use for training
	crop_size: tuple,
		dimensions for input_patch and target_patch

	"""

	input_dir = train_dir + 'raw/'; target_dir = train_dir + 'target/'

	files = glob(input_dir + '*.tif'); n = len(files)
	input = np.zeros((n, *crop_size, 1))
	target = np.zeros((n, *crop_size, 3))

	for i, file in enumerate(files):

		file = file.split('/')[-1].split('.')[0]
		this_input = imread(join(input_dir, file + '.tif'))
		this_target = imread(join(target_dir, file + '.png'))

		this_input, this_target = get_random_crop(this_input, this_target,
												  crop_size=crop_size)
		input[i, :, :, 0] = this_input
		target[i, :, :, :] = this_target

	return input, target
