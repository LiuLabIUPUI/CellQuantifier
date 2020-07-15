import os
import numpy as np
import requests
import random
import warnings
from os.path import join
from glob import glob
from skimage.io import *
from skimage.morphology import *
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte
from zipfile import ZipFile
from numpy.random import randint

import keras.backend as k
import tensorflow as tf
from tensorflow.compat.v1.nn import softmax_cross_entropy_with_logits_v2
import keras
from keras.models import *
from keras.layers import *

CONST_DO_RATE = 0.5
option_dict_conv = {"activation": "relu", "border_mode": "same"}
option_dict_bn = {"mode": 0, "momentum" : 0.9}

def train_model(train_gen,
				valid_gen,
				crop_size,
				save_dir=None,
				metrics=None,
				optimizer=None,
				loss=None,
				epochs=10,
				steps_per_epoch=None,
				callbacks=1,
				verbose=1):

	"""
	Builds a new model that can be used to segment images

	Pseudo code
	----------
	1. Check if metrics have been specified
	2. Check if optimizer has been specified
	3. Partition files into training and testing
	4. Instantiate a U-Net model
	5. Train the U-Net model
	6. Display statistics

	Parameters
	----------

	input_dir: str
		directory containing the training and testing images
	target_dir: str
		directory containing the training and testing masks
	crop_size: str
		the size of the patch extracted from each image,mask
	metrics: list
		a list of keras metrics used to judge model performance
	optimizer: keras optimizer object
		optimization method e.g. gradient descent
	loss: keras loss-function object
		mathematical loss-function to be minimized during training
	batch_size: int
		number of images to train on in each epoch
	epochs: int
		number of epochs to train for
	fraction_validation: float
		fraction of training data to use for validation
	callbacks: str,
	 utilities called at certain points during model trainin
	verbose: int
		verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

	"""

	# """
	# ~~~~~~~~~~Fill default params~~~~~~~~~~~~~~
	# """

	if not metrics:
		metrics = [keras.metrics.categorical_accuracy,
				   channel_recall(channel=0, name="background_recall"),
				   channel_precision(channel=0, name="background_precision"),
				   channel_recall(channel=1, name="interior_recall"),
				   channel_precision(channel=1, name="interior_precision"),
				   channel_recall(channel=2, name="boundary_recall"),
				   channel_precision(channel=2, name="boundary_precision"),
				  ]

	if not optimizer:
			optimizer = keras.optimizers.RMSprop(lr=1e-4)


	# """
	# ~~~~~~~~~~Train the u-net model~~~~~~~~~~~~~~
	# """

	model = unet_model(input_size=crop_size); model.summary()
	model.compile(loss=weighted_crossentropy, metrics=metrics, optimizer=optimizer)

	statistics = model.fit(x=train_gen, epochs=epochs,
						   steps_per_epoch=steps_per_epoch,
						   validation_data=valid_gen)

	stats_df = pd.DataFrame(statistics.history)

	if save_dir:
		stats_df.to_csv(save_dir + '/train_stats.csv')
		model.save(save_dir + '/model.h5')

def make_prediction(stack, model, output_path):

	"""
	Given an image and a model, makes a prediction for the segmentation
	mask

	Pseudo code
	----------

	Parameters
	----------
	im_arr: list
		a stack of images to make predictions on
	model: model weights
		the weights to plugin to model generate by unet_model()

	"""

	# """
	# ~~~~~~~~~~Check input and reshape~~~~~~~~~~~~~~
	# """

	if len(stack.shape) < 3:
		stack = stack.reshape((1,) + stack.shape)

	stack = stack.reshape(stack.shape + (1,))
	prediction = model.predict(stack, batch_size=1)

	# """
	# ~~~~~~~~~~Transform prediction to label matrices~~~~~~~~
	# """

	for i in range(len(prediction)):
		probmap = prediction[i].squeeze()
		pred = probmap_to_pred(probmap)
		out = pred_to_label(pred)

		# """
		# ~~~~~~~~~~Convert to ImageJ format, output~~~~~~~~
		# """

		out = img_as_ubyte(out); shape = out.shape
		imsave(output_path + '/out%s.tif' % str(i), out)

def probmap_to_pred(probmap, boundary_boost_factor=1):

    pred = np.argmax(probmap * [1, 1, boundary_boost_factor], -1)

    return pred

def pred_to_label(pred, cell_label=1):

    cell = (pred == cell_label)
    [lbl, num] = label(cell, return_num=True)

    return lbl

def show_train_stats(path):

    df = pd.read_csv(path)
    fig,ax = plt.subplots(2,3)


    # """
    # ~~~~~~~~~~bg-recall~~~~~~~~~~~~~~
    # """

    ax[0,0].plot(df['val_background_recall'], color='red', label='Validation')
    ax[0,0].plot(df['background_recall'], color='blue', label='Train')
    ax[0,0].set_title(r'$\mathbf{Background}$', fontsize=10)
    format_ax(ax[0,0], ylabel=r'$\mathbf{Recall}$', ax_is_box=False, yscale=[0,1,None,None])
    ax[0,0].legend(loc='lower right')

    # """
    # ~~~~~~~~~~int-recall~~~~~~~~~~~~~~
    # """

    ax[0,1].plot(df['val_interior_recall'], color='red', label='Validation')
    ax[0,1].plot(df['interior_recall'], color='blue', label='Train')
    ax[0,1].set_title(r'$\mathbf{Interior}$', fontsize=10)
    format_ax(ax[0,1], ax_is_box=False, yscale=[0,1,None,None])
    ax[0,1].legend(loc='lower right')

    # """
    # ~~~~~~~~~~boundary-recall~~~~~~~~~~~~~~
    # """

    ax[0,2].plot(df['val_boundary_recall'], color='red', label='Validation')
    ax[0,2].plot(df['boundary_recall'], color='blue', label='Train')
    ax[0,2].set_title(r'$\mathbf{Boundary}$', fontsize=10)
    format_ax(ax[0,2], ax_is_box=False, yscale=[0,1,None,None])
    ax[0,2].legend(loc='lower right')
    # """
    # ~~~~~~~~~~bg-precision~~~~~~~~~~~~~~
    # """

    ax[1,0].plot(df['val_background_precision'], color='red', label='Validation')
    ax[1,0].plot(df['background_precision'], color='blue', label='Train')
    format_ax(ax[1,0], ylabel=r'$\mathbf{Precision}$', ax_is_box=False, yscale=[0,1,None,None])
    ax[1,0].legend(loc='lower right')

    # """
    # ~~~~~~~~~~int-precision~~~~~~~~~~~~~~
    # """

    ax[1,1].plot(df['val_interior_precision'], color='red', label='Validation')
    ax[1,1].plot(df['interior_precision'], color='blue', label='Train')
    format_ax(ax[1,1], ax_is_box=False, yscale=[0,1,None,None])
    ax[1,1].legend(loc='lower right')

    # """
    # ~~~~~~~~~~boundary-precision~~~~~~~~~~~~~~
    # """

    ax[1,2].plot(df['val_boundary_precision'], color='red', label='Validation')
    ax[1,2].plot(df['boundary_precision'], color='blue', label='Train')
    format_ax(ax[1,2], ax_is_box=False, yscale=[0,1,None,None])
    ax[1,2].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def unet_model(input_size, activation='softmax'):

	"""
	Builds the U-Net architecture in keras

	Pseudo code
	----------

	Parameters
	----------

	"""

	[x, y] = get_core(input_size)

	y = keras.layers.Convolution2D(3, 1, 1, **option_dict_conv)(y)

	if activation is not None:
		y = keras.layers.Activation(activation)(y)

	model = keras.models.Model(x, y)

	return model

def get_core(input_size):

	dim1, dim2 = input_size

	x = keras.layers.Input(shape=(dim1, dim2, 1))

	a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(x)
	a = keras.layers.BatchNormalization(**option_dict_bn)(a)

	a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(a)
	a = keras.layers.BatchNormalization(**option_dict_bn)(a)

	y = keras.layers.MaxPooling2D()(a)

	b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
	b = keras.layers.BatchNormalization(**option_dict_bn)(b)

	b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(b)
	b = keras.layers.BatchNormalization(**option_dict_bn)(b)

	y = keras.layers.MaxPooling2D()(b)

	c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
	c = keras.layers.BatchNormalization(**option_dict_bn)(c)

	c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(c)
	c = keras.layers.BatchNormalization(**option_dict_bn)(c)

	y = keras.layers.MaxPooling2D()(c)

	d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(y)
	d = keras.layers.BatchNormalization(**option_dict_bn)(d)

	d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(d)
	d = keras.layers.BatchNormalization(**option_dict_bn)(d)


	# UP

	d = keras.layers.UpSampling2D()(d)

	y = keras.layers.merge.concatenate([d, c], axis=3)

	e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
	e = keras.layers.BatchNormalization(**option_dict_bn)(e)

	e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(e)
	e = keras.layers.BatchNormalization(**option_dict_bn)(e)

	e = keras.layers.UpSampling2D()(e)


	y = keras.layers.merge.concatenate([e, b], axis=3)

	f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
	f = keras.layers.BatchNormalization(**option_dict_bn)(f)

	f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(f)
	f = keras.layers.BatchNormalization(**option_dict_bn)(f)

	f = keras.layers.UpSampling2D()(f)

	y = keras.layers.merge.concatenate([f, a], axis=3)

	y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
	y = keras.layers.BatchNormalization(**option_dict_bn)(y)

	y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
	y = keras.layers.BatchNormalization(**option_dict_bn)(y)

	return [x, y]

def channel_precision(channel, name):

	"""
	Wraps the channel precision metric to evaluate the segmentation
	precision for different channels such as interior and boundary

	Pseudo code
	----------
	1. Define precision_func
	2.

	Parameters
	----------
	channel: 2D ndarray,
		A single channel from a multi-channel tensor. Each channel may
		represent the interior, boundary, etc.

	name: str,
		A name to assign to the metric e.g 'boundary_precision'

	y_true: ndarray,
		The true segmentation result

	y_pred: ndarray,
		The segmentation result predicted by the network

	"""

	def precision_func(y_true, y_pred):

		y_pred_tmp = k.cast(tf.equal(k.argmax(y_pred, axis=-1), channel), "float32")
		true_positives = k.sum(k.round(k.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
		predicted_positives = k.sum(k.round(k.clip(y_pred_tmp, 0, 1)))
		precision = true_positives / (predicted_positives + k.epsilon())

		return precision

	precision_func.__name__ = name

	return precision_func

def channel_recall(channel, name):

	"""
	Wraps the channel recall metric to evaluate the segmentation
	recall for different channels such as interior and boundary

	Pseudo code
	----------
	1. Define precision_func
	2.

	Parameters
	----------
	channel: 2D ndarray,
		A single channel from a multi-channel tensor. Each channel may
		represent the interior, boundary, etc.

	name: str,
		A name to assign to the metric e.g 'boundary_precision'

	y_true: ndarray,
		The true segmentation result

	y_pred: ndarray,
		The segmentation result predicted by the network

	"""

	def recall_func(y_true, y_pred):

		y_pred_tmp = k.cast(tf.equal( k.argmax(y_pred, axis=-1), channel), "float32")
		true_positives = k.sum(k.round(k.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
		possible_positives = k.sum(k.round(k.clip(y_true[:,:,:,channel], 0, 1)))
		recall = true_positives / (possible_positives + k.epsilon())

		return recall

	recall_func.__name__ = name

	return recall_func

def weighted_crossentropy(y_true, y_pred):

	"""
	Defines the weighted_crossentropy loss function

	Pseudo code
	----------
	1. Define precision_func
	2.

	Parameters
	----------

	y_true: ndarray,
		The true segmentation result

	y_pred: ndarray,
		The segmentation result predicted by the network

	"""

	class_weights = tf.constant([[[[1., 1., 10.]]]])
	unweighted_losses = softmax_cross_entropy_with_logits_v2(labels=y_true,
															 logits=y_pred)

	weights = tf.reduce_sum(class_weights*y_true, axis=-1)
	weighted_losses = weights*unweighted_losses
	loss = tf.reduce_mean(weighted_losses)

	return loss

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

def partition_files(input_dir, target_dir, fraction_validation=0.25):

	"""

	Couples input_files to target_files and partitions into training
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

	if fraction_validation > 1:
		print("fraction_train + fraction_validation is > 1!")
		print("setting fraction_train = 0.5")
		fraction_validation = 0.5

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	input_files = glob(input_dir + '*.tif')
	target_files = glob(target_dir + '*.png')
	all_files = np.array([sorted(input_files), \
						  sorted(target_files)]).transpose()

	random.shuffle(all_files)

	# """
	# ~~~~~~~~~~~Split into training and testing~~~~~~~~~~~~~~
	# """

	fraction_train = 1 - fraction_validation
	ind = int(len(all_files)*fraction_train)
	train = all_files[:ind]; valid = all_files[ind:].transpose()

	return (train, valid)

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

def train_stack_gen(train_data, crop_size, batch_size=10, channels=3):


	"""

	Pseudo code
	----------
	1. Create buffer for input and target images
	2. Grab a random image (iteratively)
	3. Extract a random crop from the randomly selected image
	4. Add the random crop to the training buffer

	Parameters
	----------
	train_data: list,
		list of tuples containing (input, target) file names
	crop_size: tuple,
		dimensions for input_patch and target_patch

	"""

	while True:

		input_buffer = np.zeros((batch_size, *crop_size, 1))
		target_buffer = np.zeros((batch_size, *crop_size, channels))

		for i in range(batch_size):

			rand_ind = randint(low=0, high=len(train_data))
			input_path, target_path = train_data[rand_ind]
			input = imread(input_path); target = imread(target_path)

			input, target = get_random_crop(input, target, crop_size=crop_size)
			input_buffer[i, :, :, 0] = input; target_buffer[i, :, :, :] = target

		yield (input_buffer, target_buffer)

def valid_stack_gen(valid_data, crop_size, channels=3):


	"""

	Generates a tuple of nump

	Pseudo code
	----------


	Parameters
	----------
	val_data: list,
		list of tuples containing (input, target) file names
	crop_size: tuple,
		dimensions for input_patch and target_patch

	"""

	n = len(valid_data[0])
	input, target = valid_data
	input = np.array([imread(f) for f in input])
	target = np.array([imread(f) for f in target])

	dim1, dim2 = crop_size
	input = input[:, :dim1, :dim2]
	input = input.reshape(input.shape + (1,))
	target = target[:, :dim1, :dim2]

	return (input, target)
