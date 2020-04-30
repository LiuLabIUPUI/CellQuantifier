from _unet_data_utils import *
from _unet_model import *
from _unet_model_utils import *
from _unet_vis import *
import keras
import pandas as pd

def build_new_model(input_dir,
					target_dir,
					crop_size,
					save_dir=None,
					metrics=None,
					optimizer=None,
					loss=None,
					batch_size=10,
					epochs=10,
					fraction_train=0.5,
					fraction_validation=0.25,
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
	fraction_train: float
		fraction of images to use for training purposes
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

	train_files = partition_files(input_dir,fraction_train=fraction_train)

	model = unet_model(input_size=crop_size); model.summary()
	model.compile(loss=weighted_crossentropy, metrics=metrics, optimizer=optimizer)

	input, target = read_train_files(input_dir=input_dir, target_dir=target_dir,
								     train_files=train_files,crop_size=crop_size)

	statistics = model.fit(x=input, y=target, batch_size=batch_size,
						  epochs=epochs, validation_split=fraction_validation)
	stats_df = pd.DataFrame(statistics.history)

	if save_dir:
		stats_df.to_csv(save_dir + '/train_stats.csv')
		model.save(save_dir + '/model.h5')
