import keras
from keras.models import *
from keras.layers import *
from _unet_model_utils import *

def unet_model(input_size,
			   loss,
			   optimizer,
			   metrics=[]):

	"""
	Builds the U-Net architecture in keras

	Pseudo code
	----------

	Parameters
	----------

	"""

	option_dict_conv = {"activation": "relu", "padding": "same"}
	option_dict_bn = {"mode": 0, "momentum" : 0.9}

	# """
	# ~~~~~~~~~~~conv 1~~~~~~~~~~~~~~
	# """

	inputs = Input(input_size)
	conv1 = Conv2D(64, 3, **option_dict_conv)(inputs)
	conv1 = Conv2D(64, 3, **option_dict_conv)(conv1)

	# """
	# ~~~~~~~~~~~max-pool 1~~~~~~~~~~~~~~
	# """

	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	# """
	# ~~~~~~~~~~~conv 2~~~~~~~~~~~~~~
	# """

	conv2 = Conv2D(128, 3, **option_dict_conv)(pool1)
	conv2 = Conv2D(128, 3, **option_dict_conv)(conv2)

	# """
	# ~~~~~~~~~~~max-pool 2~~~~~~~~~~~~~~
	# """

	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	# """
	# ~~~~~~~~~~~conv 3~~~~~~~~~~~~~~
	# """

	conv3 = Conv2D(256, 3, **option_dict_conv)(pool2)
	conv3 = Conv2D(256, 3, **option_dict_conv)(conv3)

	# """
	# ~~~~~~~~~~~max-pool 3~~~~~~~~~~~~~~
	# """

	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	# """
	# ~~~~~~~~~~~conv 4~~~~~~~~~~~~~~
	# """

	conv4 = Conv2D(512, 3, **option_dict_conv)(pool3)
	conv4 = Conv2D(512, 3, **option_dict_conv)(conv4)
	drop4 = Dropout(0.5)(conv4)

	# """
	# ~~~~~~~~~~~max-pool 4~~~~~~~~~~~~~~
	# """

	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	# """
	# ~~~~~~~~~~~conv 5~~~~~~~~~~~~~~
	# """

	conv5 = Conv2D(1024, 3, **option_dict_conv)(pool4)
	conv5 = Conv2D(1024, 3, **option_dict_conv)(conv5)
	drop5 = Dropout(0.5)(conv5)

	# """
	# ~~~~~~~~~~up-conv 6~~~~~~~~~~~~~~
	# """

	up6 = Conv2D(512, 2, **option_dict_conv)(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)

	# """
	# ~~~~~~~~~~conv 6~~~~~~~~~~~~~~
	# """

	conv6 = Conv2D(512, 3, **option_dict_conv)(merge6)
	conv6 = Conv2D(512, 3, **option_dict_conv)(conv6)

	# """
	# ~~~~~~~~~~up-conv 7~~~~~~~~~~~~~~
	# """

	up7 = Conv2D(256, 2, **option_dict_conv)(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)

	# """
	# ~~~~~~~~~~conv 7~~~~~~~~~~~~~~
	# """

	conv7 = Conv2D(256, 3, **option_dict_conv)(merge7)
	conv7 = Conv2D(256, 3, **option_dict_conv)(conv7)

	# """
	# ~~~~~~~~~~up-conv 8~~~~~~~~~~~~~~
	# """

	up8 = Conv2D(128, 2, **option_dict_conv)(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)

	# """
	# ~~~~~~~~~~conv 8~~~~~~~~~~~~~~
	# """

	conv8 = Conv2D(128, 3, **option_dict_conv)(merge8)
	conv8 = Conv2D(128, 3, **option_dict_conv)(conv8)

	# """
	# ~~~~~~~~~~up-conv 9~~~~~~~~~~~~~~
	# """

	up9 = Conv2D(64, 2, **option_dict_conv)(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)

	# """
	# ~~~~~~~~~~conv 9~~~~~~~~~~~~~~
	# """

	conv9 = Conv2D(64, 3, **option_dict_conv)(merge9)
	conv9 = Conv2D(64, 3, **option_dict_conv)(conv9)
	conv9 = Conv2D(2, 3, **option_dict_conv)(conv9)

	# """
	# ~~~~~~~~~~output-conv~~~~~~~~~~~~~~
	# """

	conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

	# """
	# ~~~~~~~~~~compile model~~~~~~~~~~~~~~
	# """

	model = Model(input=inputs, output=conv10)

	model.compile(optimizer=optimizer,
				  loss=loss,
				  metrics=metrics)

	return model
