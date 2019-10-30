import numpy as np
import skimage.morphology.watershed as skwatershed
import matplotlib.pyplot as plt
import centrosome.threshold
import skimage
import scipy

from scipy import sparse
from centrosome import cpmorphology, outline, propagate
from skimage.segmentation import clear_border, mark_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, binary_closing, selem, dilation, erosion, disk
from skimage.feature import peak_local_max, canny
from skimage.filters import unsharp_mask, gaussian
from skimage import img_as_ubyte
from scipy import ndimage as ndi

#==================================================================================================
# segm.py - segmentation engine - watershed, unsharp mask, canny edge detection, and unet functions
#==================================================================================================
# Author:  Clayton Seitz <cwseitz@iu.edu>
#          06/01/2019
#==================================================================================================


def segm_batch(config, im, method):

	masks = np.zeros(im.shape, dtype='uint8')
	centroid = None

	for i in range(im.shape[0]):

		if method == 'label':
			mask = tools.label_image(im[i], i)
		elif method == 'threshold':
			mask = threshold(im[i], config.ROI_SIZE)
		elif method == 'watershed':
			mask = watershed(im[i], config.ROI_SIZE, config.SEGM_THRESHOLD)
		elif method == 'edges':
			mask = edges(im[i], config, channel_id)
		elif method == 'unsharp':
			mask = unsharp(im[i], config, channel_id)

		mask = remove_small_objects(mask, min_size=config.ROI_SIZE[0])

		mask = erosion(mask, selem = disk(radius=10))
		mask = dilation(mask, selem = disk(radius=10))

		if config.SHOW_MASK:

			fig,ax = plt.subplots(1,2)
			ax[0].imshow(mask, cmap='gray')
			ax[1].imshow(mark_boundaries(im[i]*3, mask))
			plt.show()

		if i == 0:
			centroid = regionprops(mask)[0].centroid

		masks[i,:,:] = mask

	return masks, centroid

def segm(im, config, method):

	if method == 'label':
		mask = tools.label_image(im, i)
	elif method == 'threshold':
		mask = threshold(im, config.ROI_SIZE)
	elif method == 'watershed':
		mask = watershed(im, config.ROI_SIZE, config.SEGM_THRESHOLD)
	elif method == 'edges':
		mask = edges(im, config, channel_id)
	elif method == 'unsharp':
		mask = unsharp(im, config, channel_id)

	if config.SHOW_MASK:

		fig,ax = plt.subplots(1,2)
		ax[0].imshow(mask, cmap='gray')
		ax[1].imshow(mark_boundaries(im*3, mask))
		plt.show()

	centroid = regionprops(mask)[0].centroid

	return mask, centroid

def threshold(image, size_range):

	image = gaussian(image, sigma=3)
	t = .1*image.max()
	mask = image > t
	mask = mask.astype(int)
	mask = label(mask)

	return mask

def unsharp(image, channel_id):

	image = clear_border(image)
	mask = unsharp_mask(image, radius=100, amount=100)
	mask = label(mask)
	mask = remove_small_objects(mask, config.ROI_SIZE[int(channel_id)][0])
	mask = label(mask)

	return mask

def edge_detection(image, config, channel_id):

	image = gaussian(image)
	edges = canny(image)
	fill = ndi.binary_fill_holes(edges)
	label_objects, nb_labels = ndi.label(fill)
	mask = remove_small_objects(mask, config.ROI_SIZE[int(channel_id)][0])
	return cleaned



N_SETTINGS = 15

UN_INTENSITY = "Intensity"
UN_SHAPE = "Shape"
UN_LOG = "Laplacian of Gaussian"
UN_NONE = "None"

WA_INTENSITY = "Intensity"
WA_SHAPE = "Shape"
WA_PROPAGATE = "Propagate"
WA_NONE = "None"

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

FH_NEVER = "Never"
FH_THRESHOLDING = "After both thresholding and declumping"
FH_DECLUMP = "After declumping only"

def watershed(image, size_range, threshold):

	basic = True
	settings = get_settings()
	fill_holes = FH_THRESHOLDING
	threshold_operation = "Manual"

	binary_image, global_threshold, sigma = _threshold_image(image, threshold, False)

	def size_fn(size, is_foreground):
		return size < size_range[0] * size_range[1]

	if basic or fill_holes == FH_THRESHOLDING:
		binary_image = cpmorphology.fill_labeled_holes(binary_image, size_fn=size_fn)

	labeled_image, object_count = ndi.label(binary_image, np.ones((3, 3), bool))

	labeled_image, object_count, maxima_suppression_size = separate_neighboring_objects(
		image,
		labeled_image,
		object_count,
		size_range
	)

	unedited_labels = labeled_image.copy()

	# Filter out objects touching the border or mask
	border_excluded_labeled_image = labeled_image.copy()
	labeled_image = filter_on_border(image, labeled_image)
	border_excluded_labeled_image[labeled_image > 0] = 0

	# Filter out small and large objects
	size_excluded_labeled_image = labeled_image.copy()
	labeled_image, small_removed_labels = filter_on_size(labeled_image, object_count, size_range)
	size_excluded_labeled_image[labeled_image > 0] = 0

	if basic or fill_holes != FH_NEVER:
		labeled_image = cpmorphology.fill_labeled_holes(labeled_image)

	# Relabel the image
	labeled_image, object_count = cpmorphology.relabel(labeled_image)

	limit_choice = LIMIT_NONE
	maximum_object_count = 500
	if not basic and limit_choice == LIMIT_ERASE:
		if object_count > maximum_object_count:
			labeled_image = np.zeros(labeled_image.shape, int)
			border_excluded_labeled_image = np.zeros(labeled_image.shape, int)
			size_excluded_labeled_image = np.zeros(labeled_image.shape, int)
			object_count = 0

	# Make an outline image
	outline_image = outline.outline(labeled_image)
	outline_size_excluded_image = outline.outline(size_excluded_labeled_image)
	outline_border_excluded_image = outline.outline(border_excluded_labeled_image)

	return labeled_image

def get_settings():

  exclude_size = False
  exclude_border_objects = True
  unclump_method = "Intensity"
  watershed_method = "Propagate"
  automatic_smoothing = True
  smoothing_filter_size = 10
  automatic_suppression = False
  maxima_suppression_size = 7
  low_res_maxima = True
  fill_holes = "Never"
  limit_choice = "Continue"
  maximum_object_count = 500
  use_advanced = False

  settings = [
	exclude_size,
	exclude_border_objects,
	unclump_method,
	watershed_method,
	smoothing_filter_size,
	maxima_suppression_size,
	low_res_maxima,
	fill_holes,
	automatic_smoothing,
	automatic_suppression,
	limit_choice,
	maximum_object_count,
	use_advanced
]

  return settings

def smooth_image(image, size_range):

	filter_size = calc_smoothing_filter_size(size_range)

	if filter_size == 0:
		return image

	sigma = filter_size / 2.35

	filter_size = max(int(float(filter_size) / 2.0), 1)
	f = (1 / np.sqrt(2.0 * np.pi) / sigma *
		 np.exp(-0.5 * np.arange(-filter_size, filter_size + 1) ** 2 /
				   sigma ** 2))

	def fgaussian(image):
		output = ndi.convolve1d(image, f,
										  axis=0,
										  mode='constant')
		return ndi.convolve1d(output, f,
										axis=1,
										mode='constant')

	mask = image < 1000
	edge_array = fgaussian(mask.astype(float))
	masked_image = image.copy()
	masked_image[~mask] = 0
	smoothed_image = fgaussian(masked_image)
	masked_image[mask] = smoothed_image[mask] / edge_array[mask]
	return masked_image

def separate_neighboring_objects(image, labeled_image,
								 object_count, size_range):

	blurred_image = smooth_image(image, size_range)

	basic = True
	low_res_maxima = True
	automatic_suppression = True
	maxima_suppression_size = 7
	unclump_method = UN_INTENSITY
	watershed_method = WA_INTENSITY

	if size_range[0] > 10 and (basic or low_res_maxima):

		image_resize_factor = 10.0 / float(size_range[0])
		if basic or automatic_suppression:
			maxima_suppression_size = 7
		else:
			maxima_suppression_size = (maxima_suppression_size *
									   image_resize_factor + .5)
		reported_maxima_suppression_size = \
			maxima_suppression_size / image_resize_factor
	else:
		image_resize_factor = 1.0
		if basic or automatic_suppression:
			maxima_suppression_size = size_range[0] / 1.5
		else:
			maxima_suppression_size = maxima_suppression_size
		reported_maxima_suppression_size = maxima_suppression_size

	maxima_mask = cpmorphology.strel_disk(max(1, maxima_suppression_size - .5))
	distance_transformed_image = None


	if basic or unclump_method == UN_INTENSITY:

		maxima_image = get_maxima(blurred_image,
									   labeled_image,
									   maxima_mask,
									   image_resize_factor)

	else:
		raise ValueError("Unsupported local maxima method: %s" % unclump_method.value)

	# Create the image for watershed
	if basic or watershed_method == WA_INTENSITY:
		# use the reverse of the image to get valleys at peaks
		watershed_image = 1 - image
	elif watershed_method == WA_SHAPE:
		if distance_transformed_image is None:
			distance_transformed_image = \
				ndi.distance_transform_edt(labeled_image > 0)
		watershed_image = -distance_transformed_image
		watershed_image = watershed_image - np.min(watershed_image)

	elif watershed_method == WA_PROPAGATE:
		pass
	else:
		raise NotImplementedError("Watershed method %s is not implemented" % self.watershed_method.value)

	labeled_maxima, object_count = \
		ndi.label(maxima_image, np.ones((3, 3), bool))

	if not basic and watershed_method == WA_PROPAGATE:
		watershed_boundaries, distance = \
			propagate.propagate(np.zeros(labeled_maxima.shape),
										   labeled_maxima,
										   labeled_image != 0, 1.0)
	else:
		markers_dtype = (np.int16
						 if object_count < np.iinfo(np.int16).max
						 else np.int32)
		markers = np.zeros(watershed_image.shape, markers_dtype)
		markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]

		watershed_boundaries = skwatershed(
			connectivity=np.ones((3, 3), bool),
			image=watershed_image,
			markers=markers,
			mask=labeled_image != 0
		)

		watershed_boundaries = -watershed_boundaries

	return watershed_boundaries, object_count, reported_maxima_suppression_size

def get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
	if image_resize_factor < 1.0:
		shape = np.array(image.shape) * image_resize_factor
		i_j = (np.mgrid[0:shape[0], 0:shape[1]].astype(float) /
			   image_resize_factor)
		resized_image = ndi.map_coordinates(image, i_j)
		resized_labels = ndi.map_coordinates(
				labeled_image, i_j, order=0).astype(labeled_image.dtype)

	else:
		resized_image = image
		resized_labels = labeled_image

	if maxima_mask is not None:
		binary_maxima_image = cpmorphology.is_local_maximum(resized_image,
																	   resized_labels,
																	   maxima_mask)
		binary_maxima_image[resized_image <= 0] = 0
	else:
		binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
	if image_resize_factor < 1.0:
		inverse_resize_factor = (float(image.shape[0]) /
								 float(binary_maxima_image.shape[0]))
		i_j = (np.mgrid[0:image.shape[0],
			   0:image.shape[1]].astype(float) /
			   inverse_resize_factor)
		binary_maxima_image = ndi.map_coordinates(
				binary_maxima_image.astype(float), i_j) > .5
		assert (binary_maxima_image.shape[0] == image.shape[0])
		assert (binary_maxima_image.shape[1] == image.shape[1])

	shrunk_image = cpmorphology.binary_shrink(binary_maxima_image)
	return shrunk_image

def filter_on_size(labeled_image, object_count, size_range):

	exclude_size = True
	if exclude_size and object_count > 0:
		areas = ndi.measurements.sum(np.ones(labeled_image.shape),
											   labeled_image,
											   np.array(range(0, object_count + 1), dtype=np.int32))
		areas = np.array(areas, dtype=int)
		min_allowed_area = np.pi * (size_range[0] * size_range[0]) / 4
		max_allowed_area = np.pi * (size_range[1] * size_range[1]) / 4
		# area_image has the area of the object at every pixel within the object
		area_image = areas[labeled_image]
		labeled_image[area_image < min_allowed_area] = 0
		small_removed_labels = labeled_image.copy()
		labeled_image[area_image > max_allowed_area] = 0
	else:
		small_removed_labels = labeled_image.copy()
	return labeled_image, small_removed_labels

def filter_on_border(image, labeled_image):

	exclude_border_objects = True

	if exclude_border_objects:
		border_labels = list(labeled_image[0, :])
		border_labels.extend(labeled_image[:, 0])
		border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
		border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
		border_labels = np.array(border_labels)

		histogram = sparse.coo_matrix((np.ones(border_labels.shape),
											 (border_labels,
											  np.zeros(border_labels.shape))),
											shape=(np.max(labeled_image) + 1, 1)).todense()

		histogram = np.array(histogram).flatten()
		if any(histogram[1:] > 0):
			histogram_image = histogram[labeled_image]
			labeled_image[histogram_image > 0] = 0


	return labeled_image

def _threshold_image(image, threshold, automatic=False):

	binary_image, sigma = apply_threshold(image, threshold, automatic)

	return binary_image, threshold, sigma

def calc_smoothing_filter_size(size_range):

	automatic_smoothing = True

	if automatic_smoothing:
		return 2.35 * size_range[0] / 3.5


def apply_threshold(image, threshold, automatic=False):


        threshold_smoothing_scale = 0
        data = image

        mask = np.ones((len(image), len(image[0])), bool)

        if automatic:
            sigma = 1

        else:
            sigma = threshold_smoothing_scale / 0.6744 / 2.0

        blurred_image = centrosome.smooth.smooth_with_function_and_mask(
            data,
            lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
            mask
        )
        return (blurred_image >= threshold), sigma
