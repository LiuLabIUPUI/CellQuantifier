from skimage.io import imread, imsave

def split_tif(tif_path, interval_num):
	frames = imread(tif_path)
	start_ind = 0
	end_ind = start_ind + interval_num

	while end_ind <= len(frames):
		save_path = tif_path[:-4] + '-' + str(start_ind) + '-' + str(end_ind) + '.tif'
		imsave(save_path, frames[start_ind:end_ind])
		start_ind = end_ind
		end_ind = start_ind + interval_num
