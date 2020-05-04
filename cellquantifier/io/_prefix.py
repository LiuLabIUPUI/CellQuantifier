from glob import glob

def get_files_from_prefix(file, tags=['ch1-raw', 'ch2-raw']):

	"""
	Pseudo code
	----------
	1. Extract prefix from filename
	2. Return list of files which matching prefix

	Parameters
	----------
	file : str,
		Full path to file containing prefix

	tags : list,
		Identifiers of channels to use to find files

	"""

	split = file.split('/')
	input_dir, filename = '/'.join(split[:-1]) + '/', split[-1]
	prefix = '_'.join(filename.split('_')[:-1])

	fn_arr = [glob(input_dir + prefix + '_%s*' % tag) \
			  for tag in tags]

	fn_arr = [fn for sub_fn_arr in fn_arr for fn in sub_fn_arr]

	return fn_arr

def get_unique_prefixes(input_dir, tag):

	"""
	Pseudo code
	----------
	1. Get files containing tag
	2. Return list of unique prefixes

	Parameters
	----------
	input_dir : str,
		Directory containing files of interest

	tags : list,
		Identifies which files are of interest

	"""

	files = glob(input_dir + '*%s*'%(tag))
	roots = [file.split('/')[-1] for file in files]
	prefixes = ['_'.join(root.split('_')[:-1]) for root in roots]

	return prefixes
