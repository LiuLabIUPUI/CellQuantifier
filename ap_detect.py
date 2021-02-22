"""Part I: CellQuantifier Sequence Control"""

control = [
'detect',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['Mask','temp'],
  'Pixel size': 0.108,
  'Frame number': 16,

  #SETTINGS
  'Blob_thres_rel': 0.6,
  'Blob_min_sigma': 2,
  'Blob_max_sigma': 5,
  'Blob_num_sigma': 5,
  'Blob_pk_thresh_rel': 0,

  'If_fit': False,
  'max_dist_err': 2,
  'max_sig_to_sigraw': 3,
  'min_slope': 1,
  'min_mass': 0,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_detect import *
pipe_batch(settings, control)
