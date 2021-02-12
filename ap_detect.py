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
  'Frame number': 9,

  #SETTINGS
  'Blob_thres_rel': 0.7,
  'Blob_min_sigma': 2,
  'Blob_max_sigma': 5,
  'Blob_num_sigma': 5,
  'Blob_pk_thresh_rel': 0.7,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_detect import *
pipe_batch(settings, control)
