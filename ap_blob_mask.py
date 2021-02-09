"""Part I: CellQuantifier Sequence Control"""

control = [
'generate_blob_mask',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/XXXX/',
  'Output path': '/home/linhua/Desktop/XXXX/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['Mask','temp'],

  #SETTINGS
  'Mask blob_thres_rel': 0.1,
  'Mask blob_min_sigma': 3,
  'Mask blob_max_sigma': 5,
  'Mask blob_num_sigma': 5,
  'Mask blob_pk_thresh_rel': 0.1,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_blob_mask import *
pipe_batch(settings, control, load_configFile=False)
