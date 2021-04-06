"""Part I: CellQuantifier Sequence Control"""

control = [
'tif_to_pixel_intensity',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['-mhc1', '-ova'],

  #SETTINGS
  'Threshold': 20,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_tif_to_pixel_intensity import *
pipe_batch(settings, control)
