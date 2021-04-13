"""Part I: CellQuantifier Sequence Control"""

control = [
'denoise',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['detVideo'],

  #SETTINGS
  'Boxcar_radius': 10,
  'Gaus_blur_sig': 0.5,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_deno import *
pipe_batch(settings, control)
