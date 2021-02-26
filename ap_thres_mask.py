"""Part I: CellQuantifier Sequence Control"""

control = [
'generate_thres_mask',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['Mask'],

  #SETTINGS
  'Mask thres_rel': 0.99,
  'Mask sig': 0.5,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_thres_mask import *
pipe_batch(settings, control, load_configFile=False)
