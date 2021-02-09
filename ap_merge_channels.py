"""Part I: CellQuantifier Sequence Control"""

control = [
'merge_channels',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['-mhc1', '-ova', 'merge'],

  #COLOCAL SETTINGS
  'Ch1 index': 0,
  'Ch1 label': '-ova.tif',
  'Ch2 index': 1,
  'Ch2 label': '-mhc1.tif',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_merge_channels import *
pipe_batch(settings, control)
