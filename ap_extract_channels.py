"""Part I: CellQuantifier Sequence Control"""

control = [
'extract_channels',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['-mhc1', '-ova'],

  #COLOCAL SETTINGS
  'Period (frames)': 4,
  'Ch1 index': 3,
  'Ch1 label': '-mhc1',
  'Ch2 index': 1,
  'Ch2 label': '-ova',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_extract_channels import *
pipe_batch(settings, control)
