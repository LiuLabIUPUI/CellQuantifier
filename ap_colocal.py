"""Part I: CellQuantifier Sequence Control"""

control = [
'dilate_Ch1_mask',
'generate_colocalmap',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/XXXX/',
  'Output path': '/home/linhua/Desktop/XXXX/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['Mask','temp', 'colocalmap'],

  #COLOCAL SETTINGS
  'Ch1 label': '-mhc1',
  'Ch2 label': '-ova',
  'Ch1 mask dilation pixel': 5,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_colocal import *
pipe_batch(settings, control)
