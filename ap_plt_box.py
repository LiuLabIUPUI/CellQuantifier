"""Part I: CellQuantifier Sequence Control"""

control = [
'generate_boxplot',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/xx/',
  'Output path': '/home/linhua/Desktop/xx/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['Mask','temp', 'colocalmap', 'mask'],

  #SETTINGS
  'Ch1 label': '-mhc1',
  'Ch2 label': '-ova',
  'Ch1 mask dilation pixel': 1,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_box import *
pipe_batch(settings, control)
