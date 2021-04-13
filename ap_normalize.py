"""Part I: CellQuantifier Sequence Control"""

control = [
'normalize',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['-fft'],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_normalize import *
pipe_batch(settings, control)
