"""Part I: CellQuantifier Sequence Control"""

control = [
'stack_to_images',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['-ova'],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_stack_to_images import *
pipe_batch(settings, control)
