"""Part I: CellQuantifier Sequence Control"""

control = [
'images_to_stack',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '*frame*.tif',
  'Strs not in filename': ['-ova'],

  'Postfix label': '-thresMask',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_images_to_stack import *
pipe_batch(settings, control)
