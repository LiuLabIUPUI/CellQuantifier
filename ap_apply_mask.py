"""Part I: CellQuantifier Sequence Control"""

control = [
'apply_mask',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['blobMask', 'mask'],

  #COLOCAL SETTINGS
  'Mask postfix': '-blobMask.tif',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_apply_mask import *
pipe_batch(settings, control)
