"""Part I: CellQuantifier Sequence Control"""

control = [
'fit_batch',
# 'anim_fitData',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '-deno.tif',
  'Strs not in filename': ['xxx','xxx'],
  'Pixel size': 0.108,

  #SETTINGS
  'max_dist_err': 10,
  'max_sig_to_sigraw': 20,
  'min_slope': 0,
  'min_mass': 0,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_fit import *
pipe_batch(settings, control)
