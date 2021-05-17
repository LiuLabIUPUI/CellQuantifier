"""Part I: CellQuantifier Sequence Control"""

control = [
'plt_colocal_change',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '-deno.tif',
  'Strs not in filename': ['xxxx','temp'],

  #SETTINGS
  'Ch1 label': 'green-deno',
  'Ch2 label': 'red-deno',
  'Ch1 mask dilation pixel': None,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_colocal_change import *
pipe_batch(settings, control)
