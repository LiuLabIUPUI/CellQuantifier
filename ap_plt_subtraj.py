"""Part I: CellQuantifier Sequence Control"""

control = [
'plot_subtraj',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': 'deno.tif',
  'Strs not in filename': ['xxx','xxx'],

  #PLOT SUBTRAJ SETTINGS
  'Subtraj type': 'CM',
  'Subtraj length thres': None,
  'Subtraj travel min distance': None,
  'Subtraj travel max distance': None,


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_subtraj import *
pipe_batch(settings, control)
