"""Part I: CellQuantifier Sequence Control"""

control = [
'plot_subtraj_lifetime',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '-physData.csv',
  'Strs not in filename': ['XXX'],

  #PLOT SUBTRAJ SETTINGS
  'Subtraj type': 'CM',
  'Subtraj length thres': None,
  'Subtraj travel min distance': None,
  'Subtraj travel max distance': None,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_subtraj_lifetime import *
pipe_batch(settings, control)
