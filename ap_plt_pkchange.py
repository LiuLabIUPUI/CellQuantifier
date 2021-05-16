"""Part I: CellQuantifier Sequence Control"""

control = [
'plot_peak_change',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '-detData.csv',
  'Strs not in filename': ['XXX'],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_pkchange import *
pipe_batch(settings, control)
