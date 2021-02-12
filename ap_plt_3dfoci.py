"""Part I: CellQuantifier Sequence Control"""

control = [
'plot_3d_foci',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['mask'],

  #SETTINGS
  'Ch1 detData label': '-mhc1-detData.csv',
  'Ch2 detData label': '-ova-detData.csv',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_3dfoci import *
pipe_batch(settings, control)
