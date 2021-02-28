"""Part I: CellQuantifier Sequence Control"""

control = [
'plot_3d_foci',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/xx/',
  'Output path': '/home/linhua/Desktop/xx/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['Mask'],

  #SETTINGS
  'Ch1 detData label': '-mhc1-detData.csv',
  'Ch2 detData label': '-ova-detData.csv',
  'Pixel size': 0.108,
  'Z stack size': 0.5,
  'Min frame number': 2,
  'If plot even layer only': True,

  'If_plot_boundary': True,
  'Boundary label': '-thresMask.tif',


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_3dfoci import *
pipe_batch(settings, control)
