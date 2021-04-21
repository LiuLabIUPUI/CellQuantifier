"""Part I: CellQuantifier Sequence Control"""

control = [
'plot_3d_foci_1ch',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '-detData.csv',
  'Strs not in filename': ['Mask'],

  #SETTINGS
  'Ch1 detData label': '-ova-detData.csv',
  'Pixel size': 0.108,
  'Z stack size': 0.5,
  'Min frame number': 0,
  'If plot even layer only': False,

  'If_plot_boundary': True,
  'Boundary label': '-bdr-thresMask.tif',

  'Boudary focinum df label': '-bdr-foci-num.csv',


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_plt_3dfoci_1ch import *
pipe_batch(settings, control)
