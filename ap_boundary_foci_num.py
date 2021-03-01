"""Part I: CellQuantifier Sequence Control"""

control = [
'get_boundary_focinum_df',
'generate_boundary_foci_map',
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
  'Boundary mask label': '-thresMask.tif',
  'Boundary thickness': 5,
  'Ch1 foci label': 'mhc1',
  'Ch2 foci label': 'ova',


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_boundary_foci_num import *
pipe_batch(settings, control)
