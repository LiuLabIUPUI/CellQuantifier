"""Part I: CellQuantifier Sequence Control"""

control = [
'get_boundary_focinum_df',
'generate_boundary_foci_map',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['Mask'],

  #SETTINGS
  'Ch1 detData label': '-ova-detData.csv',
  'Ch2 detData label': '-ova-detData.csv',
  'Boundary mask label': '-bdr-thresMask.tif',
  'Boundary outer thickness': 15,
  'Boundary inner thickness': 5,
  'Ch1 foci label': 'ova',
  'Ch2 foci label': 'xxx',


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_boundary_foci_num_1ch import *
pipe_batch(settings, control)
