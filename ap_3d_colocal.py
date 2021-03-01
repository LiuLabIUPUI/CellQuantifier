"""Part I: CellQuantifier Sequence Control"""

control = [
'calculate_colocal_df',
'generate_colocalmap',
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
  'Ch1 mask label': '-mhc1-blobMask-dilate.tif',
  'Ch1 coloc ratio name': 'mhc1 overlap ratio',
  'Ch2 coloc ratio name': 'ova overlap ratio',


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_3d_colocal import *
pipe_batch(settings, control)
