"""Part I: CellQuantifier Sequence Control"""

control = [
'merge_df',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/output/',
  'Cols to merge': ['x', 'y', 'frame', 'r', 'dist_to_boundary'],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_merge_df import *
pipe_batch(settings, control)
