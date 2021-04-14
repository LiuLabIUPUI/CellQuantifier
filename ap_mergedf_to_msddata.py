"""Part I: CellQuantifier Sequence Control"""

control = [
'mergedf_to_msddata',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/xx/',
  'Output path': '/home/linhua/Desktop/xx/',
  'Processed by:': 'Hua Lin',
  'Str in filename': 'DataMerged.csv',
  'Strs not in filename': ['-mhc1', '-ova'],

  #SETTINGS
  'pixel_size': 0.108,
  'frame_rate': 33.3,
  'traj_length_thres': 40,


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_mergedf_to_msddata import *
pipe_batch(settings, control)
