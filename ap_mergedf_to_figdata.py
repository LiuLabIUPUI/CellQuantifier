"""Part I: CellQuantifier Sequence Control"""

control = [
'mergedf_to_figdata',
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
  'figData col name': 'D',
  'drop_duplicates': True,
  'traj_length_thres': 40,


}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_mergedf_to_figdata import *
pipe_batch(settings, control)
