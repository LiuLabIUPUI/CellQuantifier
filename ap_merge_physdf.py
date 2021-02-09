"""Part I: CellQuantifier Sequence Control"""

control = [
'merge_physdf',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': 'physData.csv',
  'Strs not in filename': ['physDataMerged.csv'],

  #SETTINGS
  'Merge mode': 'general',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_merge_physdf import *
pipe_batch(settings, control)
