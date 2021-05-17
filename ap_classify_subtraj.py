"""Part I: CellQuantifier Sequence Control"""

control = [
'classify_subtraj',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '-physData.csv',
  'Strs not in filename': ['XXX'],

  #FOCI TRACKING SETTINGS
  'Pixel size': 1,
  'Frame rate': 1,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_classify_subtraj import *
pipe_batch(settings, control)
