"""Part I: CellQuantifier Sequence Control"""

control = [
'fft_denoise',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/temp/',
  'Output path': '/home/linhua/Desktop/temp/',
  'Processed by:': 'Hua Lin',
  'Str in filename': '.tif',
  'Strs not in filename': ['-fft'],

  #SETTINGS
  

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_fftdeno import *
pipe_batch(settings, control)
