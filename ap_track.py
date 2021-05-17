"""Part I: CellQuantifier Sequence Control"""

control = [
'filt_track',
'plot_traj',
# 'anim_traj',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'Input path': '/home/linhua/Desktop/input/',
  'Output path': '/home/linhua/Desktop/input/',
  'Processed by:': 'Hua Lin',
  'Str in filename': 'green-deno.tif',
  'Strs not in filename': ['xxx','xxx'],
  'Pixel size': 0.108,

  #FOCI FILTERING SETTINGS
  'Foci filt max_dist_err': 10,
  'Foci filt max_sig_to_sigraw': 20,
  'Foci filt max_delta_area': 8,
  'Foci filt traj_length_thres': 20,

  #FOCI TRACKING SETTINGS
  'Frame rate': 1,
  'Trak search_range': 20,
  'Trak memory': 5,
  'Trak divide_num': 5,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipe_track import *
pipe_batch(settings, control)
