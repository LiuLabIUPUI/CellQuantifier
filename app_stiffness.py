"""Part I: CellQuantifier Sequence Control"""

control = [
# 'load',
# 'cell_denoise',
# 'check_cell_detection',
# 'detect_cell',
# 'track_cell',
# 'plot_cell_traj',
# 'anim_cell_traj',
# 'blob_segm',

# 'load',
# 'foci_denoise',
'check_foci_detection',
# 'detect_foci',
# 'plot_foci_dynamics',
# 'fit',
# 'plot_foci_dynamics2',
# 'merge_plot',

]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #IO
  'IO input_path': '/home/linhua/Desktop/temp/',
  'IO output_path': '/home/linhua/Desktop/temp/',
  'Processed By:': 'Hua Lin',
  'Pixel_size': 0.108,

  #CELL DENOISE SETTINGS
  'Cell deno minimum_radius': 7,

  #CELL DETECTION SETTINGS
  'Cell det blob_thres_rel': 0.01,
  'Cell det blob_min_sigma': 20,
  'Cell det blob_max_sigma': 25,
  'Cell det blob_num_sigma': 3,
  'Cell det pk_thres_rel': 0.05,
  'Cell det r_to_sigraw': 2,

  #CELL TRACKING SETTINGS
  'Cell trak search_range': 50,
  'Cell trak memory': 5,
  'Cell traj_length_thres': 250,

  #FOCI DENOISE SETTINGS
  'Foci deno boxcar_radius': 10,
  'Foci deno gaus_blur_sig': 0.5,

  #FOCI DETECTION SETTINGS
  'Foci det blob_thres_rel': 0.2,
  'Foci det blob_min_sigma': 2,
  'Foci det blob_max_sigma': 3,
  'Foci det blob_num_sigma': 50,
  'Foci det pk_thres_rel': 0.2,
  'Foci det mass_thres_rel': 0,

  #FOCI FILTERING SETTINGS
  'Foci filt max_dist_err': 0.8,
  'Foci filt max_sig_to_sigraw': 1.5,
  'Foci filt max_delta_area': 1
}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline_stiffness import *
pipeline_batch(settings, control)
