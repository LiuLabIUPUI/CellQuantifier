"""Part I: CellQuantifier Sequence Control"""

control = [

# 'load',

'mask_boundary',
# 'mask_53bp1',
# 'mask_53bp1_blob',

# 'foci_denoise',
# 'check_foci_detection',
# 'detect_foci',
# 'fit',
# 'filt_track',
# 'plot_traj',
# 'phys',

# 'merge_plot',

]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #IO
  'IO input_path': '/home/linhua/Desktop/xx/',
  'IO output_path': '/home/linhua/Desktop/xx/',
  'Processed By:': 'Hua Lin',
  'Pixel_size': 0.163,
  'Frame_rate': 20,

  #MASK_BOUNDARY SETTINGS
  'Mask boundary_mask file label': '',
  'Mask boundary_mask sig': 3,
  'Mask boundary_mask thres_rel': 0.5,
  #MASK_53BP1 SETTINGS
  'Mask 53bp1_mask file label': '',
  'Mask 53bp1_mask sig': '',
  'Mask 53bp1_mask thres_rel': '',
  #MASK_53BP1_BLOB SETTINGS
  'Mask 53bp1_blob_mask file label': '',
  'Mask 53bp1_blob_threshold': '',
  'Mask 53bp1_blob_min_sigma': '',
  'Mask 53bp1_blob_max_sigma': '',
  'Mask 53bp1_blob_num_sigma': '',
  'Mask 53bp1_blob_pk_thresh_rel': '',
  'Mask 53bp1_blob_search_range': '',
  'Mask 53bp1_blob_memory': '',
  'Mask 53bp1_blob_traj_length_thres': '',

  #FOCI DENOISE SETTINGS
  'Foci deno boxcar_radius': 10,
  'Foci deno gaus_blur_sig': 0.5,

  #FOCI DETECTION SETTINGS
  'Foci det blob_thres_rel': 0.25,
  'Foci det blob_min_sigma': 2,
  'Foci det blob_max_sigma': 4,
  'Foci det blob_num_sigma': 5,
  'Foci det pk_thres_rel': 0.15,

  #FOCI FILTERING SETTINGS
  'Foci filt max_dist_err': 1,
  'Foci filt max_sig_to_sigraw': 2,
  'Foci filt max_delta_area': 0.8,
  'Foci filt traj_length_thres': 80,

  #FOCI TRACKING SETTINGS
  'Trak search_range': 2,
  'Trak memory': 3,
  'Trak divide_num': 5,

  #SORTING SETTINGS
  'Sort dist_to_boundary': [-10000, 10000],
  'Sort dist_to_53bp1': [-10000, 10000],
  'Sort travel_dist': [-10000, 10000],
}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline_nc import *
pipeline_batch(settings, control)
