"""Part I: CellQuantifier Sequence Control"""

control = [
# 'load',
# 'mask_boundary',
# 'deno_gaus',
# 'check_detect_fit',
# 'detect',


# 'track_mrna',
'plot_mrna_traj',
# 'anim_mrna_traj',



# 'fit',
# 'filt_track',
# 'phys_dist2boundary',
# 'phys_antigen_data',
# 'plot_traj',
# 'anim_traj',
# 'merge_plot',

# 'phys_antigen_data2',
# 'plot_stub_hist',
# 'classify_antigen',
# 'plot_DM_traj',
# 'plot_BM_traj',
# 'plot_CM_traj',


]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'IO input_path': '/home/linhua/Desktop/temp/',
  'IO output_path': '/home/linhua/Desktop/temp/',
  'Processed By:': 'Hua Lin',
  'Pixel_size': 0.163,
  'Frame_rate': 2,

  #MASK_BOUNDARY SETTINGS
  'Mask boundary_mask file label': '-bdr',
  'Mask boundary_mask sig': 0,
  'Mask boundary_mask thres_rel': 0.05,

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 4,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.2,
  'Det blob_min_sigma': 5,
  'Det blob_max_sigma': 15,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': 0.1,

  #CELL TRACKING SETTINGS
  'mRNA trak search_range': 20,
  'mRNA trak memory': 5,
  'mRNA trak divide_num': 5,
  'mRNA traj_length_thres': 20,

  #FILTERING SETTINGS
  'Filt max_dist_err': 3,
  'Filt max_sig_to_sigraw': 6,
  'Filt max_delta_area': 2.4,
  'Filt traj_length_thres': 20,

  #TRACKING SETTINGS
  'Trak search_range': 10,
  'Trak memory': 3,
  'Trak divide_num': 5,

  #SORTING SETTINGS
  'Sort dist_to_boundary': [-10000, 70000],
  'Sort travel_dist': [-1000, 1000],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline_mrna import *
pipeline_batch(settings, control)
