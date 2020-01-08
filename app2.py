from datetime import datetime
from cellquantifier.util.pipeline2 import *
"""Part I: CellQuantifier Sequence Control"""

control = ['load', 'deno_box', 'deno_gaus', 'detect_fit',
            'filt_track', 'sort_plot']

"""Part II: CellQuantifier Parameter Settings"""
settings = {

  #HEADER INFO
  'Processed By:': 'Hua Lin',
  'Date': datetime.now(),
  'Raw data file': 'simulated_cell2.tif',
  'Start frame index': 0,
  'End frame index': 10,
  'Check frame index': 0,

  #IO
  'IO input_path': '/home/linhua/Desktop/input/',
  'IO output_path': '/home/linhua/Desktop/temp/',

  #REGISTRATION SETTINGS
  'Regi ref_ind_num': 0,
  'Regi sig_mask': 3,
  'Regi thres_rel': .1,
  'Regi poly_deg': 2,
  'Regi rotation_multplier': 1,
  'Regi translation_multiplier': 1,

  #SEGMENTATION SETTINGS
  'Segm min_size': 'NA',
  'Segm threshold': 'NA',

  #MASK SETTINGS
  'Phys dist2boundary_mask file': 'simulated_cell2.tif',
  'Phys dist2boundary_mask sig': 3,
  'Phys dist2boundary_mask thres_rel': 0.08,
  'Phys dist253bp1_mask file': 'simulated_cell2.tif',
  'Phys dist253bp1_mask sig': 1,
  'Phys dist253bp1_mask thres_rel': 0.35,

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.08,
  'Det blob_min_sigma': 2,
  'Det blob_max_sigma': 4,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': 0.15,
  'Det plot_r': False,

  #FITTING SETTINGS
  'Fitt r_to_sigraw': 1, #determines the patch size

  #TRACKING SETTINGS
  'Trak frame_rate': 3.33,
  'Trak pixel_size': 0.1084,
  'Trak divide_num': 1,
  'Trak search_range': 2,
  'Trak memory': 3,

  #FITTING FILTERING SETTINGS
  'Filt do_filter': True,
  'Filt max_dist_err': 10,
  'Filt max_sig_to_sigraw': 10,
  'Filt max_delta_area': 10,
  'Filt traj_length_thres': 3,

  #PHYSICS SETTINGS


  #SORTING SETTINGS
  'Sort do_sort': False,
  'Sort dist_to_boundary': [-150, 0],
  'Sort dist_to_53bp1': [-50, 10],

  #DIAGNOSTIC
  'Diag diagnostic': False,
  'Diag pltshow': False,

}

"""Part III: Run CellQuantifier"""
pipeline_batch(settings, control)
