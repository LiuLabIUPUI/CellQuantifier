from datetime import datetime
from cellquantifier.util.pipeline import pipeline_control
"""Part I: CellQuantifier Sequence Control"""
CF0 = 1         #load + regi
CF1 = 1         #(load) + deno
CF2 = 1         #check
CF3 = 1         #det_fit
CF4 = 1         #filt_plot
Automate = 1    #automate smt
#                              load regi deno check det_fit filt_plot video
if CF0:        Control_Flow = [   1,   1,   0,   0,    0,    0,         0]
if CF1:        Control_Flow = [   0,   0,   1,   0,    0,    0,         0]
if CF2:        Control_Flow = [   0,   0,   0,   1,    0,    0,         0]
if CF3:        Control_Flow = [   0,   0,   0,   0,    1,    0,         1]
if CF4:        Control_Flow = [   0,   0,   0,   0,    0,    1,         0]
if Automate:   Control_Flow = [   1,   1,   1,   1,    1,    1,         1]

"""Part II: CellQuantifier Parameter Settings"""
settings = {

  #HEADER INFO
  'Processed By:': 'Hua Lin',
  'Date': datetime.now(),
  'Raw data file': 'simulated_cell',
  'Start frame index': 0,
  'End frame index': 50,
  'Check frame index': 0,

  #IO
  # 'IO input_path': '/home/linhua/cq/191004_NcBLM50_A1-HT.tif',
  'IO input_path': 'cellquantifier/data/simulated_cell.tif',
  'IO output_path': '/home/linhua/Desktop/temp/',

  #REGISTRATION SETTINGS
  'Regi ref_ind_num': 0,
  'Regi sig_mask': 3,
  'Regi thres_rel': .2,
  'Regi poly_deg': 2,
  'Regi rotation_multplier': 1,
  'Regi translation_multiplier': 1,

  #SEGMENTATION SETTINGS
  'Segm min_size': 'NA',
  'Segm threshold': 'NA',

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.1,
  'Det blob_min_sigma': 2,
  'Det blob_max_sigma': 4,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': .1,
  'Det plot_r': False,

  #FITTING SETTINGS
  'Fitt r_to_sigraw': 1, #determines the patch size

  #TRACKING SETTINGS
  'Trak frame_rate': 20,
  'Trak pixel_size': 0.163,
  'Trak divide_num': 1,
  'Trak search_range': 3,
  'Trak memory': 3,

  #FITTING FILTERING SETTINGS
  'Filt do_filter': False,
  'Filt max_dist_err': 1,
  'Filt max_delta_area': 0.3,
  'Filt sig_to_sigraw': 3,
  'Filt traj_length_thres': 25,

  #DIAGNOSTIC
  'Diag diagnostic': False,
  'Diag pltshow': False,

}

"""Part III: Run CellQuantifier"""
label = ['load', 'regi', 'deno', 'check', 'detect_fit', 'filt_plot', 'video']
control = dict(zip(label, Control_Flow))
pipeline_control(settings, control)
