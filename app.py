from datetime import datetime
from cellquantifier.util import config
from cellquantifier.util import pipeline

import warnings
warnings.filterwarnings("ignore")

settings = {

  #HEADER INFO
  'Processed By:': '',
  'Date': datetime.now(),
  'Raw data file': 'simulated_cell',
  'Start frame index': 0,
  'End frame index': 10,

   #IO
  'IO input_path': 'cellquantifier/data/simulated_cell.tif',
  'IO output_path': '/home/cwseitz/Desktop/temp/',

  #REGISTRATION SETTINGS
  'Reg ref_ind_num': 0,
  'Reg sig_mask': 3,
  'Reg thres_rel': .2,
  'Reg poly_deg': 2,
  'Reg rotation_multplier': 1,
  'Reg translation_multiplier': 1,

  #SEGMENTATION SETTINGS
  'Segm min_size': 50,
  'Segm threshold': 7,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.1,
  'Det blob_min_sigma': 1,
  'Det blob_max_sigma': 3,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': .1,
  'Det plot_r': False,

  #FITTING SETTINGS
  'Fitt r_to_sigraw': 3, #determines the patch size

  #TRACKING SETTINGS
  'Trak search_range': 5,
  'Trak memory': 10,
  'Trak min_traj_length': 5,
  'Trak frame_rate': 1,
  'Trak pixel_size': 1e-3,
  'Trak divide_num': 5,

  #FITTING FILTERING SETTINGS
  'Filt from_csv': False,
  'Filt do_filter': True,
  'Filt max_dist_err': 1,
  'Filt max_delta_area': .3,
  'Filt sig_to_sigraw': 3,
  'Filt traj_length_thres': 5,

  #DIAGNOSTIC
  'Diag diagnostic': True,
  'Diag pltshow': True,

}

config = config.Config(settings)
pipe = pipeline.Pipeline(config)

# pipe.register()
pipe.deno(method='boxcar', arg=10)
pipe.deno(method='gaussian', arg=3)
pipe.smt()
