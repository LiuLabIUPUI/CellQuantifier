"""Part I: CellQuantifier Sequence Control"""

control = [
# 'load',
# 'mask_boundary',
# 'denoise',
# 'check_detect_fit',
# 'detect',
# 'fit',
# 'filt_track',
# 'phys_dist2boundary',
# 'phys_antigen_data',
# 'plot_traj',
# 'anim_traj',

# 'phys_antigen_data2',
# 'classify_antigen',

# 'plot_DM_subtraj',
# 'plot_BM_subtraj',
# 'plot_CM_subtraj',

# 'plot_DM_traj',

# 'merge_plot',

]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'IO input_path': '/home/linhua/Desktop/all/',
  'IO output_path': '/home/linhua/Desktop/all/',
  'Processed By:': 'Hua Lin',
  'Pixel_size': 0.163,
  'Frame_rate': 2,

  #MASK_BOUNDARY SETTINGS
  'Mask boundary_mask file label': '-bdr',
  'Mask boundary_mask sig': 0,
  'Mask boundary_mask thres_rel': 0.05,

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.05,
  'Det blob_min_sigma': 2,
  'Det blob_max_sigma': 3,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': 0.1,

  #FILTERING SETTINGS
  'Filt max_dist_err': 3,
  'Filt max_sig_to_sigraw': 6,
  'Filt max_delta_area': 2.4,
  'Filt traj_length_thres': 20,

  #TRACKING SETTINGS
  'Trak search_range': 2.5,
  'Trak memory': 5,
  'Trak divide_num': 5,

  #SORTING SETTINGS
  'Sort dist_to_boundary': [-10000, 70000],
  'Sort travel_dist': [-1000, 1000],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline_antigen import *
pipeline_batch(settings, control)

# from skimage.io import imread, imsave
# from cellquantifier.publish import *
# from cellquantifier.video import *
# import pandas as pd
#
# df = pd.read_csv('/home/linhua/Desktop/temp/200205_MalKN-E-physData.csv',
#                 index_col=False)
# df = df[ df['traj_length']>20 ]
# fig_quick_antigen(df)


# df = pd.read_csv('/home/linhua/Desktop/temp-E/200205_MalKN-E-physData.csv',
#                 index_col=False)
# tif = imread('/home/linhua/Desktop/temp-E/200205_MalKN-E-raw.tif')[0:50]
# df = df[ (df['traj_length']>100) & (df['frame']<50) ]
# anim_tif = anim_traj(df, tif,
#                 pixel_size=0.163,
#                 cb_min=2000, cb_max=10000,
#                 cb_major_ticker=2000, cb_minor_ticker=2000,
#                 show_image=True)
# anim_tif = anim_blob(df, tif, pixel_size=0.163,
#                 show_image=False)
# imsave('/home/linhua/Desktop/temp-E/anim-traj-result.tif', anim_tif)
# fig_quick_antigen(df)


# from cellquantifier.publish import *
# import pandas as pd
#
# df = pd.read_csv('/home/linhua/Desktop/temp/200426_physDataMerged.csv',
#                 index_col=False)
# fig_quick_antigen_3(df)


# from cellquantifier.publish._fig_quick_antigen_5 import *
# import pandas as pd
# df = pd.read_csv('/home/linhua/Desktop/temp/201004-subparticlesMerged.csv',
#                 index_col=False)
# fig_quick_antigen_5(df)
