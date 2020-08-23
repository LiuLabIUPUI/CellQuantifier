"""Part I: CellQuantifier Sequence Control"""

control = [
# 'load',
# 'deno_minimum',
# 'check_det',
# 'detect_cell',
# 'track_simple',
# 'plot_traj_simple',
# 'anim_traj_simple',
# 'blob_segm',

# 'check_det',
'deno_box',
'detect_foci',
'plot_foci_dynamics',
'fit',
'anim_foci',
# 'filt_track',
# 'plot_traj',
# 'anim_traj',


]

"""Part II: CellQuantifier Parameter Settings"""

settings = {
    #DETECTION SETTINGS
    'Det blob_threshold': 'auto',
    'Det blob_min_sigma': 2,
    'Det blob_max_sigma': 3,
    'Det blob_num_sigma': 50,
    'Det pk_thresh_rel': 'auto',
    'Det mean_thresh_rel': 0,
    'Det r_to_sigraw': 1,

  #IO
  'IO input_path': '/home/linhua/Desktop/input/',
  'IO output_path': '/home/linhua/Desktop/input/',

  #HEADER INFO
  'Processed By:': 'Hua Lin',
  'Start frame index': 0,
  'End frame index': 600,
  'Load existing analMeta': False,

  #DENOISE SETTINGS
  'Deno mean_radius': '',
  'Deno median_radius': '',
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,
  'Deno minimum_radius': 7,

  #TRACKING SETTINGS
  'Trak frame_rate': 1,
  'Trak pixel_size': 0.108,
  'Trak divide_num': 1,

  ###############################################
  'Trak search_range': 7,  # NO. 1
  ###############################################

  'Trak memory': 3,

  #FILTERING SETTINGS
  'Filt max_dist_err': 1,
  'Filt max_sig_to_sigraw': 2,
  'Filt max_delta_area': 1,

  ###############################################
  'Filt traj_length_thres': 3, # NO. 2
  #SORTING SETTINGS
  'Sort dist_to_boundary': '', # NO. 3
  'Sort travel_dist': '', # NO. 4
  ###############################################

  'Sort dist_to_53bp1': '',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline3_stiffness import *
pipeline_batch(settings, control)

# from cellquantifier.publish import *
# import pandas as pd
# df = pd.read_csv('/home/linhua/Desktop/temp/200303_50NcBLM-physDataMerged.csv',
#                 index_col=False)
# df= df[ ~df['raw_data'].isin(['200211_50NcLiving_D1-HT-physData.csv',
#                         '200211_50NcLiving_A2-HT-physData.csv',
#                         '200206_50NcLiving_L2-HT-physData.csv',
#                         '190925_50NcLiving_K1-HT-physData.csv']) ]
#
# # df= df[ df['raw_data'].isin(['190924_50NcLiving_B1-HT-physData.csv',
# #                         '190924_50NcLiving_D1-HT-physData.csv',
# #                         '190924_50NcLiving_E1-HT-physData.csv',
# #                         '190924_50NcLiving_I1-HT-physData.csv',
# #                         '190924_50NcLiving_J2-HT-physData.csv',
# #                         '191010_50NcLiving_B1-HT-physData.csv',
# #
# #                         '191004_50NcBLM_A1-HT-physData.csv',
# #                         '191004_50NcBLM_B1-HT-physData.csv',
# #                         '191004_50NcBLM_C1-HT-physData.csv',
# #                         '191004_50NcBLM_E1-HT-physData.csv',
# #                         '191004_50NcBLM_F1-HT-physData.csv',
# #                         '191004_50NcBLM_I1-HT-physData.csv',
# #                         '191004_50NcBLM_K1-HT-physData.csv',
# #                         '191004_50NcBLM_Q1-HT-physData.csv',
# #                         '191004_50NcBLM_T1-HT-physData.csv'
# #                         ]) ]
#
# df['date'] = df['raw_data'].astype(str).str[0:6]
# df = df[ df['date'].isin(['191004', '190924', '190925', '191010', '200206',]) ]
# print(len(df))
# fig_quick_nucleosome(df)

# import pandas as pd
# from cellquantifier.publish._fig_quick_merge import *
# df = pd.read_csv('/home/linhua/Desktop/josh/200730_50NcFixed-physDataMerged.csv',
#                 index_col=False)
# fig_quick_merge(df)


# from cellquantifier.publish import *
# import pandas as pd
# df = pd.read_csv('/home/linhua/Desktop/output/cilia-physData.csv',
#                 index_col=False)
# fig_quick_msd(df)

# import pandas as pd
# from cellquantifier.publish._fig_quick_cilia_5 import *
# from cellquantifier.phys.travel_dist import *
# df = pd.read_csv('/home/linhua/Desktop/phys/200619-physDataMerged.csv',
#                 index_col=False)
# df = add_travel_dist(df)
# fig_quick_cilia_5(df)


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
