"""Part I: CellQuantifier Sequence Control"""

control = [
# 'clean_dir',
# 'load',
# 'regi',
# 'mask_boundary',
# 'mask_53bp1',
# 'mask_53bp1_blob',
# 'deno_mean',
# 'deno_box',
# 'deno_gaus',
# 'check_detect_fit',
# 'detect',
# 'fit',

# 'filt_track',
# 'phys_xy_global',
# 'phys_dist2halfcilia',
# 'phys_cilia_halfsign',
# 'phys_cilia_otherinfo',
# 'plot_traj',
# 'anim_traj',
# 'merge_plot',

# 'phys_dist253bp1',
# 'phys_dist253bp1_blob',
# 'phys_antigen_data',

# 'filt_track',
# 'phys_dist2boundary',
# # 'plot_traj',
# 'sort_plot',
# 'anim_traj',

# 'merge_plot',
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #IO
  'IO input_path': '/home/linhua/Desktop/input/',
  'IO output_path': '/home/linhua/Desktop/temp/',

  #HEADER INFO
  'Processed By:': 'Hua Lin',
  'Start frame index': 0,
  'End frame index': 28,
  'Load existing analMeta': False,

  #REGISTRATION SETTINGS
  'Regi reference file label': '',
  'Regi ref_ind_num': '',
  'Regi sig_mask': '',
  'Regi thres_rel': '',
  'Regi poly_deg': '',
  'Regi rotation_multiplier': '',
  'Regi translation_multiplier': '',
  'Regi use_ransac': '',

  #SEGMENTATION SETTINGS
  'Segm min_size': '',
  'Segm threshold': '',

  #MASK_BOUNDARY SETTINGS
  'Mask boundary_mask file label': 'bdr',
  'Mask boundary_mask sig': 0,
  'Mask boundary_mask thres_rel': 0.05,
  #MASK_53BP1 SETTINGS
  'Mask 53bp1_mask file label': '-ncs',
  'Mask 53bp1_mask sig': 0,
  'Mask 53bp1_mask thres_rel': 0.01,
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

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,
  'Deno mean_radius': 5,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.005,
  'Det blob_min_sigma': 10,
  'Det blob_max_sigma': 30,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': 0.8,

  #TRACKING SETTINGS
  'Trak frame_rate': 0.48,
  'Trak pixel_size': 0.03,
  'Trak divide_num': 1,
  'Trak search_range': 50,
  'Trak memory': 3,

  #FILTERING SETTINGS
  'Filt max_dist_err': 5,
  'Filt max_sig_to_sigraw': 2,
  'Filt max_delta_area': 2,
  'Filt traj_length_thres': 25,

  #SORTING SETTINGS
  'Sort dist_to_boundary': [-60, 5],
  'Sort dist_to_53bp1': [-50, 10],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline3_cilia import *
pipeline_batch(settings, control)

# from cellquantifier.publish import *
# plot_fig_1()



# from cellquantifier.publish import *
# import pandas as pd
# df = pd.read_csv('/home/linhua/Desktop/temp/200303_50NcBLM-physDataMerged.csv',
#                 index_col=False)
#
# print(len(df))
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


# from cellquantifier.publish import *
# import pandas as pd
# df = pd.read_csv('/home/linhua/Desktop/temp/200407-physDataMerged.csv',
#                 index_col=False)
# fig_quick_cilia(df)



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
