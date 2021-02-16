import pandas as pd
import trackpy as tp
import numpy as np
from cellquantifier.phys import *

# Read five merged physData csv file
df_200Nc = pd.read_csv('/home/linhua/Desktop/phys/191231_200Nc-BLMvsLiv-physDataMerged.csv')
df_200Nc = add_traj_length(df_200Nc)
df_all = pd.read_csv('/home/linhua/Desktop/phys/200730_All-physDataMerged.csv')
df_bm = pd.read_csv('/home/linhua/Desktop/phys/200708_50NcLiving-MvsBphysDataMerged.csv')
df_bmt = pd.read_csv('/home/linhua/Desktop/phys/200810_50NcLivingBMT-physDataMerged.csv')
df_uv1 = pd.read_csv('/home/linhua/Desktop/phys/200813_NcUV-physDataMerged.csv')
df_uv2 = pd.read_csv('/home/linhua/Desktop/phys/200925_NcUV-physDataMerged.csv')

# Filter out short trajectories
df_200Nc = df_200Nc[ df_200Nc['traj_length']>=80 ]
df_all = df_all[ ~df_all['exp_label'].isin(['50NcLivingB', '50NcLivingM']) ]
df_all = df_all[ df_all['traj_length']>=80 ]
df_bm = df_bm[ df_bm['traj_length']>=40 ]
df_bmt = df_bmt[ df_bmt['traj_length']>=40 ]
df_uv1 = df_uv1[ df_uv1['traj_length']>=80 ]
df_uv2 = df_uv2[ df_uv2['traj_length']>=80 ]

# # Add physic parameters
# df_200Nc = add_D_alpha(df_200Nc, pixel_size=0.163, frame_rate=5, divide_num=5)
# df_all = add_D_alpha(df_all, pixel_size=0.163, frame_rate=20, divide_num=5)
# df_bm = add_D_alpha(df_bm, pixel_size=0.163, frame_rate=20, divide_num=5)
# df_bmt = add_D_alpha(df_bmt, pixel_size=0.163, frame_rate=20, divide_num=5)
# df_uv1 = add_D_alpha(df_uv1, pixel_size=0.163, frame_rate=20, divide_num=5)
# df_uv2 = add_D_alpha(df_uv2, pixel_size=0.163, frame_rate=20, divide_num=5)

# Filter alpha<0 trajectories
df_200Nc = df_200Nc[ df_200Nc['alpha']>=0 ]
df_all = df_all[ df_all['alpha']>=0 ]
df_bm = df_bm[ df_bm['alpha']>=0 ]
df_bmt = df_bmt[ df_bmt['alpha']>=0 ]
df_uv1 = df_uv1[ df_uv1['alpha']>=0 ]
df_uv2 = df_uv2[ df_uv2['alpha']>=0 ]

# """
# ~~~~~~~~
# """
df_200NcBLM = df_200Nc.loc[ df_200Nc['exp_label']=='200NcBLM', ['x', 'y', 'frame', 'particle']]
df_200NcLiving = df_200Nc.loc[ df_200Nc['exp_label']=='200NcLiving', ['x', 'y', 'frame', 'particle']]

df_50NcBLM_53bp1 = df_all.loc[ (df_all['exp_label']=='50NcBLM') &
            (df_all['sort_flag_53bp1']==True) ,
            ['x', 'y', 'frame', 'particle']]
df_50NcBLM_non53bp1 = df_all.loc[ (df_all['exp_label']=='50NcBLM') &
            (df_all['sort_flag_53bp1']==False) ,
            ['x', 'y', 'frame', 'particle']]
df_50NcBLM_boundary = df_all.loc[ (df_all['exp_label']=='50NcBLM') &
            (df_all['sort_flag_boundary']==True) ,
            ['x', 'y', 'frame', 'particle']]
df_50NcBLM_nonboundary = df_all.loc[ (df_all['exp_label']=='50NcBLM') &
            (df_all['sort_flag_boundary']==False) ,
            ['x', 'y', 'frame', 'particle']]
df_50NcLiving_53bp1 = df_all.loc[ (df_all['exp_label']=='50NcLiving') &
            (df_all['sort_flag_53bp1']==True) ,
            ['x', 'y', 'frame', 'particle']]
df_50NcLiving_non53bp1 = df_all.loc[ (df_all['exp_label']=='50NcLiving') &
            (df_all['sort_flag_53bp1']==False) ,
            ['x', 'y', 'frame', 'particle']]
df_50NcLiving_boundary = df_all.loc[ (df_all['exp_label']=='50NcLiving') &
            (df_all['sort_flag_boundary']==True) ,
            ['x', 'y', 'frame', 'particle']]
df_50NcLiving_nonboundary = df_all.loc[ (df_all['exp_label']=='50NcLiving') &
            (df_all['sort_flag_boundary']==False) ,
            ['x', 'y', 'frame', 'particle']]

df_50NcBLM = df_all.loc[ df_all['exp_label']=='50NcBLM', ['x', 'y', 'frame', 'particle']]
df_50NcLiving = df_all.loc[ df_all['exp_label']=='50NcLiving', ['x', 'y', 'frame', 'particle']]
df_50NcMOCK = df_all.loc[ df_all['exp_label']=='50NcMOCK', ['x', 'y', 'frame', 'particle']]
df_50NcATP = df_all.loc[ df_all['exp_label']=='50NcATP', ['x', 'y', 'frame', 'particle']]
df_50NcFixed = df_all.loc[ df_all['exp_label']=='50NcFixed', ['x', 'y', 'frame', 'particle']]
df_50NcLivingB200708 = df_bm.loc[ df_bm['exp_label']=='B', ['x', 'y', 'frame', 'particle']]
df_50NcLivingM200708 = df_bm.loc[ df_bm['exp_label']=='M', ['x', 'y', 'frame', 'particle']]
df_50NcLivingB200810 = df_bmt.loc[ df_bmt['exp_label']=='50NcLivingB', ['x', 'y', 'frame', 'particle']]
df_50NcLivingM200810 = df_bmt.loc[ df_bmt['exp_label']=='50NcLivingM', ['x', 'y', 'frame', 'particle']]
df_50NcLivingT200810 = df_bmt.loc[ df_bmt['exp_label']=='50NcLivingT', ['x', 'y', 'frame', 'particle']]
df_NcUV1s200813 = df_uv1.loc[ df_uv1['exp_label']=='NcUV1s', ['x', 'y', 'frame', 'particle']]
df_NcUV10s200813 = df_uv1.loc[ df_uv1['exp_label']=='NcUV10s', ['x', 'y', 'frame', 'particle']]
df_NcUV20s200813 = df_uv1.loc[ df_uv1['exp_label']=='NcUV20s', ['x', 'y', 'frame', 'particle']]
df_NcUV30s200813 = df_uv1.loc[ df_uv1['exp_label']=='NcUV30s', ['x', 'y', 'frame', 'particle']]
df_NcUV40s200813 = df_uv1.loc[ df_uv1['exp_label']=='NcUV40s', ['x', 'y', 'frame', 'particle']]
df_NcUVCTL200925 = df_uv2.loc[ df_uv2['exp_label']=='NcUVCTL', ['x', 'y', 'frame', 'particle']]
df_NcUVhalfs200925 = df_uv2.loc[ df_uv2['exp_label']=='NcUV0.5', ['x', 'y', 'frame', 'particle']]
df_NcUV1s200925 = df_uv2.loc[ df_uv2['exp_label']=='NcUV1s', ['x', 'y', 'frame', 'particle']]
df_NcUV2s200925 = df_uv2.loc[ df_uv2['exp_label']=='NcUV2s', ['x', 'y', 'frame', 'particle']]
df_NcUV3s200925 = df_uv2.loc[ df_uv2['exp_label']=='NcUV3s', ['x', 'y', 'frame', 'particle']]
df_NcUV5s200925 = df_uv2.loc[ df_uv2['exp_label']=='NcUV5s', ['x', 'y', 'frame', 'particle']]
df_NcUV7s200925 = df_uv2.loc[ df_uv2['exp_label']=='NcUV7s', ['x', 'y', 'frame', 'particle']]




data_dfs = []

dfs = [df_200NcBLM, df_200NcLiving]
names = ['200NcBLM', '200NcLiving']
for (df, name) in zip(dfs, names):
    im = tp.imsd(df,
                mpp=0.163,
                fps=5,
                max_lagtime=np.inf,
                )

    n = len(im.index) #for use in stand err calculation
    m = int(round(len(im.index)/5))
    im = im.head(m)
    im = im*1e6

    imsd_mean = im.mean(axis=1)
    imsd_std = im.std(axis=1, ddof=0)
    x = imsd_mean.index.to_numpy()
    y = imsd_mean.to_numpy()
    n_data_pts = np.sqrt(np.linspace(n-1, n-m, m))
    yerr = np.divide(imsd_std.to_numpy(), n_data_pts)

    tmp_df = pd.DataFrame({
            name + '_t': x,
            name + '_MSD': y,
            name + '_err': yerr,
            })

    data_dfs.append(tmp_df)




dfs = [
    df_50NcBLM_53bp1, df_50NcBLM_non53bp1, df_50NcBLM_nonboundary, df_50NcBLM_boundary,
    df_50NcLiving_53bp1, df_50NcLiving_non53bp1, df_50NcLiving_nonboundary, df_50NcLiving_boundary,
    df_50NcBLM, df_50NcLiving, df_50NcMOCK, df_50NcATP, df_50NcFixed,
    df_50NcLivingB200708, df_50NcLivingM200708, df_50NcLivingB200810, df_50NcLivingM200810, df_50NcLivingT200810,
    df_NcUV1s200813, df_NcUV10s200813, df_NcUV20s200813, df_NcUV30s200813, df_NcUV40s200813,
    df_NcUVCTL200925, df_NcUVhalfs200925, df_NcUV1s200925, df_NcUV2s200925, df_NcUV3s200925, df_NcUV5s200925, df_NcUV7s200925,
    ]
names = [
    '50NcBLM_53bp1', '50NcBLM_non53bp1', '50NcBLM_nonboundary', '50NcBLM_boundary',
    '50NcLiving_53bp1', '50NcLiving_non53bp1', '50NcLiving_nonboundary', '50NcLiving_boundary',
    '50NcBLM', '50NcLiving', '50NcMOCK', '50NcATP', '50NcFixed',
    '50NcLivingB200708', '50NcLivingM200708', '50NcLivingB200810', '50NcLivingM200810', '50NcLivingT200810',
    'NcUV1s200813', 'NcUV10s200813', 'NcUV20s200813', 'NcUV30s200813', 'NcUV40s200813',
    'NcUVCTL200925', 'NcUVhalfs200925', 'NcUV1s200925', 'NcUV2s200925', 'NcUV3s200925', 'NcUV5s200925', 'NcUV7s200925',
    ]
for (df, name) in zip(dfs, names):
    im = tp.imsd(df,
                mpp=0.163,
                fps=20,
                max_lagtime=np.inf,
                )

    n = len(im.index) #for use in stand err calculation
    m = int(round(len(im.index)/5))
    im = im.head(m)
    im = im*1e6

    imsd_mean = im.mean(axis=1)
    imsd_std = im.std(axis=1, ddof=0)
    x = imsd_mean.index.to_numpy()
    y = imsd_mean.to_numpy()
    n_data_pts = np.sqrt(np.linspace(n-1, n-m, m))
    yerr = np.divide(imsd_std.to_numpy(), n_data_pts)

    tmp_df = pd.DataFrame({
            name + '_t': x,
            name + '_MSD': y,
            name + '_err': yerr,
            })

    data_dfs.append(tmp_df)




msd_df = pd.concat(data_dfs, axis=1)
print(msd_df)

msd_df.round(3).to_csv('/home/linhua/Desktop/MSD.csv', index=False)
