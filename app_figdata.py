import pandas as pd
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

# Add physic parameters
df_200Nc = add_D_alpha(df_200Nc, pixel_size=0.163, frame_rate=5, divide_num=5)
df_200Nc = add_xy_spring_constant(df_200Nc, pixel_size=0.163)
df_200Nc = add_stepsize(df_200Nc, pixel_size=0.163)
df_200Nc = add_constrain_length(df_200Nc, pixel_size=0.163)

df_all = add_D_alpha(df_all, pixel_size=0.163, frame_rate=20, divide_num=5)
df_all = add_xy_spring_constant(df_all, pixel_size=0.163)
df_all = add_stepsize(df_all, pixel_size=0.163)
df_all = add_constrain_length(df_all, pixel_size=0.163)

df_bm = add_D_alpha(df_bm, pixel_size=0.163, frame_rate=20, divide_num=5)
df_bm = add_xy_spring_constant(df_bm, pixel_size=0.163)
df_bm = add_stepsize(df_bm, pixel_size=0.163)
df_bm = add_constrain_length(df_bm, pixel_size=0.163)

df_bmt = add_D_alpha(df_bmt, pixel_size=0.163, frame_rate=20, divide_num=5)
df_bmt = add_xy_spring_constant(df_bmt, pixel_size=0.163)
df_bmt = add_stepsize(df_bmt, pixel_size=0.163)
df_bmt = add_constrain_length(df_bmt, pixel_size=0.163)

df_uv1 = add_D_alpha(df_uv1, pixel_size=0.163, frame_rate=20, divide_num=5)
df_uv1 = add_xy_spring_constant(df_uv1, pixel_size=0.163)
df_uv1 = add_stepsize(df_uv1, pixel_size=0.163)
df_uv1 = add_constrain_length(df_uv1, pixel_size=0.163)

df_uv2 = add_D_alpha(df_uv2, pixel_size=0.163, frame_rate=20, divide_num=5)
df_uv2 = add_xy_spring_constant(df_uv2, pixel_size=0.163)
df_uv2 = add_stepsize(df_uv2, pixel_size=0.163)
df_uv2 = add_constrain_length(df_uv2, pixel_size=0.163)

# Filter alpha<0 trajectories
df_200Nc = df_200Nc[ df_200Nc['alpha']>=0 ]
df_all = df_all[ df_all['alpha']>=0 ]
df_bm = df_bm[ df_bm['alpha']>=0 ]
df_bmt = df_bmt[ df_bmt['alpha']>=0 ]
df_uv1 = df_uv1[ df_uv1['alpha']>=0 ]
df_uv2 = df_uv2[ df_uv2['alpha']>=0 ]

# Drop duplicate value of same particle
dfp_200Nc = df_200Nc.drop_duplicates('particle')
dfp_all = df_all.drop_duplicates('particle')
dfp_bm = df_bm.drop_duplicates('particle')
dfp_bmt = df_bmt.drop_duplicates('particle')
dfp_uv1 = df_uv1.drop_duplicates('particle')
dfp_uv2 = df_uv2.drop_duplicates('particle')

# """
# ~~~~~~~~
# """
D_200NcBLM = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcBLM', 'D'].to_numpy(), columns=['200NcBLM'])
D_200NcLiving = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcLiving', 'D'].to_numpy(), columns=['200NcLiving'])

D_50NcBLM_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'D'].to_numpy(), columns=['50NcBLM_53bp1'])
D_50NcBLM_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'D'].to_numpy(), columns=['50NcBLM_non53bp1'])
D_50NcBLM_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==True) ,
            'D'].to_numpy(), columns=['50NcBLM_boundary'])
D_50NcBLM_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==False) ,
            'D'].to_numpy(), columns=['50NcBLM_nonboundary'])
D_50NcLiving_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'D'].to_numpy(), columns=['50NcLiving_53bp1'])
D_50NcLiving_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'D'].to_numpy(), columns=['50NcLiving_non53bp1'])
D_50NcLiving_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==True) ,
            'D'].to_numpy(), columns=['50NcLiving_boundary'])
D_50NcLiving_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==False) ,
            'D'].to_numpy(), columns=['50NcLiving_nonboundary'])

D_50NcBLM = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcBLM', 'D'].to_numpy(), columns=['50NcBLM'])
D_50NcLiving = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcLiving', 'D'].to_numpy(), columns=['50NcLiving'])
D_50NcMOCK = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcMOCK', 'D'].to_numpy(), columns=['50NcMOCK'])
D_50NcATP = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcATP', 'D'].to_numpy(), columns=['50NcATP'])
D_50NcFixed = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcFixed', 'D'].to_numpy(), columns=['50NcFixed'])
D_50NcLivingB200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='B', 'D'].to_numpy(), columns=['50NcLivingB200708'])
D_50NcLivingM200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='M', 'D'].to_numpy(), columns=['50NcLivingM200708'])
D_50NcLivingB200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingB', 'D'].to_numpy(), columns=['50NcLivingB200810'])
D_50NcLivingM200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingM', 'D'].to_numpy(), columns=['50NcLivingM200810'])
D_50NcLivingT200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingT', 'D'].to_numpy(), columns=['50NcLivingT200810'])
D_NcUV1s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV1s', 'D'].to_numpy(), columns=['NcUV1s200813'])
D_NcUV10s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV10s', 'D'].to_numpy(), columns=['NcUV10s200813'])
D_NcUV20s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV20s', 'D'].to_numpy(), columns=['NcUV20s200813'])
D_NcUV30s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV30s', 'D'].to_numpy(), columns=['NcUV30s200813'])
D_NcUV40s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV40s', 'D'].to_numpy(), columns=['NcUV40s200813'])
D_NcUVCTL200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUVCTL', 'D'].to_numpy(), columns=['NcUVCTL200925'])
D_NcUVhalfs200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV0.5', 'D'].to_numpy(), columns=['NcUV0.5s200925'])
D_NcUV1s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV1s', 'D'].to_numpy(), columns=['NcUV1s200925'])
D_NcUV2s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV2s', 'D'].to_numpy(), columns=['NcUV2s200925'])
D_NcUV3s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV3s', 'D'].to_numpy(), columns=['NcUV3s200925'])
D_NcUV5s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV5s', 'D'].to_numpy(), columns=['NcUV5s200925'])
D_NcUV7s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV7s', 'D'].to_numpy(), columns=['NcUV7s200925'])

df_D = pd.concat([
    D_200NcBLM, D_200NcLiving,
    D_50NcBLM_53bp1, D_50NcBLM_non53bp1, D_50NcBLM_nonboundary, D_50NcBLM_boundary,
    D_50NcLiving_53bp1, D_50NcLiving_non53bp1, D_50NcLiving_nonboundary, D_50NcLiving_boundary,
    D_50NcBLM, D_50NcLiving, D_50NcMOCK, D_50NcATP, D_50NcFixed,
    D_50NcLivingB200708, D_50NcLivingM200708, D_50NcLivingB200810, D_50NcLivingM200810, D_50NcLivingT200810,
    D_NcUV1s200813, D_NcUV10s200813, D_NcUV20s200813, D_NcUV30s200813, D_NcUV40s200813,
    D_NcUVCTL200925, D_NcUVhalfs200925, D_NcUV1s200925, D_NcUV2s200925, D_NcUV3s200925, D_NcUV5s200925, D_NcUV7s200925,
    ], axis=1)

df_D.round(3).to_csv('/home/linhua/Desktop/D.csv', index=False)



# """
# ~~~~~~~~
# """
alpha_200NcBLM = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcBLM', 'alpha'].to_numpy(), columns=['200NcBLM'])
alpha_200NcLiving = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcLiving', 'alpha'].to_numpy(), columns=['200NcLiving'])

alpha_50NcBLM_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'alpha'].to_numpy(), columns=['50NcBLM_53bp1'])
alpha_50NcBLM_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'alpha'].to_numpy(), columns=['50NcBLM_non53bp1'])
alpha_50NcBLM_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==True) ,
            'alpha'].to_numpy(), columns=['50NcBLM_boundary'])
alpha_50NcBLM_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==False) ,
            'alpha'].to_numpy(), columns=['50NcBLM_nonboundary'])
alpha_50NcLiving_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'alpha'].to_numpy(), columns=['50NcLiving_53bp1'])
alpha_50NcLiving_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'alpha'].to_numpy(), columns=['50NcLiving_non53bp1'])
alpha_50NcLiving_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==True) ,
            'alpha'].to_numpy(), columns=['50NcLiving_boundary'])
alpha_50NcLiving_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==False) ,
            'alpha'].to_numpy(), columns=['50NcLiving_nonboundary'])

alpha_50NcBLM = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcBLM', 'alpha'].to_numpy(), columns=['50NcBLM'])
alpha_50NcLiving = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcLiving', 'alpha'].to_numpy(), columns=['50NcLiving'])
alpha_50NcMOCK = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcMOCK', 'alpha'].to_numpy(), columns=['50NcMOCK'])
alpha_50NcATP = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcATP', 'alpha'].to_numpy(), columns=['50NcATP'])
alpha_50NcFixed = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcFixed', 'alpha'].to_numpy(), columns=['50NcFixed'])
alpha_50NcLivingB200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='B', 'alpha'].to_numpy(), columns=['50NcLivingB200708'])
alpha_50NcLivingM200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='M', 'alpha'].to_numpy(), columns=['50NcLivingM200708'])
alpha_50NcLivingB200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingB', 'alpha'].to_numpy(), columns=['50NcLivingB200810'])
alpha_50NcLivingM200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingM', 'alpha'].to_numpy(), columns=['50NcLivingM200810'])
alpha_50NcLivingT200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingT', 'alpha'].to_numpy(), columns=['50NcLivingT200810'])
alpha_NcUV1s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV1s', 'alpha'].to_numpy(), columns=['NcUV1s200813'])
alpha_NcUV10s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV10s', 'alpha'].to_numpy(), columns=['NcUV10s200813'])
alpha_NcUV20s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV20s', 'alpha'].to_numpy(), columns=['NcUV20s200813'])
alpha_NcUV30s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV30s', 'alpha'].to_numpy(), columns=['NcUV30s200813'])
alpha_NcUV40s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV40s', 'alpha'].to_numpy(), columns=['NcUV40s200813'])
alpha_NcUVCTL200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUVCTL', 'alpha'].to_numpy(), columns=['NcUVCTL200925'])
alpha_NcUVhalfs200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV0.5', 'alpha'].to_numpy(), columns=['NcUV0.5s200925'])
alpha_NcUV1s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV1s', 'alpha'].to_numpy(), columns=['NcUV1s200925'])
alpha_NcUV2s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV2s', 'alpha'].to_numpy(), columns=['NcUV2s200925'])
alpha_NcUV3s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV3s', 'alpha'].to_numpy(), columns=['NcUV3s200925'])
alpha_NcUV5s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV5s', 'alpha'].to_numpy(), columns=['NcUV5s200925'])
alpha_NcUV7s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV7s', 'alpha'].to_numpy(), columns=['NcUV7s200925'])

df_alpha = pd.concat([
    alpha_200NcBLM, alpha_200NcLiving,
    alpha_50NcBLM_53bp1, alpha_50NcBLM_non53bp1, alpha_50NcBLM_nonboundary, alpha_50NcBLM_boundary,
    alpha_50NcLiving_53bp1, alpha_50NcLiving_non53bp1, alpha_50NcLiving_nonboundary, alpha_50NcLiving_boundary,
    alpha_50NcBLM, alpha_50NcLiving, alpha_50NcMOCK, alpha_50NcATP, alpha_50NcFixed,
    alpha_50NcLivingB200708, alpha_50NcLivingM200708, alpha_50NcLivingB200810, alpha_50NcLivingM200810, alpha_50NcLivingT200810,
    alpha_NcUV1s200813, alpha_NcUV10s200813, alpha_NcUV20s200813, alpha_NcUV30s200813, alpha_NcUV40s200813,
    alpha_NcUVCTL200925, alpha_NcUVhalfs200925, alpha_NcUV1s200925, alpha_NcUV2s200925, alpha_NcUV3s200925, alpha_NcUV5s200925, alpha_NcUV7s200925,
    ], axis=1)

df_alpha.round(3).to_csv('/home/linhua/Desktop/alpha.csv', index=False)


# """
# ~~~~~~~~
# """
x_spring_constant_200NcBLM = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcBLM', 'x_spring_constant'].to_numpy(), columns=['200NcBLM'])
x_spring_constant_200NcLiving = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcLiving', 'x_spring_constant'].to_numpy(), columns=['200NcLiving'])

x_spring_constant_50NcBLM_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'x_spring_constant'].to_numpy(), columns=['50NcBLM_53bp1'])
x_spring_constant_50NcBLM_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'x_spring_constant'].to_numpy(), columns=['50NcBLM_non53bp1'])
x_spring_constant_50NcBLM_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==True) ,
            'x_spring_constant'].to_numpy(), columns=['50NcBLM_boundary'])
x_spring_constant_50NcBLM_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==False) ,
            'x_spring_constant'].to_numpy(), columns=['50NcBLM_nonboundary'])
x_spring_constant_50NcLiving_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'x_spring_constant'].to_numpy(), columns=['50NcLiving_53bp1'])
x_spring_constant_50NcLiving_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'x_spring_constant'].to_numpy(), columns=['50NcLiving_non53bp1'])
x_spring_constant_50NcLiving_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==True) ,
            'x_spring_constant'].to_numpy(), columns=['50NcLiving_boundary'])
x_spring_constant_50NcLiving_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==False) ,
            'x_spring_constant'].to_numpy(), columns=['50NcLiving_nonboundary'])

x_spring_constant_50NcBLM = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcBLM', 'x_spring_constant'].to_numpy(), columns=['50NcBLM'])
x_spring_constant_50NcLiving = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcLiving', 'x_spring_constant'].to_numpy(), columns=['50NcLiving'])
x_spring_constant_50NcMOCK = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcMOCK', 'x_spring_constant'].to_numpy(), columns=['50NcMOCK'])
x_spring_constant_50NcATP = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcATP', 'x_spring_constant'].to_numpy(), columns=['50NcATP'])
x_spring_constant_50NcFixed = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcFixed', 'x_spring_constant'].to_numpy(), columns=['50NcFixed'])
x_spring_constant_50NcLivingB200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='B', 'x_spring_constant'].to_numpy(), columns=['50NcLivingB200708'])
x_spring_constant_50NcLivingM200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='M', 'x_spring_constant'].to_numpy(), columns=['50NcLivingM200708'])
x_spring_constant_50NcLivingB200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingB', 'x_spring_constant'].to_numpy(), columns=['50NcLivingB200810'])
x_spring_constant_50NcLivingM200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingM', 'x_spring_constant'].to_numpy(), columns=['50NcLivingM200810'])
x_spring_constant_50NcLivingT200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingT', 'x_spring_constant'].to_numpy(), columns=['50NcLivingT200810'])
x_spring_constant_NcUV1s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV1s', 'x_spring_constant'].to_numpy(), columns=['NcUV1s200813'])
x_spring_constant_NcUV10s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV10s', 'x_spring_constant'].to_numpy(), columns=['NcUV10s200813'])
x_spring_constant_NcUV20s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV20s', 'x_spring_constant'].to_numpy(), columns=['NcUV20s200813'])
x_spring_constant_NcUV30s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV30s', 'x_spring_constant'].to_numpy(), columns=['NcUV30s200813'])
x_spring_constant_NcUV40s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV40s', 'x_spring_constant'].to_numpy(), columns=['NcUV40s200813'])
x_spring_constant_NcUVCTL200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUVCTL', 'x_spring_constant'].to_numpy(), columns=['NcUVCTL200925'])
x_spring_constant_NcUVhalfs200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV0.5', 'x_spring_constant'].to_numpy(), columns=['NcUV0.5s200925'])
x_spring_constant_NcUV1s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV1s', 'x_spring_constant'].to_numpy(), columns=['NcUV1s200925'])
x_spring_constant_NcUV2s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV2s', 'x_spring_constant'].to_numpy(), columns=['NcUV2s200925'])
x_spring_constant_NcUV3s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV3s', 'x_spring_constant'].to_numpy(), columns=['NcUV3s200925'])
x_spring_constant_NcUV5s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV5s', 'x_spring_constant'].to_numpy(), columns=['NcUV5s200925'])
x_spring_constant_NcUV7s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV7s', 'x_spring_constant'].to_numpy(), columns=['NcUV7s200925'])

df_x_spring_constant = pd.concat([
    x_spring_constant_200NcBLM, x_spring_constant_200NcLiving,
    x_spring_constant_50NcBLM_53bp1, x_spring_constant_50NcBLM_non53bp1, x_spring_constant_50NcBLM_nonboundary, x_spring_constant_50NcBLM_boundary,
    x_spring_constant_50NcLiving_53bp1, x_spring_constant_50NcLiving_non53bp1, x_spring_constant_50NcLiving_nonboundary, x_spring_constant_50NcLiving_boundary,
    x_spring_constant_50NcBLM, x_spring_constant_50NcLiving, x_spring_constant_50NcMOCK, x_spring_constant_50NcATP, x_spring_constant_50NcFixed,
    x_spring_constant_50NcLivingB200708, x_spring_constant_50NcLivingM200708, x_spring_constant_50NcLivingB200810, x_spring_constant_50NcLivingM200810, x_spring_constant_50NcLivingT200810,
    x_spring_constant_NcUV1s200813, x_spring_constant_NcUV10s200813, x_spring_constant_NcUV20s200813, x_spring_constant_NcUV30s200813, x_spring_constant_NcUV40s200813,
    x_spring_constant_NcUVCTL200925, x_spring_constant_NcUVhalfs200925, x_spring_constant_NcUV1s200925, x_spring_constant_NcUV2s200925, x_spring_constant_NcUV3s200925, x_spring_constant_NcUV5s200925, x_spring_constant_NcUV7s200925,
    ], axis=1)

df_x_spring_constant.round(3).to_csv('/home/linhua/Desktop/x_spring_constant.csv', index=False)


# """
# ~~~~~~~~
# """
y_spring_constant_200NcBLM = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcBLM', 'y_spring_constant'].to_numpy(), columns=['200NcBLM'])
y_spring_constant_200NcLiving = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcLiving', 'y_spring_constant'].to_numpy(), columns=['200NcLiving'])

y_spring_constant_50NcBLM_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'y_spring_constant'].to_numpy(), columns=['50NcBLM_53bp1'])
y_spring_constant_50NcBLM_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'y_spring_constant'].to_numpy(), columns=['50NcBLM_non53bp1'])
y_spring_constant_50NcBLM_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==True) ,
            'y_spring_constant'].to_numpy(), columns=['50NcBLM_boundary'])
y_spring_constant_50NcBLM_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==False) ,
            'y_spring_constant'].to_numpy(), columns=['50NcBLM_nonboundary'])
y_spring_constant_50NcLiving_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'y_spring_constant'].to_numpy(), columns=['50NcLiving_53bp1'])
y_spring_constant_50NcLiving_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'y_spring_constant'].to_numpy(), columns=['50NcLiving_non53bp1'])
y_spring_constant_50NcLiving_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==True) ,
            'y_spring_constant'].to_numpy(), columns=['50NcLiving_boundary'])
y_spring_constant_50NcLiving_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==False) ,
            'y_spring_constant'].to_numpy(), columns=['50NcLiving_nonboundary'])

y_spring_constant_50NcBLM = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcBLM', 'y_spring_constant'].to_numpy(), columns=['50NcBLM'])
y_spring_constant_50NcLiving = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcLiving', 'y_spring_constant'].to_numpy(), columns=['50NcLiving'])
y_spring_constant_50NcMOCK = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcMOCK', 'y_spring_constant'].to_numpy(), columns=['50NcMOCK'])
y_spring_constant_50NcATP = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcATP', 'y_spring_constant'].to_numpy(), columns=['50NcATP'])
y_spring_constant_50NcFixed = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcFixed', 'y_spring_constant'].to_numpy(), columns=['50NcFixed'])
y_spring_constant_50NcLivingB200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='B', 'y_spring_constant'].to_numpy(), columns=['50NcLivingB200708'])
y_spring_constant_50NcLivingM200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='M', 'y_spring_constant'].to_numpy(), columns=['50NcLivingM200708'])
y_spring_constant_50NcLivingB200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingB', 'y_spring_constant'].to_numpy(), columns=['50NcLivingB200810'])
y_spring_constant_50NcLivingM200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingM', 'y_spring_constant'].to_numpy(), columns=['50NcLivingM200810'])
y_spring_constant_50NcLivingT200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingT', 'y_spring_constant'].to_numpy(), columns=['50NcLivingT200810'])
y_spring_constant_NcUV1s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV1s', 'y_spring_constant'].to_numpy(), columns=['NcUV1s200813'])
y_spring_constant_NcUV10s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV10s', 'y_spring_constant'].to_numpy(), columns=['NcUV10s200813'])
y_spring_constant_NcUV20s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV20s', 'y_spring_constant'].to_numpy(), columns=['NcUV20s200813'])
y_spring_constant_NcUV30s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV30s', 'y_spring_constant'].to_numpy(), columns=['NcUV30s200813'])
y_spring_constant_NcUV40s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV40s', 'y_spring_constant'].to_numpy(), columns=['NcUV40s200813'])
y_spring_constant_NcUVCTL200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUVCTL', 'y_spring_constant'].to_numpy(), columns=['NcUVCTL200925'])
y_spring_constant_NcUVhalfs200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV0.5', 'y_spring_constant'].to_numpy(), columns=['NcUV0.5s200925'])
y_spring_constant_NcUV1s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV1s', 'y_spring_constant'].to_numpy(), columns=['NcUV1s200925'])
y_spring_constant_NcUV2s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV2s', 'y_spring_constant'].to_numpy(), columns=['NcUV2s200925'])
y_spring_constant_NcUV3s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV3s', 'y_spring_constant'].to_numpy(), columns=['NcUV3s200925'])
y_spring_constant_NcUV5s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV5s', 'y_spring_constant'].to_numpy(), columns=['NcUV5s200925'])
y_spring_constant_NcUV7s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV7s', 'y_spring_constant'].to_numpy(), columns=['NcUV7s200925'])

df_y_spring_constant = pd.concat([
    y_spring_constant_200NcBLM, y_spring_constant_200NcLiving,
    y_spring_constant_50NcBLM_53bp1, y_spring_constant_50NcBLM_non53bp1, y_spring_constant_50NcBLM_nonboundary, y_spring_constant_50NcBLM_boundary,
    y_spring_constant_50NcLiving_53bp1, y_spring_constant_50NcLiving_non53bp1, y_spring_constant_50NcLiving_nonboundary, y_spring_constant_50NcLiving_boundary,
    y_spring_constant_50NcBLM, y_spring_constant_50NcLiving, y_spring_constant_50NcMOCK, y_spring_constant_50NcATP, y_spring_constant_50NcFixed,
    y_spring_constant_50NcLivingB200708, y_spring_constant_50NcLivingM200708, y_spring_constant_50NcLivingB200810, y_spring_constant_50NcLivingM200810, y_spring_constant_50NcLivingT200810,
    y_spring_constant_NcUV1s200813, y_spring_constant_NcUV10s200813, y_spring_constant_NcUV20s200813, y_spring_constant_NcUV30s200813, y_spring_constant_NcUV40s200813,
    y_spring_constant_NcUVCTL200925, y_spring_constant_NcUVhalfs200925, y_spring_constant_NcUV1s200925, y_spring_constant_NcUV2s200925, y_spring_constant_NcUV3s200925, y_spring_constant_NcUV5s200925, y_spring_constant_NcUV7s200925,
    ], axis=1)

df_y_spring_constant.round(3).to_csv('/home/linhua/Desktop/y_spring_constant.csv', index=False)


# """
# ~~~~~~~~
# """
stepsize_200NcBLM = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcBLM', 'stepsize'].to_numpy(), columns=['200NcBLM'])
stepsize_200NcLiving = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcLiving', 'stepsize'].to_numpy(), columns=['200NcLiving'])

stepsize_50NcBLM_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'stepsize'].to_numpy(), columns=['50NcBLM_53bp1'])
stepsize_50NcBLM_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'stepsize'].to_numpy(), columns=['50NcBLM_non53bp1'])
stepsize_50NcBLM_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==True) ,
            'stepsize'].to_numpy(), columns=['50NcBLM_boundary'])
stepsize_50NcBLM_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==False) ,
            'stepsize'].to_numpy(), columns=['50NcBLM_nonboundary'])
stepsize_50NcLiving_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'stepsize'].to_numpy(), columns=['50NcLiving_53bp1'])
stepsize_50NcLiving_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'stepsize'].to_numpy(), columns=['50NcLiving_non53bp1'])
stepsize_50NcLiving_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==True) ,
            'stepsize'].to_numpy(), columns=['50NcLiving_boundary'])
stepsize_50NcLiving_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==False) ,
            'stepsize'].to_numpy(), columns=['50NcLiving_nonboundary'])

stepsize_50NcBLM = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcBLM', 'stepsize'].to_numpy(), columns=['50NcBLM'])
stepsize_50NcLiving = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcLiving', 'stepsize'].to_numpy(), columns=['50NcLiving'])
stepsize_50NcMOCK = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcMOCK', 'stepsize'].to_numpy(), columns=['50NcMOCK'])
stepsize_50NcATP = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcATP', 'stepsize'].to_numpy(), columns=['50NcATP'])
stepsize_50NcFixed = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcFixed', 'stepsize'].to_numpy(), columns=['50NcFixed'])
stepsize_50NcLivingB200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='B', 'stepsize'].to_numpy(), columns=['50NcLivingB200708'])
stepsize_50NcLivingM200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='M', 'stepsize'].to_numpy(), columns=['50NcLivingM200708'])
stepsize_50NcLivingB200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingB', 'stepsize'].to_numpy(), columns=['50NcLivingB200810'])
stepsize_50NcLivingM200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingM', 'stepsize'].to_numpy(), columns=['50NcLivingM200810'])
stepsize_50NcLivingT200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingT', 'stepsize'].to_numpy(), columns=['50NcLivingT200810'])
stepsize_NcUV1s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV1s', 'stepsize'].to_numpy(), columns=['NcUV1s200813'])
stepsize_NcUV10s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV10s', 'stepsize'].to_numpy(), columns=['NcUV10s200813'])
stepsize_NcUV20s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV20s', 'stepsize'].to_numpy(), columns=['NcUV20s200813'])
stepsize_NcUV30s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV30s', 'stepsize'].to_numpy(), columns=['NcUV30s200813'])
stepsize_NcUV40s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV40s', 'stepsize'].to_numpy(), columns=['NcUV40s200813'])
stepsize_NcUVCTL200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUVCTL', 'stepsize'].to_numpy(), columns=['NcUVCTL200925'])
stepsize_NcUVhalfs200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV0.5', 'stepsize'].to_numpy(), columns=['NcUV0.5s200925'])
stepsize_NcUV1s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV1s', 'stepsize'].to_numpy(), columns=['NcUV1s200925'])
stepsize_NcUV2s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV2s', 'stepsize'].to_numpy(), columns=['NcUV2s200925'])
stepsize_NcUV3s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV3s', 'stepsize'].to_numpy(), columns=['NcUV3s200925'])
stepsize_NcUV5s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV5s', 'stepsize'].to_numpy(), columns=['NcUV5s200925'])
stepsize_NcUV7s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV7s', 'stepsize'].to_numpy(), columns=['NcUV7s200925'])

df_stepsize = pd.concat([
    stepsize_200NcBLM, stepsize_200NcLiving,
    stepsize_50NcBLM_53bp1, stepsize_50NcBLM_non53bp1, stepsize_50NcBLM_nonboundary, stepsize_50NcBLM_boundary,
    stepsize_50NcLiving_53bp1, stepsize_50NcLiving_non53bp1, stepsize_50NcLiving_nonboundary, stepsize_50NcLiving_boundary,
    stepsize_50NcBLM, stepsize_50NcLiving, stepsize_50NcMOCK, stepsize_50NcATP, stepsize_50NcFixed,
    stepsize_50NcLivingB200708, stepsize_50NcLivingM200708, stepsize_50NcLivingB200810, stepsize_50NcLivingM200810, stepsize_50NcLivingT200810,
    stepsize_NcUV1s200813, stepsize_NcUV10s200813, stepsize_NcUV20s200813, stepsize_NcUV30s200813, stepsize_NcUV40s200813,
    stepsize_NcUVCTL200925, stepsize_NcUVhalfs200925, stepsize_NcUV1s200925, stepsize_NcUV2s200925, stepsize_NcUV3s200925, stepsize_NcUV5s200925, stepsize_NcUV7s200925,
    ], axis=1)

df_stepsize.round(3).to_csv('/home/linhua/Desktop/stepsize.csv', index=False)


# """
# ~~~~~~~~
# """
constrain_length_200NcBLM = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcBLM', 'constrain_length'].to_numpy(), columns=['200NcBLM'])
constrain_length_200NcLiving = pd.DataFrame(dfp_200Nc.loc[ dfp_200Nc['exp_label']=='200NcLiving', 'constrain_length'].to_numpy(), columns=['200NcLiving'])

constrain_length_50NcBLM_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'constrain_length'].to_numpy(), columns=['50NcBLM_53bp1'])
constrain_length_50NcBLM_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'constrain_length'].to_numpy(), columns=['50NcBLM_non53bp1'])
constrain_length_50NcBLM_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==True) ,
            'constrain_length'].to_numpy(), columns=['50NcBLM_boundary'])
constrain_length_50NcBLM_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcBLM') &
            (dfp_all['sort_flag_boundary']==False) ,
            'constrain_length'].to_numpy(), columns=['50NcBLM_nonboundary'])
constrain_length_50NcLiving_53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==True) ,
            'constrain_length'].to_numpy(), columns=['50NcLiving_53bp1'])
constrain_length_50NcLiving_non53bp1 = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_53bp1']==False) ,
            'constrain_length'].to_numpy(), columns=['50NcLiving_non53bp1'])
constrain_length_50NcLiving_boundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==True) ,
            'constrain_length'].to_numpy(), columns=['50NcLiving_boundary'])
constrain_length_50NcLiving_nonboundary = pd.DataFrame(dfp_all.loc[ (dfp_all['exp_label']=='50NcLiving') &
            (dfp_all['sort_flag_boundary']==False) ,
            'constrain_length'].to_numpy(), columns=['50NcLiving_nonboundary'])

constrain_length_50NcBLM = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcBLM', 'constrain_length'].to_numpy(), columns=['50NcBLM'])
constrain_length_50NcLiving = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcLiving', 'constrain_length'].to_numpy(), columns=['50NcLiving'])
constrain_length_50NcMOCK = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcMOCK', 'constrain_length'].to_numpy(), columns=['50NcMOCK'])
constrain_length_50NcATP = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcATP', 'constrain_length'].to_numpy(), columns=['50NcATP'])
constrain_length_50NcFixed = pd.DataFrame(dfp_all.loc[ dfp_all['exp_label']=='50NcFixed', 'constrain_length'].to_numpy(), columns=['50NcFixed'])
constrain_length_50NcLivingB200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='B', 'constrain_length'].to_numpy(), columns=['50NcLivingB200708'])
constrain_length_50NcLivingM200708 = pd.DataFrame(dfp_bm.loc[ dfp_bm['exp_label']=='M', 'constrain_length'].to_numpy(), columns=['50NcLivingM200708'])
constrain_length_50NcLivingB200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingB', 'constrain_length'].to_numpy(), columns=['50NcLivingB200810'])
constrain_length_50NcLivingM200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingM', 'constrain_length'].to_numpy(), columns=['50NcLivingM200810'])
constrain_length_50NcLivingT200810 = pd.DataFrame(dfp_bmt.loc[ dfp_bmt['exp_label']=='50NcLivingT', 'constrain_length'].to_numpy(), columns=['50NcLivingT200810'])
constrain_length_NcUV1s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV1s', 'constrain_length'].to_numpy(), columns=['NcUV1s200813'])
constrain_length_NcUV10s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV10s', 'constrain_length'].to_numpy(), columns=['NcUV10s200813'])
constrain_length_NcUV20s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV20s', 'constrain_length'].to_numpy(), columns=['NcUV20s200813'])
constrain_length_NcUV30s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV30s', 'constrain_length'].to_numpy(), columns=['NcUV30s200813'])
constrain_length_NcUV40s200813 = pd.DataFrame(dfp_uv1.loc[ dfp_uv1['exp_label']=='NcUV40s', 'constrain_length'].to_numpy(), columns=['NcUV40s200813'])
constrain_length_NcUVCTL200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUVCTL', 'constrain_length'].to_numpy(), columns=['NcUVCTL200925'])
constrain_length_NcUVhalfs200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV0.5', 'constrain_length'].to_numpy(), columns=['NcUV0.5s200925'])
constrain_length_NcUV1s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV1s', 'constrain_length'].to_numpy(), columns=['NcUV1s200925'])
constrain_length_NcUV2s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV2s', 'constrain_length'].to_numpy(), columns=['NcUV2s200925'])
constrain_length_NcUV3s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV3s', 'constrain_length'].to_numpy(), columns=['NcUV3s200925'])
constrain_length_NcUV5s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV5s', 'constrain_length'].to_numpy(), columns=['NcUV5s200925'])
constrain_length_NcUV7s200925 = pd.DataFrame(dfp_uv2.loc[ dfp_uv2['exp_label']=='NcUV7s', 'constrain_length'].to_numpy(), columns=['NcUV7s200925'])

df_constrain_length = pd.concat([
    constrain_length_200NcBLM, constrain_length_200NcLiving,
    constrain_length_50NcBLM_53bp1, constrain_length_50NcBLM_non53bp1, constrain_length_50NcBLM_nonboundary, constrain_length_50NcBLM_boundary,
    constrain_length_50NcLiving_53bp1, constrain_length_50NcLiving_non53bp1, constrain_length_50NcLiving_nonboundary, constrain_length_50NcLiving_boundary,
    constrain_length_50NcBLM, constrain_length_50NcLiving, constrain_length_50NcMOCK, constrain_length_50NcATP, constrain_length_50NcFixed,
    constrain_length_50NcLivingB200708, constrain_length_50NcLivingM200708, constrain_length_50NcLivingB200810, constrain_length_50NcLivingM200810, constrain_length_50NcLivingT200810,
    constrain_length_NcUV1s200813, constrain_length_NcUV10s200813, constrain_length_NcUV20s200813, constrain_length_NcUV30s200813, constrain_length_NcUV40s200813,
    constrain_length_NcUVCTL200925, constrain_length_NcUVhalfs200925, constrain_length_NcUV1s200925, constrain_length_NcUV2s200925, constrain_length_NcUV3s200925, constrain_length_NcUV5s200925, constrain_length_NcUV7s200925,
    ], axis=1)

df_constrain_length.round(3).to_csv('/home/linhua/Desktop/constrain_length.csv', index=False)
