import pandas as pd
import numpy as np

# Read five merged physData csv file
df_all = pd.read_csv('/home/linhua/Desktop/phys/200730_All-physDataMerged.csv')
df_bm = pd.read_csv('/home/linhua/Desktop/phys/200708_50NcLiving-MvsBphysDataMerged.csv')
df_bmt = pd.read_csv('/home/linhua/Desktop/phys/200810_50NcLivingBMT-physDataMerged.csv')
df_uv1 = pd.read_csv('/home/linhua/Desktop/phys/200813_NcUV-physDataMerged.csv')
df_uv2 = pd.read_csv('/home/linhua/Desktop/phys/200925_NcUV-physDataMerged.csv')

# Filter out short trajectories
df_all = df_all[ ~df_all['exp_label'].isin(['50NcLivingB', '50NcLivingM']) ]
df_all = df_all[ df_all['traj_length']>=80 ]
df_bm = df_bm[ df_bm['traj_length']>=40 ]
df_bmt = df_bmt[ df_bmt['traj_length']>=40 ]
df_uv1 = df_uv1[ df_uv1['traj_length']>=80 ]
df_uv2 = df_uv2[ df_uv2['traj_length']>=80 ]

# Filter alpha<0 trajectories
df_all = df_all[ df_all['alpha']>=0 ]
df_bm = df_bm[ df_bm['alpha']>=0 ]
df_bmt = df_bmt[ df_bmt['alpha']>=0 ]
df_uv1 = df_uv1[ df_uv1['alpha']>=0 ]
df_uv2 = df_uv2[ df_uv2['alpha']>=0 ]

# Drop duplicate value of same particle
dfp_all = df_all.drop_duplicates('particle')
dfp_bm = df_bm.drop_duplicates('particle')
dfp_bmt = df_bmt.drop_duplicates('particle')
dfp_uv1 = df_uv1.drop_duplicates('particle')
dfp_uv2 = df_uv2.drop_duplicates('particle')



















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

df_D = pd.concat([D_50NcBLM, D_50NcLiving, D_50NcMOCK, D_50NcATP, D_50NcFixed,
    D_50NcLivingB200708, D_50NcLivingM200708, D_50NcLivingB200810, D_50NcLivingM200810, D_50NcLivingT200810,
    D_NcUV1s200813, D_NcUV10s200813, D_NcUV20s200813, D_NcUV30s200813, D_NcUV40s200813,
    D_NcUVCTL200925, D_NcUVhalfs200925, D_NcUV1s200925, D_NcUV2s200925, D_NcUV3s200925, D_NcUV5s200925, D_NcUV7s200925,
    ], axis=1)

df_D.round(3).to_csv('/home/linhua/Desktop/D.csv', index=False)





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

df_alpha = pd.concat([alpha_50NcBLM, alpha_50NcLiving, alpha_50NcMOCK, alpha_50NcATP, alpha_50NcFixed,
    alpha_50NcLivingB200708, alpha_50NcLivingM200708, alpha_50NcLivingB200810, alpha_50NcLivingM200810, alpha_50NcLivingT200810,
    alpha_NcUV1s200813, alpha_NcUV10s200813, alpha_NcUV20s200813, alpha_NcUV30s200813, alpha_NcUV40s200813,
    alpha_NcUVCTL200925, alpha_NcUVhalfs200925, alpha_NcUV1s200925, alpha_NcUV2s200925, alpha_NcUV3s200925, alpha_NcUV5s200925, alpha_NcUV7s200925,
    ], axis=1)

df_alpha.round(3).to_csv('/home/linhua/Desktop/alpha.csv', index=False)
