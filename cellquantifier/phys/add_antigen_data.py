import pandas as pd
import numpy as np

def add_antigen_data(df, sorters=None):

    avg_dist = df.groupby('particle')['dist_to_boundary'].mean()

    df = df.sort_values(['particle', 'frame'])
    delta_x = (df.groupby('particle')['x'].apply(pd.Series.diff))
    delta_y = (df.groupby('particle')['y'].apply(pd.Series.diff))
    # df['v'] = (delta_x**2 + delta_y**2) ** 0.5 * df['pixel_size']
    df['v'] = (delta_x**2 + delta_y**2) ** 0.5

    particles = sorted(df['particle'].unique())

    for particle in particles:
        curr_df = df[ df['particle']==particle ]
        v_max = curr_df['v'].max()
        # travel_dist = ((curr_df['x'].max() - curr_df['x'].min())**2 + \
        #             (curr_df['y'].max() - curr_df['y'].min())**2) ** 0.5 \
        #              * df['pixel_size']
        travel_dist = ((curr_df['x'].max() - curr_df['x'].min())**2 + \
                    (curr_df['y'].max() - curr_df['y'].min())**2) ** 0.5

        df.loc[df['particle']==particle, 'v_max'] = v_max
        df.loc[df['particle']==particle, 'travel_dist'] = travel_dist
        df.loc[df['particle']==particle, 'lifetime'] = curr_df['frame'].max() - \
                                                    curr_df['frame'].min() + 1


        # add 'particle_type' based on dist_to_boundary sorters
        if sorters!=None and sorters['DIST_TO_BOUNDARY'] != None:
            if avg_dist[particle] >= sorters['DIST_TO_BOUNDARY'][0] \
            and avg_dist[particle] <= sorters['DIST_TO_BOUNDARY'][1]:
                df.loc[df['particle'] == particle, 'particle_type'] = 'A'
            elif avg_dist[particle] < sorters['DIST_TO_BOUNDARY'][0]:
                df.loc[df['particle'] == particle, 'particle_type'] = 'B'
            else:
                df.loc[df['particle'] == particle, 'particle_type'] = '--none--'

    # df_particle = df.drop_duplicates('particle')
    # type_counts = df_particle['particle_type'].value_counts()
    # print(len(particles))
    # print(type_counts)

    return df
