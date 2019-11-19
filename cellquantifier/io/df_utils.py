import pandas as pd

def relabel_particles(df):

    file_names = df['raw_data'].unique()
    i = 0
    for file_name in file_names:
      sub_df = df.loc[df['raw_data'] == file_name]
      particles = sub_df['particle'].unique()

      for particle in particles:

        df.loc[(df['raw_data'] == file_name) & \
        (df['particle'] == particle), 'particle'] = i

        i+=1

    return df

def merge_dfs(files):

    temp_df = pd.read_csv(files[0], index_col=False)
    columns = temp_df.columns.tolist()
    merged_df = pd.DataFrame([], columns=columns)

    for file in files:
        df = pd.read_csv(file, index_col=False)
        df['raw_data'] = file
        merged_df = pd.concat([merged_df, df], sort=True, ignore_index=True)

    merged_df = merged_df.apply(pd.to_numeric)

    return merged_df
