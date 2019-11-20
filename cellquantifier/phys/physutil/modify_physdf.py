import pandas as pd
import re

def relabel_particles(df):

    """
    Relabel particles after merging dataframes from several experiments

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    df: DataFrame
        input df with relabeled particles

    """
    df.sort_values(by=['raw_data', 'particle'])
    file_names = df['raw_data'].unique()
    i = 0

    for file_name in file_names:
      sub_df = df.loc[df['raw_data'] == file_name]
      particles = sub_df['particle'].unique()

      for particle in particles:
        df.loc[(df['raw_data'] == file_name) & \
        (df['particle'] == particle), 'new_label'] = i
        i+=1

    df['new_label'] = df['new_label'].astype('int')
    df['particle'] = df['new_label']; del df['new_label']

    return df

def merge_physdfs(files):

    """
    Relabel particles after merging dataframes from several experiments

    Parameters
    ----------
    files: list
        list of physData files to be merged

    Returns
    -------
    merged_df: DataFrame
        DataFrame after merging

    """

    temp_df = pd.read_csv(files[0], index_col=False)
    columns = temp_df.columns.tolist()
    merged_df = pd.DataFrame([], columns=columns, dtype='int')

    for file in files:
        df = pd.read_csv(file, index_col=False)
        root_name = file.split('/')[-1]
        exp = re.findall(r'[a-zA-Z]{3}\d{1}', file)
        df = df.assign(raw_data=root_name)
        df = df.assign(exp_label=exp[0][:-1])
        merged_df = pd.concat([merged_df, df], sort=True, ignore_index=True)

    return merged_df
