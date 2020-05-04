
def add_region_class(int_df,
                     cls_div_mat=[[1,.1],[1,-.1]],
                     min_class_int=[.1,.1]):

	"""

	Pseudo code
	----------
	1. Add cell types to DataFrame

	Parameters
	----------

	int_df : DataFrame
		DataFrame containing avg intensity and label columns

	cls_div_mat: ndarray, optional
        function parameters for partitioning intensity space
        (two linear functions, one each row)

	min_class_int: ndarray, optional
        lowest avg intensity that will be classified

	Returns
	-------
	int_df : DataFrame
		DataFrame with added cell_type column

	"""

	m1,b1 = cls_div_mat[0]
	m2,b2 = cls_div_mat[1]

	cols = [col for col in int_df.columns if 'avg_' in col]

	int_df.loc[int_df[cols[1]] > \
			   m1*int_df[cols[0]] + b1, 'cell_type'] = 'type1'
	int_df.loc[int_df[cols[1]] < \
			  m2*int_df[cols[0]] + b2, 'cell_type'] = 'type2'

	int_df.loc[(int_df[cols[1]] > m2*int_df[cols[0]] + b2) &  \
			   (int_df[cols[1]] < m1*int_df[cols[0]] + b1), \
               'cell_type'] = 'type3'

	int_df.loc[(int_df[cols[0]] < min_class_int[0]) &  \
			   (int_df[cols[1]] < min_class_int[1]), \
               'cell_type'] = 'type4'

	return int_df
