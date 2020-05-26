import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..phys import *
from ..phys.physutil import *
from ..plot.plotutil import *

from itertools import combinations
from scipy import stats


def fig_quick_rna_2(input_dir, typ_arr=['type1','type2','type3','type4'],
					sample_type=None):

	"""

	Table that shows the number of cells detected for each sample type

	Pseudo code
	----------
	1. Build the figure
	2. Build the table

	Parameters
	----------
	input_dir : str,
		Directory containing the files of interest
	typ_arr: list
		List of unique types some of which may be in merged_xxx_df
	sample_type : str, optional
		A specific sample type to generate the table for

	"""

	fig, ax = plt.subplots(figsize=(10,2))

	# """
	# ~~~~~~~~~~~Merge blobs_dfs and int_dfs~~~~~~~~~~~~~~
	# """

	prefixes = get_unique_prefixes(input_dir, tag='hla-fittData-lbld.csv')
	merged_blobs_df, merged_int_df = merge_physdfs3(input_dir, prefixes)

	if sample_type:
		merged_int_df = merged_int_df.loc[merged_int_df['sample_type'] == sample_type]

	merged_int_df = merged_int_df.reindex()


	# """
	# ~~~~~~~~~~~Generate table~~~~~~~~~~~~~
	# """

	for i, row in merged_int_df.iterrows():
		row['prefix'] = row['prefix'].split('_')[-2]

	count_df = merged_int_df.groupby(['cell_type','prefix'] \
									 ).size().reset_index(name="count")

	pivoted = pd.pivot_table(count_df,
					   index=['cell_type','prefix'],
					   values='count',
					   fill_value = 0,
					   dropna=False,
					   aggfunc=np.sum)

	count_df = pd.DataFrame(pivoted.to_records())

	# """
	# ~~~~~~~~~~~Cell counts table by patient, cell_types~~~~~~~~~~~~~~
	# """

	count_df2 = pd.DataFrame(columns=['prefix']+typ_arr)
	patients = count_df['prefix'].unique()
	for i, patient in enumerate(patients):
		counts = count_df.loc[count_df['prefix'] == patient, \
							 'count'].to_numpy()
		count_df2.loc[i] = [patient] + list(counts)

	count_df2['sum'] = count_df2[typ_arr].sum(axis=1)
	count_df2.loc['Total']= count_df2[typ_arr+['sum']].sum(axis=0)

	cell_text = []
	for row in range(len(count_df2)):
		cell_text.append(count_df2.iloc[row])

	table = ax.table(cellText=cell_text, colLabels=count_df2.columns, \
					  loc='center')

	# table.scale(1,2)
	ax.axis('off')
	plt.tight_layout()
