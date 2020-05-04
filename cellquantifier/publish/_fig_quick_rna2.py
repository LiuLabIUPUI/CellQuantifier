import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cellquantifier.plot.plotutil import format_ax
from itertools import combinations
from scipy import stats
from statannot import add_stat_annotation

def fig_quick_rna2(merged_blobs_df, merged_int_df,
				   typ_arr=['type1','type2','type3','type4']):

	"""

	Post-processing figure or RNA expression analysis

	Pseudo code
	----------
	1. Build the figure
	2. Group by 'label', 'cell_type' and 'prefix' columns

	Parameters
	----------
	merged_blobs_df : DataFrame
		DataFrame containing x,y,peak, and label columns
	merged_int_df: DataFrame
		Dataframe containing label column and avg intensity columns
		(1 avg intensity column per channel used for cell classification)
	typ_arr: list
		List of unique types some of which may be in merged_xxx_df

	"""

	fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))

	# """
	# ~~~~~~~~~~~Get dataframe arrays~~~~~~~~~~~~~~
	# """

	count_df = merged_blobs_df.groupby(['label','cell_type','prefix'] \
								 ).size().reset_index(name="count")

	count_df_arr = [count_df.loc[count_df['cell_type'] == typ, 'count'] \
					for typ in typ_arr]

	peak_df_arr = [merged_blobs_df.loc[merged_blobs_df['cell_type'] == typ, 'peak'] \
					for typ in typ_arr]

	# """
	# ~~~~~~~~~~~Copy number box plots~~~~~~~~~~~~~~
	# """

	bp1 = ax1.boxplot(count_df_arr, showfliers=False, patch_artist=True)
	format_ax(ax1, ax_is_box=False)
	ax1.set_ylabel(r'$\mathbf{Copy-Number}$', fontsize=12)
	ax1.set_xticklabels([r'$\mathbf{Type 1}$',\
						 r'$\mathbf{Type 2}$',\
						 r'$\mathbf{Type 3}$',\
						 r'$\mathbf{Type 4}$'],\
						 fontsize=12)

	# """
	# ~~~~~~~~~~~Peak intensity box plots~~~~~~~~~~~~~~
	# """

	bp2 = ax2.boxplot(peak_df_arr, showfliers=False, patch_artist=True)
	format_ax(ax2, ax_is_box=False)
	ax2.set_ylabel(r'$\mathbf{Peak-Intensity}$', fontsize=12)
	ax2.set_xticklabels([r'$\mathbf{Type 1}$',\
						 r'$\mathbf{Type 2}$',\
						 r'$\mathbf{Type 3}$',\
						 r'$\mathbf{Type 4}$'],\
						 fontsize=12)
	# """
	# ~~~~~~~~~~~Add patient column~~~~~~~~~~~~~~
	# """

	for row in range(len(merged_int_df)):
		merged_int_df.at[row, 'patient'] = \
		merged_int_df.at[row, 'prefix'].split('_')[-2]

	count_df = merged_int_df.groupby(['cell_type','patient'] \
									 ).size().reset_index(name="count")

	pivoted = pd.pivot_table(count_df,
					   index=['cell_type','patient'],
					   values='count',
					   fill_value = 0,
					   dropna=False,
					   aggfunc=np.sum)

	count_df = pd.DataFrame(pivoted.to_records())

	# """
	# ~~~~~~~~~~~Cell counts table by patient, cell_types~~~~~~~~~~~~~~
	# """

	count_df2 = pd.DataFrame(columns=['patient']+typ_arr)
	patients = count_df['patient'].unique()
	for i, patient in enumerate(patients):
		counts = count_df.loc[count_df['patient'] == patient, \
							 'count'].to_numpy()
		count_df2.loc[i] = [patient] + list(counts)

	count_df2['sum'] = count_df2[typ_arr].sum(axis=1)
	count_df2.loc['Total']= count_df2[typ_arr+['sum']].sum(axis=0)

	cell_text = []
	for row in range(len(count_df2)):
		cell_text.append(count_df2.iloc[row])

	table = ax3.table(cellText=cell_text, colLabels=count_df2.columns, \
					  loc='center')

	# """
	# ~~~~~~~~~~~Get p values~~~~~~~~~~~~~
	# """

	count_pval_df = pd.DataFrame(columns=['pair','pval'])
	peak_pval_df = pd.DataFrame(columns=['pair','pval'])
	num_types = len(typ_arr)

	pairs = list(combinations([0,1,2,3],2))
	for i,pair in enumerate(pairs):
		s1 = stats.ttest_ind(peak_df_arr[pair[0]],peak_df_arr[pair[1]])
		s2 = stats.ttest_ind(count_df_arr[pair[0]],count_df_arr[pair[1]])
		count_pval_df.loc[i] = [pair, round(s1[1], 5)]
		peak_pval_df.loc[i] = [pair, round(s2[1],5)]


	table.scale(1,2)
	ax3.axis('off')
	plt.tight_layout()
