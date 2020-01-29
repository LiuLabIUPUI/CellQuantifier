import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from cellquantifier.math import interpolate_lin
from cellquantifier.phys.physutil import bin_df
from cellquantifier.plot.plotutil import *


def plot_fig_3(df,
			   bp1_thres=10,
			   nbins=10,
			   hole_size=1,
			   pixel_size=.1083,
			   frame_rate=3.33,
			   divide_num=5):

	"""
	Construct Figure 3

	Parameters
	----------

	df : DataFrame
	DataFrame containing 'particle', 'alpha' columns

	pixel_size : float

	frame_rate : float

	divide_num : float

	Example
	--------
	import pandas as pd
	from cellquantifier.publish import plot_fig_2
	from cellquantifier.phys.physutil import add_avg_dist
	from cellquantifier.plot.plotutil import *

	df = pd.read_csv('cellquantifier/data/physDataMerged.csv')
	df = add_avg_dist(df)

	boundary_sorter = [-20, 0]
	bp1_sorter = [-50, 10]

	df['sort_flag_boundary'] = df['avg_dist_bound'].between(boundary_sorter[0], \
														boundary_sorter[1],
														inclusive=True)

	df['sort_flag_53bp1'] = df['avg_dist_53bp1'].between(bp1_sorter[0], \
													 bp1_sorter[1],
													 inclusive=True)
	plot_fig_2(df)
	"""

	fig = plt.figure(figsize=(14,6))
	shape = (8,10)

	# """
	# ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
	# """

	ax1 = plt.subplot2grid(shape, (0, 0), rowspan=8, colspan=4) #msd curve
	ax2 = plt.subplot2grid(shape, (0, 4), rowspan=4, colspan=2) #up sp
	ax3 = plt.subplot2grid(shape, (4, 4), rowspan=4, colspan=2) #down sp

	ax4 = plt.subplot2grid(shape, (0, 6), rowspan=4, colspan=2, projection='polar') #heat maps
	ax5 = plt.subplot2grid(shape, (4, 6), rowspan=4, colspan=2, projection='polar') #heat maps

	df_cpy = deepcopy(df)

	# """
	# ~~~~~~~~~~~BLM Data~~~~~~~~~~~~~~
	# """

	blm_df = df.loc[df['exp_label'] == 'BLM']
	blm_df = blm_df.loc[blm_df['avg_dist_53bp1'] < bp1_thres]

	blm_df_bincenters, blm_df_binned = bin_df(blm_df,
											  'avg_dist_53bp1',
											  nbins=nbins)

	blm_df_binned = blm_df_binned.groupby(['category'])
	D_blm = blm_df_binned['D'].mean().to_numpy()
	alpha_blm = blm_df_binned['alpha'].mean().to_numpy()

	r_cont_blm, D_blm = interpolate_lin(blm_df_bincenters,
										D_blm,
										pad_size=hole_size)

	r_cont_blm, alpha_blm = interpolate_lin(blm_df_bincenters,
											alpha_blm,
											pad_size=hole_size)

	# """
	# ~~~~~~~~~~~BLM MSD Curve~~~~~~~~~~~~~~
	# """

	blm_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'BLM']
	add_mean_msd(ax1,
			 blm_df_cpy,
			 'sort_flag_53bp1',
			 pixel_size,
			 frame_rate,
			 divide_num)

	ax1.set_title(r'$\mathbf{BLM}$')

	# """
	# ~~~~~~~~~~~BLM Strip Plots~~~~~~~~~~~~~~
	# """

	add_strip_plot(ax2,
			   blm_df_cpy,
			   'D',
			   'sort_flag_53bp1',
			   xlabels=['Far', 'Near'],
			   ylabel=r'\mathbf{D (nm^{2}/s)}',
			   palette=['blue', 'red'],
			   x_labelsize=8,
			   drop_duplicates=True)

	add_t_test(ax2,
		   blm_df_cpy,
		   cat_col='sort_flag_53bp1',
		   hist_col='D',
		   text_pos=[0.9, 0.9])

	add_strip_plot(ax3,
				blm_df_cpy,
				'alpha',
				'sort_flag_53bp1',
				xlabels=['Far', 'Near'],
				ylabel=r'\mathbf{\alpha}',
				palette=['blue', 'red'],
				x_labelsize=8,
				drop_duplicates=True)


	add_t_test(ax3,
		   blm_df_cpy,
		   cat_col='sort_flag_53bp1',
		   hist_col='alpha',
		   text_pos=[0.9, 0.9])


	# """
	# ~~~~~~~~~~~Heat Maps~~~~~~~~~~~~~~
	# """

	add_heat_map(ax4,
				 blm_df_bincenters,
				 r_cont_blm,
				 D_blm,
				 ylabel=r'$\mathbf{D (nm^{2}/s)}$',
				 nbins=nbins,
				 hole_size=hole_size,
				 edge_ring=True)

	add_heat_map(ax5,
				 blm_df_bincenters,
				 r_cont_blm,
				 alpha_blm,
				 ylabel=r'$\mathbf{\alpha}$',
				 nbins=nbins,
				 hole_size=hole_size,
				 edge_ring=True)

	plt.subplots_adjust(wspace=5, hspace=5)
	plt.show()
