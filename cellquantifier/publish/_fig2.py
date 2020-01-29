import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from cellquantifier.math import interpolate_lin
from cellquantifier.phys.physutil import bin_df
from cellquantifier.plot.plotutil import *


def plot_fig_2(df,
			   hole_size=10,
			   nbins=12,
			   pixel_size=.1083,
			   frame_rate=3.33,
			   divide_num=5):

	"""
	Construct Figure 2

	Parameters
	----------

	df : DataFrame
		DataFrame containing 'particle', 'alpha' columns

	hole_size: int
		Size of the black hole in the center of the heat map

	nbins: int
		Number of bins to use for heatmap

	pixel_size : float
		Pixel size in um

	frame_rate : float
		Frame rate in frames per second (fps)

	divide_num : float
		Divide number to use for the MSD curves

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

	fig = plt.figure(figsize=(14,12))
	shape = (16,12)

	# """
	# ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
	# """

	ax1 = plt.subplot2grid(shape, (0, 0), rowspan=3, colspan=3, projection='polar') #heat maps
	ax2 = plt.subplot2grid(shape, (0, 3), rowspan=3, colspan=3, projection='polar') #heat maps
	ax3 = plt.subplot2grid(shape, (3, 0), rowspan=3, colspan=3, projection='polar') #heat maps
	ax4 = plt.subplot2grid(shape, (3, 3), rowspan=3, colspan=3, projection='polar') #heat maps
	# ax5 = plt.subplot2grid(shape, (0, 4), rowspan=2, colspan=4) #mask fig
	ax6 = plt.subplot2grid(shape, (6, 0), rowspan=8, colspan=4) #ctrl msd curve
	ax7 = plt.subplot2grid(shape, (6, 4), rowspan=4, colspan=2) #ctrl up sp
	ax8 = plt.subplot2grid(shape, (10, 4), rowspan=4, colspan=2) #ctrl down sp
	ax9 = plt.subplot2grid(shape, (6, 6), rowspan=8, colspan=4) #blm msd curve
	ax10 = plt.subplot2grid(shape, (6, 10), rowspan=4, colspan=2) #blm up sp
	ax11 = plt.subplot2grid(shape, (10, 10), rowspan=4, colspan=2) #blm down sp

	df_cpy = deepcopy(df)

	# """
	# ~~~~~~~~~~~CTRL Data~~~~~~~~~~~~~~
	# """

	ctrl_df = df.loc[df['exp_label'] == 'Ctr']

	ctrl_df_bincenters, ctrl_df_binned = bin_df(ctrl_df,
											    'avg_dist_bound',
												nbins=nbins)

	ctrl_df_binned = ctrl_df_binned.groupby(['category'])
	D_ctrl = ctrl_df_binned['D'].mean().to_numpy()
	alpha_ctrl = ctrl_df_binned['alpha'].mean().to_numpy()

	r_cont_ctrl, D_ctrl = interpolate_lin(ctrl_df_bincenters,
										  D_ctrl,
										  pad_size=hole_size)

	r_cont_ctrl, alpha_ctrl = interpolate_lin(ctrl_df_bincenters,
											  alpha_ctrl,
											  pad_size=hole_size)

	# """
	# ~~~~~~~~~~~BLM Data~~~~~~~~~~~~~~
	# """

	blm_df = df.loc[df['exp_label'] == 'BLM']

	blm_df_bincenters, blm_df_binned = bin_df(blm_df,
											  'avg_dist_bound',
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
	# ~~~~~~~~~~~Heat Maps~~~~~~~~~~~~~~
	# """

	D_all =  np.concatenate((D_ctrl, D_blm))
	min = D_all[D_all != 0].min()
	max = D_all[D_all != 0].max()
	xlabel = r'$\mathbf{D (nm^{2}/s)}$'
	ylabel = r'$\mathbf{D (nm^{2}/s)}$'

	add_heat_map(ax1,
				 ctrl_df_bincenters,
				 r_cont_ctrl,
				 D_ctrl,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max))

	add_heat_map(ax2,
				 blm_df_bincenters,
				 r_cont_blm,
				 D_blm,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max))


	xlabel = r'$\mathbf{\alpha}$'
	ylabel = r'$\mathbf{\alpha}$'

	alpha_all =  np.concatenate((alpha_ctrl, alpha_blm))
	min = alpha_all[alpha_all != 0].min()
	max = alpha_all[alpha_all != 0].max()

	add_heat_map(ax3,
				 ctrl_df_bincenters,
				 r_cont_ctrl,
				 alpha_ctrl,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max))

	add_heat_map(ax4,
				 blm_df_bincenters,
				 r_cont_blm,
				 alpha_blm,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max))

	ax1.set_title(r'$\mathbf{CTRL}$')
	ax2.set_title(r'$\mathbf{BLM}$')

	# """
	# ~~~~~~~~~~~CTRL MSD Curve~~~~~~~~~~~~~~
	# """

	ctrl_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'Ctr']
	add_mean_msd(ax6,
				 ctrl_df_cpy,
				 'sort_flag_boundary',
				 pixel_size,
				 frame_rate,
				 divide_num)

	ax6.set_title(r'$\mathbf{CTRL}$')

	# """
	# ~~~~~~~~~~~CTRL Strip Plots~~~~~~~~~~~~~~
	# """

	add_strip_plot(ax7,
				   ctrl_df,
				   'D',
				   'sort_flag_boundary',
				   xlabels=['Interior', 'Boundary'],
				   ylabel=r'\mathbf{D (nm^{2}/s)}',
				   palette=['blue', 'red'],
				   x_labelsize=8,
				   drop_duplicates=True)

	add_t_test(ax7,
			   ctrl_df,
			   cat_col='sort_flag_boundary',
			   hist_col='D',
			   text_pos=[0.9, 0.9])

	ax = add_strip_plot(ax8,
						ctrl_df,
						'alpha',
						'sort_flag_boundary',
						xlabels=['Interior', 'Boundary'],
						ylabel=r'\mathbf{\alpha}',
						palette=['blue', 'red'],
						x_labelsize=8,
						drop_duplicates=True)

	add_t_test(ax8,
			   ctrl_df,
			   cat_col='sort_flag_boundary',
			   hist_col='alpha',
			   text_pos=[0.9, 0.9])


	# """
	# ~~~~~~~~~~~BLM MSD Curve~~~~~~~~~~~~~~
	# """

	blm_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'BLM']
	add_mean_msd(ax9,
				 blm_df_cpy,
				 'sort_flag_boundary',
				 pixel_size,
				 frame_rate,
				 divide_num)

	ax9.set_title(r'$\mathbf{BLM}$')

	# """
	# ~~~~~~~~~~~BLM Strip Plot~~~~~~~~~~~~~~
	# """

	add_strip_plot(ax10,
				   blm_df,
				   'D',
				   'sort_flag_boundary',
				   xlabels=['Interior', 'Boundary'],
				   ylabel=r'\mathbf{D (nm^{2}/s)}',
				   palette=['blue', 'red'],
				   x_labelsize=8,
				   drop_duplicates=True)

	add_t_test(ax10,
			   blm_df,
			   cat_col='sort_flag_boundary',
			   hist_col='D',
			   text_pos=[0.9, 0.9])

	add_strip_plot(ax11,
				   blm_df,
				   'alpha',
				   'sort_flag_boundary',
				   xlabels=['Interior', 'Boundary'],
				   ylabel=r'\mathbf{\alpha}',
				   palette=['blue', 'red'],
				   x_labelsize=8,
				   drop_duplicates=True)

	add_t_test(ax11,
			   blm_df,
			   cat_col='sort_flag_boundary',
			   hist_col='alpha',
			   text_pos=[0.9, 0.9])

	plt.subplots_adjust(wspace=100, hspace=100)
	plt.show()
