import matplotlib.pyplot as plt
from cellquantifier.plot.plotutil import *

def plot_engine(plot_params, df, image=None):

	"""Primary plotting function

	Parameters
	----------

	"""

	fig, ax = plt.subplots(1,1)

	if plot_params['t_test']:
		add_t_test(ax,
				   df,
				   cat_col=plot_params['cat_col'],
				   data_col=plot_params['data_col'])

	if plot_params['plot_type'] == 'hist':

		add_hist(ax,
				 df,
				 cat_col=plot_params['cat_col'],
				 nbins=plot_params['hist_nbins'],
				 color=plot_params['color'],
				 density=plot_params['hist_normalize'])

	if plot_params['plot_type'] == 'msd':

		add_mean_msd(ax,
					 df,
					 cat_col=plot_params['cat_col'],
					 pixel_size=1,
					 frame_rate=1,
					 divide_num=5)

	if plot_params['plot_type'] == 'scatter':
		add_scatter(ax,
					df,
					col1=plot_params['x'],
					col2=plot_params['y'],
					color=plot_params['color'],
					fit=plot_params['scatter_fit'],
					norm=plot_params['scatter_norm']
					)

	if plot_params['plot_type'] == 'violin':
		add_violin(ax,
				   df,
				   cat_col=plot_params['cat_col'],
				   data_col=plot_params['data_col'])


	if plot_params['plot_type'] == 'imshow':
		add_image(ax,
				  image,
				  plot_params['pixel_size'],
				  df,
				  color=plot_params['color'],
				  plot_r=plot_params['plot_r'])


	ax.set_xlabel(plot_params['x_unit'], fontsize=plot_params['x_unit_fontsize'])
	ax.set_ylabel(plot_params['y_unit'], fontsize=plot_params['y_unit_fontsize'])
	plt.axis('off')
	plt.tight_layout()
	plt_array = plt2array(fig)

	return plt_array
