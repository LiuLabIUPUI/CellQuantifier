import pandas as pd; import numpy as np
import os
from datetime import date, datetime
import glob
import sys
import matplotlib.pyplot as plt

import seaborn as sns
from ..plot.plotutil import *
from ..plot.plotutil._add_mean_msd2 import add_mean_msd2

class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

	def save_config(self):
		settings_df = pd.DataFrame.from_dict(data=self.settings, orient='index')
		settings_df = settings_df.drop(['Input path', 'Output path'])
		settings_df.to_csv(self.settings['Output path'] + self.root_name + \
				'-config-pltbox.csv', header=False)

	def generate_boxplot(self):
		# """
		# ~~~~Initialize the colors~~~~
		# """
		print("\n")
		print("Preparing colors")
		# palette = sns.color_palette('muted')
		# c1 = palette[0]
		# c2 = palette[1]
		# c3 = palette[2]
		c1 = (1, 0, 0)
		c2 = (0, 0, 0)
		c3 = (0, 0, 1)
		c4 = (0.5, 0.5, 0.5)
		RGBA_alpha = 0.8
		c1_alp = (c1[0], c1[1], c1[2], RGBA_alpha)
		c2_alp = (c2[0], c2[1], c2[2], RGBA_alpha)
		c3_alp = (c3[0], c3[1], c3[2], RGBA_alpha)

		p = [c1, c2, c3]

		p0 = [c2, c3]

		p1 = [c2, c4]


		pa = [c3, c2]
		pb = [c1, c2]


		# """
		# ~~~~Initialize the page layout~~~~
		# """
		# Layout settings
		col_num = 3
		row_num = 1
		divide_index = [
			]
		hidden_index = []
		# Sub_axs_1 settings
		col_num_s1 = 1
		row_num_s1 = 2
		index_s1 = [
			]
		# Sub_axs_2 settings
		col_num_s2 = 1
		row_num_s2 = 2
		index_s2 = [
			]

		# Layout implementation
		print("\n")
		print("Preparing layout")
		tot_width = col_num * 4
		tot_height = row_num * 3
		all_figures, page = plt.subplots(1, 1, figsize=(tot_width, tot_height))

		grids = []
		axs = []

		axs_s1_bg = []
		axs_s1 = []
		axs_s1_base = []
		axs_s1_slave = []

		axs_s2_bg = []
		axs_s2 = []
		axs_s2_base = []
		axs_s2_slave = []
		for i in range(col_num*row_num):
			r = i // col_num
			c = i % col_num
			w = 1 / col_num
			h = 1 / row_num
			x0 = c * w
			y0 = 1 - (r+1) * h

			# Generate Grids
			grids.append(page.inset_axes([x0, y0, w, h]))

			# Generate individual axs
			axs.append(grids[i].inset_axes([0.33, 0.33, 0.6, 0.6]))

			# Customize axs_s1
			if i in index_s1:
				axs_s1_bg.append(axs[i])
				for i_s1 in range(col_num_s1*row_num_s1):
					r_s1 = i_s1 // col_num_s1
					c_s1 = i_s1 % col_num_s1
					w_s1 = 1 / col_num_s1
					h_s1 = 1 / row_num_s1
					x0_s1 = c_s1 * w_s1
					y0_s1 = 1 - (r_s1+1) * h_s1
					# Generate axs_s1, axs_s1_base, axs_s1_slave
					temp = axs[i].inset_axes([x0_s1, y0_s1, w_s1, h_s1])
					axs_s1.append(temp)
					if y0_s1 == 0:
						axs_s1_base.append(temp)
					else:
						axs_s1_slave.append(temp)

			# Customize axs_s2
			if i in index_s2:
				axs_s2_bg.append(axs[i])
				for i_s2 in range(col_num_s2*row_num_s2):
					r_s2 = i_s2 // col_num_s2
					c_s2 = i_s2 % col_num_s2
					w_s2 = 1 / col_num_s2
					h_s2 = 1 / row_num_s2
					x0_s2 = c_s2 * w_s2
					y0_s2 = 1 - (r_s2+1) * h_s2
					# Generate axs_s2, axs_s2_base, axs_s2_slave
					temp = axs[i].inset_axes([x0_s2, y0_s2, w_s2, h_s2])
					axs_s2.append(temp)
					if y0_s2 == 0:
						axs_s2_base.append(temp)
					else:
						axs_s2_slave.append(temp)

		# """
		# ~~~~format figures~~~~
		# """
		print("\n")
		print("Formating figures")
		# Format page
		for ax in [page]:
			ax.set_xticks([]);
			ax.set_yticks([])
			format_spine(ax, spine_linewidth=2)

		# Format grids
		for ax in grids:
			ax.set_xticks([]);
			ax.set_yticks([])
			format_spine(ax, spine_linewidth=2)
			for spine in ['top', 'bottom', 'left', 'right']:
				ax.spines[spine].set_visible(False)

		for i in divide_index:
			for spine in ['bottom']:
				grids[i].spines[spine].set_visible(True)

		# Format axs
		for ax in axs:
			format_spine(ax, spine_linewidth=0.5)
			format_tick(ax, tk_width=0.5)
			format_tklabel(ax, tklabel_fontsize=10)
			format_label(ax, label_fontsize=10)

		for i in hidden_index:
			axs[i].set_xticks([]);
			axs[i].set_yticks([])
			for spine in ['top', 'bottom', 'left', 'right']:
				axs[i].spines[spine].set_visible(False)

		# Format sub_axs_background
		for ax in axs_s1_bg + axs_s2_bg:
			ax.set_xticks([])
			ax.set_yticks([])
			for spine in ['top', 'bottom', 'left', 'right']:
				ax.spines[spine].set_visible(False)

		# Format sub_axs
		for ax in axs_s1 + axs_s2:
			format_spine(ax, spine_linewidth=0.5)
			format_tick(ax, tk_width=0.5)
			format_tklabel(ax, tklabel_fontsize=10)
			format_label(ax, label_fontsize=10)
			ax.set_yticks([])

		# Format sub_axs_slave
		for ax in axs_s1_slave + axs_s2_slave:
			# labels = [item.get_text() for item in ax.get_xticklabels()]
			# empty_string_labels = ['']*len(labels)
			# ax.set_xticklabels(empty_string_labels)
			#
			# labels = [item.get_text() for item in ax.get_yticklabels()]
			# empty_string_labels = ['']*len(labels)
			# ax.set_yticklabels(empty_string_labels)
			ax.set_xticks([])

		# """
		# ~~~~Prepare df for the whole page~~~~
		# """
		df = pd.read_csv('/home/linhua/Desktop/input/colocal.csv')

		# """
		# ~~~~Plot boxplot~~~~
		# """
		figs = [
				axs[0], axs[1], axs[2],
				]
		datas = [
				df, df, df,
				]
		data_cols = [
				'PCC', 'ovacolocal', 'mhc1colocal',
				]
		palettes = [
				p0, p0, p0,
				]
		orders = [
				['WT', 'KD'], ['WT', 'KD'], ['WT', 'KD'],
				]
		xlabels = [
				'', '', '',
				]
		ylabels = [
				'Pixel corelation coeff',
				'OVA overlap ratio',
				'MHC1 overlap ratio'
				]
		for i, (fig, data, data_col, palette, order, xlabel, ylabel,) \
		in enumerate(zip(figs, datas, data_cols, palettes, orders, xlabels, ylabels,)):
			print("\n")
			print("Plotting (%d/%d)" % (i+1, len(figs)))
			sns.boxplot(ax=fig,
						x='exp_label',
						y=data_col,
						data=data,
						order=order,
						palette=palette,
						linewidth=1,
						boxprops=dict(alpha=RGBA_alpha, linewidth=1, edgecolor=(0,0,0)),
						saturation=1,
						fliersize=2,
						# whis=[0, 100],
						)
			sns.swarmplot(ax=fig,
						x='exp_label',
						y=data_col,
						data=data,
						order=order,
						color="0",
						size=3,
						)

			set_xylabel(fig,
						xlabel=xlabel,
						ylabel=ylabel,
						)



		# # """
		# # ~~~~Add t test~~~~
		# # """
		# figs = [
		#         axs[1], axs[2], axs[3], axs[4],
		#         ]
		# datas = [
		#         dfp, dfp, dfp, dfv,
		#         ]
		# data_cols = [
		#         'D', 'alpha', 'travel_dist', 'v',
		#         ]
		# cat_cols = [
		#         'exp_label', 'exp_label', 'exp_label', 'exp_label',
		#         ]
		# text_poss = [
		#         (0.98, 0.88), (0.98, 0.88), (0.98, 0.88), (0.98, 0.88),
		#         ]
		# for i, (fig, data, data_col, cat_col, text_pos, ) \
		# in enumerate(zip(figs, datas, data_cols, cat_cols, text_poss, )):
		#     print("\n")
		#     print("Plotting (%d/%d)" % (i+1, len(figs)))
		#
		#     add_t_test(fig,
		#                 blobs_df=data,
		#                 cat_col=cat_col,
		#                 hist_col=data_col,
		#                 drop_duplicates=False,
		#                 text_pos=text_pos,
		#                 color=(0,0,0,1),
		#                 fontname='Liberation Sans',
		#                 fontweight=9,
		#                 fontsize=9,
		#                 horizontalalignment='right',
		#                 format='general',
		#                 )

		# # """
		# # ~~~~Add figure text~~~~
		# # """
		# figs = grids
		# fig_texts = [
		#         'Fig.1a. Mean MSD curve comparion',
		#         'Fig.1b. D value comparison',
		#         'Fig.1c. Alpha value comparion',
		#         'Fig.2. Trajectory travel range comparison',
		#         'Fig.3. Velocity comparison',
		#         ]
		# for i, (fig, fig_text, ) \
		# in enumerate(zip(figs, fig_texts, )):
		#     print("\n")
		#     print("Plotting (%d/%d)" % (i+1, len(figs)))
		#
		#     fig.text(0.1,
		#             0.05,
		#             fig_text,
		#             horizontalalignment='left',
		#             color=(0,0,0,1),
		#             family='Liberation Sans',
		#             fontweight=10,
		#             fontsize=10,
		#             transform=fig.transAxes,
		#             )


		# """
		# ~~~~Additional figures format~~~~
		# """
		# # Format legend
		# for ax in msd_figs:
		#     format_legend(ax,
		#             show_legend=True,
		#             legend_loc='lower right',
		#             legend_fontweight=7,
		#             legend_fontsize=7,
		#             )
		# # Rename legend
		# for ax in rename_msd_figs:
		#     rename_legend(ax,
		#             new_labels=['Boundary', 'Inside'],
		#             replace_ind=1,
		#             replace_type='prefix',
		#             legend_loc='lower right',
		#             legend_fontweight=7,
		#             legend_fontsize=7,
		#             )
		# # Format scale
		# figs = axs
		# xscales = [
		#         [-2, 32],
		#         [None, None],  [None, None], [None, None], [None, None],
		#         ]
		# yscales = [
		#         [5000, 30000],
		#         # [-1000,18000],  [-0.1, 2], [0, 1.5], [-0.01, 0.2],
		#         [-300, 15500],  [-0.1, 2], [0, 1.5], [None, None],
		#         ]
		# for i, (fig, xscale, yscale, ) \
		# in enumerate(zip(figs, xscales, yscales,)):
		#     format_scale(fig,
		#             xscale=xscale,
		#             yscale=yscale,
		#             )


		# """
		# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~
		# """
		all_figures.savefig('/home/linhua/Desktop/Figure_1.pdf', dpi=600)
		plt.clf(); plt.close()
		sys.exit()



def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob(settings['Input path'] + '*' + \
		settings['Str in filename'])
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('.')]
		root_name_list.append(root_name)

		for exclude_str in settings['Strs not in filename']:
			if exclude_str in root_name:
				root_name_list.remove(root_name)

	return np.array(sorted(root_name_list))


def pipe_batch(settings_dict, control_list):

	root_name_list = get_root_name_list(settings_dict)

	print("######################################")
	print("Total data num to be processed: %d" % len(root_name_list))
	print(root_name_list)
	print("######################################")

	ind = 0
	tot = len(root_name_list)
	for root_name in root_name_list:
		ind = ind + 1
		print("\n")
		print("Processing (%d/%d): %s" % (ind, tot, root_name))

		pipe = Pipe(settings_dict, control_list, root_name)
		for func in control_list:
			getattr(pipe, func)()
