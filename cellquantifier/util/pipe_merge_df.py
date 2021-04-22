import pandas as pd; import numpy as np
from datetime import date, datetime
import glob
import sys

from ..phys.physutil import merge_physdfs


class Pipe():

	def __init__(self, settings_dict, control_list, root_name):
		settings_dict['Processed date'] = \
			datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		self.settings = settings_dict
		self.control = control_list
		self.root_name = root_name

	def merge_df(self):
		df_list = np.array(sorted(glob.glob(self.settings['Input path'] + \
			'*' + self.root_name + '*.csv')))
		print(df_list)

		if self.settings['Cols to merge']:
			dfs = []
			for file in df_list:
				tmp_df = pd.read_csv(file, index_col=False)
				tmp_df = tmp_df[self.settings['Cols to merge']]
				dfs.append(tmp_df)
			phys_df = pd.concat(dfs, ignore_index=True)
			print(phys_df)
		else:
			phys_df = merge_physdfs(df_list, mode='general')
			print(phys_df)

		phys_df.round(3).to_csv(self.settings['Output path'] + \
			self.root_name + '-detData.csv', index=False)



def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob.glob(settings['Input path'] + '*' + '.tif')
	for path in path_list:
		filename = path.split('/')[-1]
		root_name = filename[:filename.find('.tif')]
		root_name_list.append(root_name)

	return np.array(sorted(root_name_list))


def pipe_batch(settings_dict, control_list):

	root_name_list = np.array(sorted(get_root_name_list(settings_dict)))
	print(root_name_list)

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
