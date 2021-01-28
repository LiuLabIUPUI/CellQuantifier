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

	def merge_physdf(self):
		today = str(date.today().strftime("%y%m%d"))

		phys_files = glob.glob(self.settings['Input path'] + '*' + \
			self.settings['Str in filename'])
		for phys_file in phys_files:
			for exclude_str in self.settings['Strs not in filename']:
				if exclude_str in phys_file:
					phys_files.remove(phys_file)

		phys_files = np.array(phys_files)
		print("######################################")
		print("Total number of physData to be merged: %d" % len(phys_files))
		print("######################################")
		print(phys_files)

		if len(phys_files) > 1:
			ind = 1
			tot = len(phys_files)
			phys_df = merge_physdfs(phys_files, mode=self.settings['Merge mode'])
		else:
			phys_df = pd.read_csv(phys_files[0])

		print("######################################")
		print("Rename particles...")
		print("######################################")
		phys_df['particle'] = phys_df['raw_data'] + phys_df['particle'].apply(str)
		phys_df.round(3).to_csv(self.settings['Output path'] + today + \
						'-physDataMerged.csv', index=False)

		sys.exit()


def get_root_name_list(settings_dict):
	settings = settings_dict.copy()
	root_name_list = []
	path_list = glob.glob(settings['Input path'] + '*' + \
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
