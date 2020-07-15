import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, Qt, QSettings

from cellquantifier.util.config import *
from cellquantifier.util.pipeline import Pipeline
from .widgets import *
from ._util import isfloat, isint
from pathlib import Path
from shutil import copyfile

class Ui_MainWindow(object):


	def setup_ui(self, MainWindow):

		"""
		Initialize the containers (panels) containing widgets
		"""

		# """
		# ~~~~~~~~~~Settings~~~~~~~~~~~~~~
		# """

		self.settings = CQSettings()

		# """
		# ~~~~~~~~~~~Central Widget~~~~~~~~~~~~~~
		# """

		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1920, 1080)
		self.centralwidget = QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.centralwidget.layout = QHBoxLayout()

		# """
		# ~~~~~~~~~~Left Dock~~~~~~~~~~~~~~
		# """

		self.left_dock = QDockWidget(MainWindow)
		self.file_widget = FileWidget(self)
		self.left_dock.setWidget(self.file_widget)
		MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.left_dock)

		# """
		# ~~~~~~~~~~Right Dock~~~~~~~~~~~~~~
		# """

		self.right_dock = QDockWidget(MainWindow)
		self.pipeline_widget = PipelineWidget(self)
		self.right_dock.setWidget(self.pipeline_widget)
		MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.right_dock)


		# """
		# ~~~~~~~~~~~Tab Widget~~~~~~~~~~~~
		# """

		self.tabs = QTabWidget()

		df = pd.read_csv('cellquantifier/data/testData.csv')
		self.data_view = DataView(self)
		self.data_view.update_df(df)
		self.plot_widget = PlotWidget(self)

		self.tabs.addTab(self.plot_widget, 'Plot')
		self.tabs.addTab(self.data_view, 'Data')
		self.centralwidget.layout.addWidget(self.tabs)

		# """
		# ~~~~~~~~~~Menu Bar~~~~~~~~~~~~~~
		# """

		self.menu_bar = MainWindow.menuBar()
		self.menu_bar.file_menu = self.menu_bar.addMenu('File')
		self.menu_bar.edit_menu = self.menu_bar.addMenu('Edit')

		self.menu_bar.file_menu.addAction('Open')
		self.menu_bar.file_menu.addAction('Exit')
		self.menu_bar.edit_menu.addAction('Plot Preferences')

		self.menu_bar.file_menu.triggered[QAction].connect(self.process_file_trigger)
		self.menu_bar.edit_menu.triggered[QAction].connect(self.process_edit_trigger)

		# """
		# ~~~~~~~~~~Finish~~~~~~~~~~~~~
		# """

		self.centralwidget.setLayout(self.centralwidget.layout)
		MainWindow.setCentralWidget(self.centralwidget)


	def process_edit_trigger(self,q):

		dialog = PlotPrefsDialog(self)
		dialog.exec_()

	def process_file_trigger(self, q):

		action = q.text()
		if action == 'Exit':
			sys.exit()
		elif action == 'Open':
			self.open_file_dialog()

	def open_file_dialog(self):

		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(None,\
		"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files \
		(*.py)", options=options)

		if file_name:
			root = file_name.split('/')[-1]
			dst = str(Path.home()) + '/Desktop/temp/' + root
			copyfile(file_name, dst)
			self.file_widget.update_file_list()

	def pull_config(self):

		"""
		Pull the configuration from the ui and generate config dict
		"""

		self.config = {}
		pipe_tabs = self.pipeline_widget.tabs

		# """
		# ~~~~~~~~~~~Pull pipeline params~~~~~~~~~~~~~~
		# """

		num_tabs = pipe_tabs.count()

		for i in range(num_tabs):
			tab = pipe_tabs.widget(i)
			rows = tab.table.rowCount()

			for row in range(rows):
				key = tab.table.vert_labels[row]
				key = ' '.join(key.split('/'))
				value = tab.table.item(row,0).text()
				self.config[key] = value

		# """
		# ~~~~~~~~~~~Add Input Path~~~~~~~~~~~~~~
		# """

		items = self.file_widget.file_list.selectedItems()

		for item in items:
			path = str(Path.home()) + '/Desktop/temp/' + item.text(0)

		root_name = path.split('/')[-1].split('.')[0]
		path = '/'.join(path.split('/')[:-1]) + '/'
		self.config['Meta input_path'] = path
		self.config['Meta root_name'] = root_name

		self.config['Diag diagnostic'] = False
		self.config['Diag pltshow'] = False

		# """
		# ~~~~~~~~~~~Enforce Config Datatypes~~~~~~~~~~~~~~
		# """

		for key, value in self.config.items():

			if isint(value):
				self.config[key] = int(value)
			elif isfloat(value):
				self.config[key] = float(value)

	def pipeline_control(self):

		"""
		Instantiate and run the pipeline
		"""

		self.pull_config()
		self.config = Config(self.config)

		ops = self.pipeline_widget.chkbxs.group
		pipe = Pipeline(self.config, is_new=True)

		# """
		# ~~~~~~~~~Run the pipeline~~~~~~~~~~~~~~
		# """

		self.plot_widget.figure.clear()

		if ops.regi_box.isChecked():
			pipe.register()

		if ops.segm_box.isChecked():
			pipe.segmentation()

		if ops.deno_box.isChecked():

			filtered = pipe.deno(method='boxcar', \
							arg=self.config.BOXCAR_RADIUS)
			filtered = pipe.deno(method='gaussian', \
							arg=self.config.GAUS_BLUR_SIG)

			self.file_widget.update_file_list()

		if ops.det_box.isChecked():

			pipe.detect_fit(detect_video=True,
							fit_psf_video=True)

		if ops.trak_box.isChecked():

			pipe.filter_and_track()

class CQSettings(QSettings):

	"""
	Global application settings (superset of Config object w/ extra params
	for the GUI)
	"""

	def __init__(self):

		super(QSettings, self).__init__()
		self.clear()

		# """
		# ~~~~~~~~~Pipeline Settings~~~~~~~~~~~~~~
		# """

		self.setValue('Meta/input_path', '')
		self.setValue('Meta/output_path', str(Path.home()) + '/Desktop/temp/')
		self.setValue('Meta/root_name', '')
		self.setValue('Meta/start_frame', 0)
		self.setValue('Meta/end_frame', 100)
		self.setValue('Meta/check_frame', 0)
		self.setValue('Meta/frame_rate', 3.33)
		self.setValue('Meta/pixel_size', 108.4)

		self.setValue('Regi/ref_ind_num', 50)
		self.setValue('Regi/sig_mask', 3)
		self.setValue('Regi/thres_rel', .1)
		self.setValue('Regi/poly_deg', 2)
		self.setValue('Regi/rotation_multiplier', 1)
		self.setValue('Regi/translation_multiplier', 1)

		self.setValue('Segm/min_size', 1000)
		self.setValue('Segm/mask_sig', 3)
		self.setValue('Segm/mask_thres', 0.09)

		self.setValue('Deno/boxcar_radius', 10)
		self.setValue('Deno/gaus_blur_sig', 0.5)

		self.setValue('Det/blob_threshold', 0.05)
		self.setValue('Det/blob_min_sigma', 2)
		self.setValue('Det/blob_max_sigma', 4)
		self.setValue('Det/blob_num_sigma', 5)
		self.setValue('Det/pk_thres_rel', 0.15)
		self.setValue('Det/plot_r', False)
		self.setValue('Det/r_to_sigraw', 1)

		self.setValue('Trak/do_filter', False)
		self.setValue('Trak/max_dist_err', 10)
		self.setValue('Trak/max_sig_to_sigraw', 10)
		self.setValue('Trak/max_delta_area', 10)
		self.setValue('Trak/traj_length_thres', 10)

		self.setValue('Trak/search_range', 2)
		self.setValue('Trak/memory', 3)
		self.setValue('Trak/divide_num', 5)

		# """
		# ~~~~~~~~~Plot Settings~~~~~~~~~~~~~~
		# """

		self.setValue('Plot/plot_type', 'Line')
		self.setValue('Plot/p_value', False)
		self.setValue('Plot/pearson_corr', False)
		self.setValue('Plot/spearman_cor', False)
