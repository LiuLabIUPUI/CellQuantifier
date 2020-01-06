from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, Qt

from cellquantifier.util.config import *
from cellquantifier.util.pipeline import Pipeline
from cellquantifier.plot import Animation

from .pipeline_widget import PipelineWidget
from .plot_widget import PlotWidget
from .channel_dock import ChannelWidget

import matplotlib.pyplot as plt
import math
import os


class Ui_MainWindow(object):

	def setupUi(self, MainWindow):

		# """
		# ~~~~~~~~~~~Main Window~~~~~~~~~~~~~~
		# """

		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1920, 1080)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.centralwidget.layout = QtWidgets.QVBoxLayout(self.centralwidget)

		# """
		# ~~~~~~~~~~~Pipeline Widget~~~~~~~~~~~~~~
		# """

		self.pipeline = PipelineWidget(self.centralwidget)
		self.centralwidget.layout.addWidget(self.pipeline)

		self.run_button = QPushButton('Run')
		self.centralwidget.layout.addWidget(self.run_button)
		self.run_button.clicked.connect(self.pipeline_control)

		# """
		# ~~~~~~~~~~~Plot Widget~~~~~~~~~~~~~~
		# """

		self.plot_widget = PlotWidget(self.centralwidget)
		self.centralwidget.layout.addWidget(self.plot_widget)

		MainWindow.setCentralWidget(self.centralwidget)

		# """
		# ~~~~~~~~~~Dock Widget~~~~~~~~~~~~~~
		# """

		self.dock = QtWidgets.QDockWidget(MainWindow)
		self.dockContents = ChannelWidget()
		self.dock.setWidget(self.dockContents)
		MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dock)


		# """
		# ~~~~~~~~~~Misc~~~~~~~~~~~~~~
		# """

		self.actionAction = QtWidgets.QAction(MainWindow)
		self.actionSub_menu = QtWidgets.QAction(MainWindow)
		self.actionAction_C = QtWidgets.QAction(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def pull_config(self):

		def sanitize(str):

			if str.isdigit():
				return int(str)

			if str.lower() == 'true' or str.lower() == 'false':
				return bool(str)

			try:
				return float(str)
			except ValueError:
				pass

			return str

		# """
		# ~~~~~~~~~~~Pull metadata params~~~~~~~~~~~~~~
		# """
		self.config = {}
		meta_table = self.pipeline.meta.table

		rows = meta_table.rowCount()
		for row in range(rows):
			key = self.pipeline.meta.vert_labels[row]
			value = sanitize(meta_table.item(row,0).text())
			self.config[key] = value

		self.config['Meta output_path'] += '/'
		self.config['Meta input_path'] = None

		# """
		# ~~~~~~~~~~~Pull plot params~~~~~~~~~~~~~~
		# """
		pipe_tabs = self.pipeline.tabs
		num_tabs = pipe_tabs.count()

		for i in range(num_tabs):
			tab = pipe_tabs.widget(i)
			table = tab.children()[1]
			rows = table.rowCount()

			for row in range(rows):
				key = table.vert_labels[row]
				value = sanitize(table.item(row,0).text())
				self.config[key] = value

	def animate_det(self, det_plt_array, fit_plt_array):

		ax1 = self.plot_widget.figure.add_subplot(121)
		ax2 = self.plot_widget.figure.add_subplot(122)

		#set plots
		pl1 = ax1.imshow(det_plt_array[0])
		pl2 = ax2.imshow(fit_plt_array[0])

		def animate_func(i):

			pl1.set_array(det_plt_array[i])
			pl2.set_array(fit_plt_array[i])

			return pl1,pl2

		anim = Animation(self.plot_widget.figure,
						  animate_func,
						  mini=0,
						  maxi=len(det_plt_array)-1,
						  interval=200)


	def animate_deno(self, deno_array):

		ax1 = self.plot_widget.figure.add_subplot(111)

		#set plots
		pl1 = ax1.imshow(deno_array[0], cmap='gray')

		def animate_func(i):

			pl1.set_array(deno_array[i])

			return pl1

		anim = Animation(self.plot_widget.figure,
						  animate_func,
						  mini=0,
						  maxi=len(deno_array)-1,
						  interval=200)

	def animate_segm(self, masks_array_3d, dist_array_3d, px_array_3d):

		ax1 = self.plot_widget.figure.add_subplot(131)
		ax2 = self.plot_widget.figure.add_subplot(132)
		ax3 = self.plot_widget.figure.add_subplot(133)

		#set plots
		pl1 = ax1.imshow(masks_array_3d[0], cmap='gray')
		pl2 = ax2.imshow(dist_array_3d[0], cmap='gray')
		pl3 = ax3.imshow(px_array_3d[0], cmap='gray')

		def animate_func(i):

			pl1.set_array(masks_array_3d[i])
			pl2.set_array(dist_array_3d[i])
			pl3.set_array(px_array_3d[i])

			return pl1,pl2,pl3

		anim = Animation(self.plot_widget.figure,
						  animate_func,
						  mini=0,
						  maxi=len(masks_array_3d)-1,
						  interval=200)

	def get_active_channel(self):

		files = []
		tab_widget = self.dockContents.tabs
		nchannels = tab_widget.count()

		for i in range(nchannels):

			current_widget = tab_widget.widget(i)
			if current_widget.activate_box.isChecked():
				file = current_widget.fileEdit.text()
				files.append(file)

		return files[0]


	def pipeline_control(self):

		# """
		# ~~~~~~~~~~Get IO Params~~~~~~~~~~~~~~
		# """


		path = self.get_active_channel()
		self.pull_config()

		self.config['Meta input_path'] = path
		self.config['Meta output_path'] += '/'
		self.config = Config(self.config)

		pipe = Pipeline(self.config)
		pipe.load(self.config)

		# """
		# ~~~~~~~~~Clear the figure~~~~~~~~~~~~~~
		# """

		self.plot_widget.figure.clear()

		# """
		# ~~~~~~~~~Run the pipeline~~~~~~~~~~~~~~
		# """

		if self.pipeline.checkboxes.regi_box.isChecked():
			pipe.register(self.config)

		if self.pipeline.checkboxes.segm_box.isChecked():

			masks_array_3d, dist_array_3d, px_array_3d = pipe.segmentation(\
														 self.config)

			self.animate_segm(masks_array_3d, dist_array_3d, px_array_3d)

		if self.pipeline.checkboxes.deno_box.isChecked():

			filtered = pipe.deno(self.config, method='boxcar', \
							arg=self.config.BOXCAR_RADIUS)
			filtered = pipe.deno(self.config, method='gaussian', \
							arg=self.config.GAUS_BLUR_SIG)

			self.animate_deno(filtered)

		if self.pipeline.checkboxes.det_box.isChecked():

			det_plt_array, fit_plt_array = pipe.detect_fit(self.config)
			self.animate_det(det_plt_array, fit_plt_array)

		if self.pipeline.checkboxes.trak_box.isChecked():
			pipe.filter_and_track(self.config)

		if self.pipeline.checkboxes.sort_box.isChecked():
			pipe.sort_and_plot(self.config)
