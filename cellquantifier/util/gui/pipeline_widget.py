from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, Qt
from datetime import datetime

import csv

class PipelineWidget(QWidget):

	def __init__(self, parent):

		super(QWidget, self).__init__(parent)

		stages = ['Regi', 'Segm', 'Deno', 'Det', 'Fitt', 'Trak', 'Filt', 'Sort']
		self.load_defaults()

		# """
		# ~~~~~~~~~~~Initialize Widgets~~~~~~~~~~~~~
		# """

		self.layout = QHBoxLayout()
		self.tabs = QTabWidget()
		self.meta = MetaWidget(self)
		self.checkboxes = PipelineCheckBoxes(self)

		# """
		# ~~~~~~~~~~~Add default params~~~~~~~~~~~~~
		# """

		for stage in stages:

			sub = {key:value for key,value in self.settings.items() if stage in key}
			self.tab = QWidget(self.tabs)
			self.tab_layout = QGridLayout(self.tab)
			self.table = QTableWidget(len(sub), 1)

			for index, (key,value) in enumerate(sub.items()):
				param = str(value)
				self.table.setItem(index, 0, QTableWidgetItem(param))

			self.tab_layout.addWidget(self.table, 0, 0, 1, 1)
			self.table.vert_labels = list(sub.keys())
			self.table.setVerticalHeaderLabels(self.table.vert_labels)
			self.tabs.addTab(self.tab, stage)

		# """
		# ~~~~~~~~~~~Add to layout~~~~~~~~~~~~~
		# """

		self.layout.addWidget(self.meta)
		self.layout.addWidget(self.checkboxes)
		self.layout.addWidget(self.tabs)

		self.setLayout(self.layout)

	def load_defaults(self):

		# """
		# ~~~~~~~~~~~Load Default Configuration~~~~~~~~~~~~~
		# """

		reader = csv.reader(open('cellquantifier/util/default_config.csv', 'r'))
		self.settings = {}
		for row in reader:
		   k, v = row
		   self.settings[k] = v

class PipelineCheckBoxes(QWidget):

	def __init__(self,parent):

		super(QWidget, self).__init__(parent)

		self.layout = QVBoxLayout()

		self.regi_box = QCheckBox('Regi')
		self.segm_box = QCheckBox('Segm')
		self.deno_box = QCheckBox('Deno')
		self.det_box = QCheckBox('Det')
		self.trak_box = QCheckBox('Trak')
		self.filt_box = QCheckBox('Filt')
		self.sort_box = QCheckBox('Sort')

		self.layout.addWidget(self.regi_box)
		self.layout.addWidget(self.segm_box)
		self.layout.addWidget(self.deno_box)
		self.layout.addWidget(self.det_box)
		self.layout.addWidget(self.trak_box)
		self.layout.addWidget(self.filt_box)
		self.layout.addWidget(self.sort_box)

		self.setLayout(self.layout)

class MetaWidget(QWidget):

	def __init__(self,parent):

		super(QWidget, self).__init__(parent)

		self.layout = QFormLayout()

		# """
		# ~~~~~~~~~~~Add meta params~~~~~~~~~~~~~
		# """

		sub = {key:value for key,value in parent.settings.items() if 'Meta' in key}
		self.table = QTableWidget(len(sub),1)

		for index, (key,value) in enumerate(sub.items()):
			param = str(value)
			self.table.setItem(index, 0, QTableWidgetItem(param))

		self.vert_labels = list(sub.keys())
		self.table.setVerticalHeaderLabels(self.vert_labels)

		# """
		# ~~~~~~~~~~~Output path~~~~~~~~~~~~~
		# """

		self.output_path_button = QPushButton('Add Output Path')
		self.output_path_button.clicked.connect(self.selectFile)

		self.layout.addRow(self.table)
		self.layout.addRow(self.output_path_button)

		self.setLayout(self.layout)

	def selectFile(self):
		dir = QFileDialog.getExistingDirectory(self, 'Select Directory')
		row_count = self.table.rowCount()
		self.table.setItem(row_count-1,0, QTableWidgetItem(str(dir)))
