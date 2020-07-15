import pandas as pd
import math
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from skimage.io import imread
from cellquantifier.plot import *
from datetime import datetime
from pathlib import Path

# """
# ~~~~~~~~~~~IO Widgets~~~~~~~~~~~~~~
# """

class FileWidget(QWidget):

	def __init__(self, parent):

		super(QWidget, self).__init__()

		self.parent = parent
		self.path = str(Path.home()) + '/Desktop/temp'
		self.layout = QVBoxLayout()
		self.file_list = FileTreeWidget(self.parent)
		self.update_file_list()

		self.clr_btn = QPushButton('Clear')
		self.clr_btn.clicked.connect(self.clear_dir)
		self.add_btn = QPushButton('Add to DataView')
		self.add_btn.clicked.connect(self.open_file)

		self.layout.addWidget(self.file_list)
		self.layout.addWidget(self.add_btn)
		self.layout.addWidget(self.clr_btn)
		self.setLayout(self.layout)

	def update_file_list(self):
		self.file_list.clear()
		files = []
		for name in sorted(os.listdir(self.path)):
			row = QtWidgets.QTreeWidgetItem()
			ext = name.split('.')[-1]
			row.setText(0, name)
			row.setText(1, ext)
			files.append(row)
		self.file_list.addTopLevelItems(files)

	def clear_dir(self):

		flist = [f for f in os.listdir(self.path)]
		for f in flist:
			os.remove(os.path.join(self.path, f))
		self.file_list.clear()

	def open_file(self):

		items = self.file_list.selectedItems()

		for item in items:
			filename = item.text(0)
			filetype = item.text(1)

			if filetype == 'csv':
				self.open_csv(filename)
			elif filetype == 'tif':
				self.animate(filename)

	def animate(self, filename):

		"""
		If file is an image, animate it
		"""

		self.parent.plot_widget.figure.clear()
		im = imread(self.path + '/' + filename)
		ax = self.parent.plot_widget.figure.add_subplot(111)
		pl = ax.imshow(im[0], cmap='gray')

		def animate_func(i):
			pl.set_array(im[i])
			return pl

		anim = Animation(self.parent.plot_widget.figure,
			animate_func,
			mini=0,
			maxi=len(im)-1,
			interval=50)

	def open_csv(self, filename):

		"""
		If file is a csv, load it into the DataView
		"""

		df = pd.read_csv(self.path + '/' + filename)
		self.parent.data_view.update_df(df)


class FileTreeWidget(QTreeWidget):

	def __init__(self, parent):

		super(QWidget, self).__init__()

		self.parent = parent
		self.setHeaderLabels(['Filename', 'Type'])
		self.itemClicked.connect(self.on_tree_clicked)

	@QtCore.pyqtSlot(QtWidgets.QTreeWidgetItem, int)
	def on_tree_clicked(self):

		items = self.selectedItems()
		for item in items:
			filename = item.text(0)
			filetype = item.text(1)


# """
# ~~~~~~~~~~~Pipeline Widgets~~~~~~~~~~~~~~
# """

class PipelineWidget(QWidget):

	def __init__(self, parent):

		super(QWidget, self).__init__()

		self.parent = parent
		settings = parent.settings
		self.layout = QVBoxLayout()

		# """
		# ~~~~~~~~~~~Initialize Widgets~~~~~~~~~~~~~
		# """

		self.pipe = QWidget()
		self.pipe.layout = QHBoxLayout()

		self.tabs = QTabWidget()
		self.tabs.setTabPosition(QTabWidget.East)

		self.chkbxs = PipelineCheckBoxes()
		self.pipe.layout.addWidget(self.chkbxs)
		self.pipe.layout.addWidget(self.tabs)
		self.pipe.setLayout(self.pipe.layout)
		self.layout.addWidget(self.pipe)

		self.btn_grp = QButtonGroup()
		self.run_btn = QPushButton()
		self.stop_btn = QPushButton()
		self.run_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
		self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
		self.btns = QWidget()
		self.btns.layout = QHBoxLayout()

		self.btns.layout.addWidget(self.run_btn)
		self.btns.layout.addWidget(self.stop_btn)
		self.btns.setLayout(self.btns.layout)
		self.layout.addWidget(self.btns)
		self.run_btn.clicked.connect(self.parent.pipeline_control)

		# """
		# ~~~~~~~~~~~Add tabs~~~~~~~~~~~~~
		# """
		stages = ['Meta', 'Regi', 'Segm', 'Deno', \
				  'Det', 'Fitt', 'Trak']

		for stage in stages:

			settings = parent.settings
			settings.beginGroup(stage)
			sub = settings.allKeys()
			sub = [stage + '/' + s for s in sub]
			settings.endGroup()

			self.tab = PipelineTab(sub, settings)
			self.tabs.addTab(self.tab, stage)

		self.setLayout(self.layout)

class PipelineTab(QWidget):

	def __init__(self, sub, settings):

		super(QWidget, self).__init__()

		self.layout = QGridLayout()
		self.table = PipelineTable(sub, settings)

		self.table.cellChanged.connect(self.update_settings)

		self.layout.addWidget(self.table)
		self.setLayout(self.layout)

	def update_settings(self, row, column):

		print('Cell Changed!')

class PipelineTable(QTableWidget):

	def __init__(self, sub, settings):

		super(QTableWidget, self).__init__(len(sub),1)

		for i, setting in enumerate(sub):
			value = settings.value(setting)
			self.setItem(i, 0, QTableWidgetItem(str(value)))

		self.vert_labels = sub
		self.horiz_labels = ['','']
		self.setVerticalHeaderLabels(self.vert_labels)
		self.setHorizontalHeaderLabels(self.horiz_labels)


class PipelineCheckBoxes(QWidget):

	def __init__(self):

		super(QWidget, self).__init__()

		self.layout = QVBoxLayout()

		self.group = QGroupBox("Pipeline")
		self.group.layout = QVBoxLayout()

		self.group.regi_box = QCheckBox('Regi')
		self.group.segm_box = QCheckBox('Segm')
		self.group.deno_box = QCheckBox('Deno')
		self.group.det_box = QCheckBox('Det')
		self.group.trak_box = QCheckBox('Trak')

		self.group.layout.addWidget(self.group.regi_box)
		self.group.layout.addWidget(self.group.segm_box)
		self.group.layout.addWidget(self.group.deno_box)
		self.group.layout.addWidget(self.group.det_box)
		self.group.layout.addWidget(self.group.trak_box)

		self.group.setLayout(self.group.layout)

		self.layout.addWidget(self.group)
		self.setLayout(self.layout)

# """
# ~~~~~~~~~~~Plot Widgets~~~~~~~~~~~~~~
# """

class PlotWidget(QWidget):

	def __init__(self, parent):

		super(QWidget, self).__init__()

		self.parent = parent
		self.layout = QHBoxLayout()
		self.figure = plt.figure()
		self.figure.set_facecolor('lightgray')
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.clear_btn = QPushButton('Clear Figure')

		# """
		# ~~~~~~~~~~~Graph Widget~~~~~~~~~~~~~~
		# """

		self.layout = QVBoxLayout()
		self.layout.addWidget(self.toolbar)
		self.layout.addWidget(self.canvas)
		self.layout.addWidget(self.clear_btn)
		self.layout.setContentsMargins(0, 0, 0, 100)

		self.clear_btn.clicked.connect(self.clear_figure)
		self.setLayout(self.layout)

	def plot(self, *args, **kwargs):

		plot_type = self.parent.settings.value('plot_type')
		ax = self.figure.add_subplot(111)
		data = kwargs['data']
		col = kwargs['col']

		# """
		# ~~~~~~~~~~~Build Dialog~~~~~~~~~~~~~~
		# """

		dialog = QDialog()
		dialog.layout = QVBoxLayout()
		combo = QComboBox()
		combo.addItems([None])
		combo.addItems(data.columns)
		btn = QPushButton("Ok")
		dialog.layout.addWidget(combo)
		dialog.layout.addWidget(btn)
		dialog.setLayout(dialog.layout)

		if plot_type == 'Line':

			ax.plot(data[col])

		if plot_type == 'MSD':

			dialog.exec_()

			cat_col = combo.currentText()

			add_mean_msd(ax,
						 data,
						 cat_col,
						 pixel_size=108.4,
						 frame_rate=3.3,
						 divide_num=5
						 )

		elif plot_type == 'Strip':

			dialog.exec_()

			cat_col = combo.currentText()

			add_strip_plot(ax,
						   data,
						   col,
						   cat_col
						   )

		elif plot_type == 'Histogram':

			dialog.exec_()

			cat_col = combo.currentText()

			add_hist(ax,
					 data,
					 col,
					 cat_col
					)

		elif plot_type == 'Box':

			dialog.exec_()

			cat_col = combo.currentText()

			add_box_plot(ax,
					 data,
					 col,
					 cat_col
					)

		self.canvas.draw()


	def clear_figure(self):
		self.figure.clf()
		self.canvas.draw()

class PlotPrefsDialog(QDialog):

	def __init__(self, parent):

		super(QDialog, self).__init__()

		self.parent = parent
		self.setWindowTitle('Plot Preferences')

		grid = QGridLayout()
		self.apply_btn = QPushButton('Apply')

		grid.addWidget(self.create_plt_grp(), 0, 0)
		grid.addWidget(self.create_stat_grp(), 0, 1)

		self.apply_btn.clicked.connect(self.update_prefs)
		grid.addWidget(self.apply_btn, 1, 1)
		self.setLayout(grid)

	def update_prefs(self):

		text = self.plot_type.currentText()
		self.parent.settings.setValue('plot_type', text)

	def create_stat_grp(self):

		groupBox = QGroupBox("Statistics")

		checkbox1 = QCheckBox("P-Value")
		checkbox2 = QCheckBox("Pearson Correlation")
		checkbox3 = QCheckBox("Spearman Correlation")
		checkbox1.setChecked(True)

		layout = QVBoxLayout()
		layout.addWidget(checkbox1)
		layout.addWidget(checkbox2)
		layout.addWidget(checkbox3)
		groupBox.setLayout(layout)

		return groupBox

	def create_plt_grp(self):

		groupBox = QGroupBox("Plot Type")

		self.plot_type = QComboBox()
		self.plot_type.addItems(['Line','Histogram', 'Box', 'Strip', 'MSD'])

		layout = QVBoxLayout()
		layout.addWidget(self.plot_type)

		groupBox.setLayout(layout)

		return groupBox


# """
# ~~~~~~~~~~~DataView Widgets~~~~~~~~~~~~~~
# """

class DataView(QWidget):

	def __init__(self, parent):

		super(QWidget, self).__init__()

		self.parent = parent
		self.layout = QVBoxLayout()

	def update_df(self, df):

		# """
		# ~~~~~~~~~~~Clear Current Widget~~~~~~~~~~~~~
		# """

		try:
			self.table_view.deleteLater()
			self.filt_widget.deleteLater()
		except:
			pass

		# """
		# ~~~~~~~~~~~Data Frame~~~~~~~~~~~~~
		# """

		self.table_view = TableView(self)
		self.table_model = PandasModel(df)
		self.table_view.setModel(self.table_model)
		self.layout.addWidget(self.table_view)

		# """
		# ~~~~~~~~~~~Filters~~~~~~~~~~~~~
		# """

		columns = df.columns.to_numpy()
		self.filt_widget = FilterWidget(columns, self)
		self.layout.addWidget(self.filt_widget)

		self.setLayout(self.layout)


class FilterWidget(QWidget):

	def __init__(self, columns, parent):

		super(QWidget,self).__init__()

		num_filters = 5
		self.layout = QGridLayout()
		self.parent = parent
		self.filters = \
				pd.DataFrame({'column': [None for i in range(num_filters)],\
							  'enabled': [False for i in range(num_filters)],\
							  'filter': [None for i in range(num_filters)]})

		self.btn_grp = QButtonGroup()
		self.btn_grp.buttonClicked.connect(self.toggle_filter)

		self.combos = [QComboBox() for i in range(num_filters)]
		self.ledits = [QLineEdit() for i in range(num_filters)]

		for i in range(num_filters):

			btn = QPushButton('Enable')
			self.btn_grp.addButton(btn)
			self.btn_grp.setId(btn, i)

			self.combos[i].addItems(columns)

			self.layout.addWidget(btn, i, 0)
			self.layout.addWidget(self.combos[i], i, 1)
			self.layout.addWidget(self.ledits[i], i, 2)

		self.setLayout(self.layout)

	def toggle_filter(self, btn):

		"""
		Frontend filter switch
		"""

		index = self.btn_grp.id(btn)
		enabled = self.filters.iloc[index]['enabled']

		if enabled:
			self.btn_grp.button(index).setText('Enable')
			self.update_filt_df(index,
								enabled=False,
								filter=None,
								column=None)

		else:
			_col = self.combos[index].currentText()
			_filt = self.ledits[index].text()
			self.update_filt_df(index,
								enabled=True,
								filter=_filt,
								column=_col)

			self.btn_grp.button(index).setText('Disable')

		self.update_df()

	def update_filt_df(self, index, enabled, filter, column):

		"""
		Backend filter switch
		"""

		self.filters.set_value(index, 'enabled', enabled)
		self.filters.set_value(index, 'filter', filter)
		self.filters.set_value(index, 'column', column)

	def update_df(self):

		"""
		Apply or remove a filter from table model
		"""

		filt_df = self.filters[self.filters['enabled']]
		df = self.parent.table_model._data

		for index, row in filt_df.iterrows():

			col = row['column'].strip()
			cond = row['filter']
			cond = cond.replace('_', "df.loc[df['%s']" % col) + ']'
			df = eval(cond)

		self.parent.table_view.setModel(PandasModel(df))


class TableView(QTableView):

	"""
	Class to implement the abstract table model
	"""

	def __init__(self, parent):

		super(QTableView, self).__init__()
		self.parent = parent
		self.horizontalHeader().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
		self.horizontalHeader().customContextMenuRequested.connect(self.selectColumnMenu)
		self.verticalHeader().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
		self.verticalHeader().customContextMenuRequested.connect(self.selectRowMenu)

	def selectColumnMenu(self, event):

		""" A right click on a column name allows the info to be displayed in the graphView """
		self.column = self.columnAt(event.x())
		self.col_name = self.parent.table_model._data.columns[self.column]
		self.menu = QMenu(self)
		self.plotit = QAction("Plot", self)
		self.menu.addAction(self.plotit)
		self.menu.popup(QtGui.QCursor.pos())
		self.plotit.triggered.connect(self.plot_data)

	def selectRowMenu(self, event):

		""" A right click on a column name allows the info to be displayed in the graphView """
		self.col_name = None
		self.row = self.rowAt(event.y())
		self.menu = QMenu(self)
		self.plotit = QAction("Plot", self)
		self.menu.addAction(self.plotit)
		self.menu.popup(QtGui.QCursor.pos())
		self.plotit.triggered.connect(self.plot_data)

	def plot_data(self):

		centralwidget = self.parent.parent
		centralwidget.plot_widget.plot(data = self.parent.table_model._data,
									   col=self.col_name)


class PandasModel(QtCore.QAbstractTableModel):

	"""
	Class to populate a table view with a pandas dataframe
	"""

	def __init__(self, data, parent=None):
		super(PandasModel, self).__init__()
		self._data = data

	def rowCount(self, parent=None):
		return self._data.shape[0]

	def columnCount(self, parent=None):
		return self._data.shape[1]

	def data(self, index, role=QtCore.Qt.DisplayRole):
		if index.isValid():
			if role == QtCore.Qt.DisplayRole:
				return str(self._data.iloc[index.row(), index.column()])
		return None

	def headerData(self, col, orientation, role):
		if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
			return self._data.columns[col]
		return None
