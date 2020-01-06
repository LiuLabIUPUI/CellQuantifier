from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from cellquantifier.util.gui.plot_settings import *

import matplotlib.pyplot as plt
import pandas as pd

class PlotWidget(QWidget):

	def __init__(self, parent):

		super(QWidget, self).__init__(parent)

		self.layout = QHBoxLayout()
		self.figure = plt.figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.button = QPushButton('Plot')
		self.button.clicked.connect(self.plot)

		# """
		# ~~~~~~~~~~~Graph Widget~~~~~~~~~~~~~~
		# """


		self.graph = QWidget()
		self.graph.layout = QVBoxLayout()
		self.graph.layout.addWidget(self.toolbar)
		self.graph.layout.addWidget(self.canvas)
		self.graph.layout.addWidget(self.button)
		self.graph.setLayout(self.graph.layout)

		# """
		# ~~~~~~~~~~~Graph Options Widget~~~~~~~~~~~~~~
		# """

		self.graph_options = QWidget()
		self.graph_options.layout = QFormLayout()

		self.plt_type_combo_label = QLabel('Plot Type')
		self.plt_type_combo = QComboBox(self.graph_options)

		self.filename = QLineEdit()
		self.filename_label = QLabel('File Name')

		self.addButton = QPushButton('...')
		self.addButton.clicked.connect(self.openFileNameDialog)


		self.graph_options.layout.addRow(self.plt_type_combo)
		self.graph_options.layout.addRow(self.filename, self.addButton)


		self.plt_type_combo.addItems(["Histogram", "MSD", "Box", "Scatter", "Violin", "General"])
		self.plt_type_combo.activated[int].connect(self.onActivatedIndex)

		self.stack = QStackedWidget(self)
		self.hist_options = HistogramOptions(self)
		self.msd_options = MSDOptions(self)
		self.box_plot_options = BoxPlotOptions(self)

		self.stack.addWidget(self.hist_options)
		self.stack.addWidget(self.msd_options)
		self.stack.addWidget(self.box_plot_options)

		self.graph_options.layout.addRow(self.stack)
		self.graph_options.setLayout(self.graph_options.layout)

		# """
		# ~~~~~~~~~~~Build~~~~~~~~~~~~~~
		# """

		splitter = QSplitter()
		splitter.addWidget(self.graph_options)
		splitter.addWidget(self.graph)
		splitter.setStretchFactor(1,50)
		self.layout.addWidget(splitter)
		self.setLayout(self.layout)
		self.show()

	def openFileNameDialog(self):

		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)

		if fileName:

			self.filename.setText(fileName)

	def build_axis(self):

		ax = self.figure.add_subplot(111)

		x = self.hist_options.x.text()
		y = self.hist_options.y.text()

		ax.set_xlabel(x)
		ax.set_ylabel(y)

		return ax

	def plot(self, ax=None, diagnostic=False):

		plot_type = self.plt_type_combo.currentText()

		if diagnostic:

			self.canvas.draw()

		else:

			file = self.filename.text()
			df = pd.read_csv(file)

			self.figure.clear()
			ax = self.build_axis()

			if plot_type == "Histogram":

				col = self.hist_options.col_name.text()
				nbins = int(self.hist_options.nbins.text())
				color = self.hist_options.color.text()
				norm = self.hist_options.normalize.isChecked()
				ax.hist(df[col], bins=nbins, color=color, density=norm)
				self.canvas.draw()

			elif plot_type == "MSD":

				self.canvas.draw()

	@pyqtSlot(int)
	def onActivatedIndex(self, index):
		self.stack.setCurrentIndex(index)
