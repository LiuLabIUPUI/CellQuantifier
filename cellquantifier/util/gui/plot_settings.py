import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

import random

class HistogramOptions(QWidget):

    def __init__(self, parent):

        super(QWidget, self).__init__(parent)

        self.layout = QFormLayout(self)

        self.col_name_label = QLabel('Column Name')
        self.col_name = QLineEdit()

        self.nbins_label = QLabel()
        self.nbins_label.setText('Bins')
        self.nbins = QLineEdit()
        self.nbins.setFixedWidth(50)

        self.normalize_label = QLabel()
        self.normalize_label.setText('Normalize')
        self.normalize = QCheckBox()

        self.color_label = QLabel()
        self.color_label.setText('Color')
        self.color = QLineEdit()
        self.color.setFixedWidth(50)

        self.x_label = QLabel()
        self.x_label.setText('X')
        self.x = QLineEdit()
        self.x.setFixedWidth(50)

        self.y_label = QLabel()
        self.y_label.setText('Y')
        self.y = QLineEdit()
        self.y.setFixedWidth(50)

        self.layout.addRow(self.col_name_label, self.col_name)
        self.layout.addRow(self.nbins_label, self.nbins)
        self.layout.addRow(self.color_label, self.color)
        self.layout.addRow(self.x_label, self.x)
        self.layout.addRow(self.y_label, self.y)
        self.layout.addRow(self.normalize_label, self.normalize)

        self.setLayout(self.layout)


class MSDOptions(QWidget):

    def __init__(self, parent):

        super(QWidget, self).__init__(parent)

        self.layout = QFormLayout(self)

        self.orientation_label = QLabel()
        self.orientation_label.setText('Normalize')
        self.orientation = QCheckBox()

        self.color_label = QLabel()
        self.color_label.setText('Color')
        self.color = QLineEdit()
        self.color.setFixedWidth(50)

        self.x_label = QLabel()
        self.x_label.setText('X')
        self.x = QLineEdit()
        self.x.setFixedWidth(50)

        self.y_label = QLabel()
        self.y_label.setText('Y')
        self.y = QLineEdit()
        self.y.setFixedWidth(50)

        self.layout.addRow(self.color_label, self.color)
        self.layout.addRow(self.x_label, self.x)
        self.layout.addRow(self.y_label, self.y)

        self.setLayout(self.layout)

class BoxPlotOptions(QWidget):

    def __init__(self, parent):

        super(QWidget, self).__init__(parent)

        self.layout = QFormLayout(self)

        self.orientation_label = QLabel()
        self.orientation_label.setText('Horizontal')
        self.orientation = QCheckBox()

        self.layout.addRow(self.orientation_label, self.orientation)
        self.setLayout(self.layout)
