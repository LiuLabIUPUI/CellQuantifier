from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, Qt

import math
import os


class ChannelTab(QWidget):

	def __init__(self):

		super(QWidget, self).__init__()

		self.layout = QFormLayout()

		self.fileEdit = QLineEdit()
		self.fileLabel = QLabel('File')
		self.addButton = QPushButton('Choose File')
		self.combo = QComboBox()
		self.combo.addItems(["Particle", "Object"])
		self.combo_label = QLabel('Type')
		self.aliasLabel = QLabel('Alias')
		self.aliasEdit = QLineEdit()
		self.activate_label = QLabel('Activate')
		self.activate_box = QCheckBox()

		self.addButton.clicked.connect(self.openFileNameDialog)

		self.layout.addRow(self.fileLabel, self.fileEdit)
		self.layout.addRow(self.combo_label, self.combo)
		self.layout.addRow(self.aliasLabel, self.aliasEdit)
		self.layout.addWidget(self.addButton)
		self.layout.addRow(self.activate_label, self.activate_box)

		self.setLayout(self.layout)

	def openFileNameDialog(self):

		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)

		if fileName:

			self.fileEdit.setText(fileName)


	def removeaFile(self):

		iterator = QTreeWidgetItemIterator(self.fileTree)

		while iterator.value():
			item = iterator.value()
			if item.isSelected(): #check value here
				item.setText(0, "") #set text here
				item.setText(1, "") #set text here
				self.fileTree.takeTopLevelItem(self.fileTree.indexOfTopLevelItem(item))
			iterator+=1

	def getFileList(self):

		iterator = QTreeWidgetItemIterator(self.fileTree)

		file_list = []
		while iterator.value():
			item = iterator.value()
			file_list.append(str(item.text(0)))

			iterator+=1

		return file_list

class ChannelWidget(QWidget):

	def __init__(self):

		super(QWidget, self).__init__()

		self.layout = QVBoxLayout()

		self.addButton = QPushButton('Add Channel')

		self.tabs = QTabWidget()
		self.tabs.setTabPosition(QTabWidget.West)
		self.tabs.setTabsClosable(True)

		self.addButton.clicked.connect(self.add_channel)
		self.tabs.tabCloseRequested.connect(self.closeTab)


		self.layout.addWidget(self.addButton)
		self.layout.addWidget(self.tabs)

		self.setLayout(self.layout)

		self.add_channel()


	def add_channel(self):

		index = self.tabs.count()
		self.tabs.insertTab(index, ChannelTab(), "Channel")
		self.tabs.setCurrentIndex(index)


	@pyqtSlot(int)
	def closeTab(self, index):

		current_widget = self.tabs.widget(index)
		current_widget.deleteLater()
