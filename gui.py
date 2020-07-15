from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from cellquantifier.util.gui import view
from cellquantifier.util.gui.breeze import breeze_resources

import logging
import sys


def main():
	"""
	Application entry point
	"""
	logging.basicConfig(level=logging.DEBUG)
	# create the application and the main window
	app = QApplication(sys.argv)
	#app.setStyle(QtWidgets.QStyleFactory.create("fusion"))

	# setup ui
	ui = view.Ui_MainWindow()
	window = QMainWindow()
	window.setWindowTitle("")
	ui.setup_ui(window)

	# setup stylesheet
	file = QFile(":/dark.qss")
	file.open(QFile.ReadOnly | QFile.Text)
	stream = QTextStream(file)
	app.setStyleSheet(stream.readAll())

	# auto quit after 2s when testing on travis-ci
	if "--travis" in sys.argv:
		QTimer.singleShot(2000, app.exit)

	# run
	window.show()
	app.exec_()


if __name__ == "__main__":
	main()
