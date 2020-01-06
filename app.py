from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QFile, QTextStream
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
    app = QtWidgets.QApplication(sys.argv)
    #app.setStyle(QtWidgets.QStyleFactory.create("fusion"))
    window = QtWidgets.QMainWindow()

    # setup ui
    ui = view.Ui_MainWindow()
    ui.setupUi(window)
    window.setWindowTitle("CellQuantifier")

    # setup stylesheet
    file = QFile(":/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())

    # auto quit after 2s when testing on travis-ci
    if "--travis" in sys.argv:
        QtCore.QTimer.singleShot(2000, app.exit)

    # run
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
