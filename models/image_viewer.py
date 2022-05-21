"""
View the images training/testing images.
"""

import os
import sys

import idx2numpy
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow
)
import pyqtgraph as pg

class Window(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)

        self.imageWidget = pg.ImageView()
        self.setCentralWidget(self.imageWidget)

        self._loadImages()

        self.imageWidget.setImage(self.testImages[0])

    def _loadImages(self, path = "data"):

        self.files = {
            "TRAIN_IMAGES" : os.path.join(path, "train-images-idx3-ubyte"),
            "TRAIN_LABELS" : os.path.join(path, "train-labels-idx1-ubyte"),
            "TEST_IMAGES" : os.path.join(path, "t10k-images-idx3-ubyte"),
            "TEST_LABELS" : os.path.join(path, "t10k-labels-idx1-ubyte")
        }

        self.testImages = idx2numpy.convert_from_file(self.files["TEST_IMAGES"])
        self.testLabels = idx2numpy.convert_from_file(self.files["TEST_LABELS"])
        self.trainImages = idx2numpy.convert_from_file(self.files["TEST_IMAGES"])
        self.trainLabels = idx2numpy.convert_from_file(self.files["TEST_IMAGES"])

def main():
    app = QApplication(sys.argv)
    gui = Window()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
