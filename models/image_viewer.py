"""
View the images training/testing images.
"""

# Standard libraries
import os
import sys
from dataclasses import dataclass
from typing import Any

# Third-party libraries
import idx2numpy
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QComboBox,
    QWidget, QSlider, QVBoxLayout,
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

@dataclass
class Data:

    images: Any
    labels: Any
    file: str
    currentIdx: int

class Window(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.setWindowTitle("Image Viewer")

        self.centralWidget = QWidget()
        layout = QVBoxLayout()
        self.selectedData = QComboBox()
        self.selectedData.addItem("Test Data")
        self.selectedData.addItem("Train Data")
        self.selectedData.currentIndexChanged.connect(self._changeData)
        self.imageWindow = pg.GraphicsLayoutWidget()
        self.figure = self.imageWindow.addPlot(title = "")
        self.image = pg.ImageItem()
        self.figure.addItem(self.image)
        self.figure.invertY(True)
        self.imageSlider = QSlider(Qt.Horizontal)
        self.imageSlider.setMinimum(0)
        self.imageSlider.valueChanged.connect(self._changeFrame)
        layout.addWidget(self.selectedData)
        layout.addWidget(self.imageWindow)
        layout.addWidget(self.imageSlider)
        self.centralWidget.setLayout(layout)
        self.setCentralWidget(self.centralWidget)

        self.data = None
        self._loadImages()

    def _loadImages(self, path = "data"):

        trainImagesFile = os.path.join(path, "train-images-idx3-ubyte")
        trainLabelsFile = os.path.join(path, "train-labels-idx1-ubyte")
        testImagesFile = os.path.join(path, "t10k-images-idx3-ubyte")
        testLabelsFile = os.path.join(path, "t10k-labels-idx1-ubyte")

        testImages = idx2numpy.convert_from_file(testImagesFile)
        testLabels = idx2numpy.convert_from_file(testLabelsFile)
        trainImages = idx2numpy.convert_from_file(trainImagesFile)
        trainLabels = idx2numpy.convert_from_file(trainLabelsFile)

        self.testData = Data(testImages, testLabels, testImagesFile, 0)
        self.trainData = Data(trainImages, trainLabels, trainImagesFile, 0)
        self.data = self.testData

        self.imageSlider.setMaximum(self.data.images.shape[0] - 1)
        self.imageSlider.setTickPosition(self.data.currentIdx)
        self._changeFrame()

    def _changeFrame(self):

        idx = self.imageSlider.value()
        self.data.currentIdx = idx
        db = os.path.basename(self.data.file)
        image = self.data.images[idx]
        label = self.data.labels[idx]
        self.image.setImage(image.T)
        title = f"Database: {db}  Label: {label}"
        self.figure.setTitle(title)
    
    def _changeData(self):

        if self.selectedData.currentText() == "Test Data":
            self.data = self.testData
        elif self.selectedData.currentText() == "Train Data":
            self.data = self.trainData
        
        self.data.currentIdx = 0
        self.imageSlider.setMaximum(self.data.images.shape[0] - 1)
        self.imageSlider.setValue(self.data.currentIdx)
        self._changeFrame()

def main():
    app = QApplication(sys.argv)
    gui = Window()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
