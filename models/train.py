"""
Train the model.
"""

from mnist import MNIST

mndata = MNIST("samples")

images, labels = mndata.load_training()


