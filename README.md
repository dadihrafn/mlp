# Multilayer Perceptron in C

This repo contains an MLP for recognizing handwritten digits from the MNIST dataset. Written in C with no external libraries.

- **train.c**: Trains the network on the MNIST training data, and outputs it's parameters in a file.
- **run.c**: Loads the trained network, runs inference on the test data in MNIST, and prints the percentage of test images it categorized correctly.

The training and test data can be found at http://yann.lecun.com/exdb/mnist/.

To train the model, compile `train.c`, and run:
```
a.out train-images-idx3-ubyte train-labels-idx1-ubyte 30
```
- `train-images-idx3-ubyte` is the path to the training images from MNIST.
- `train-labels-idx1-ubyte` is the path to the training labels from MNIST.
- `30` is the number of epochs, which should be somewhere between 10-100.

To run the model, compile `run.c`, and run:
```
a.out parameters.bin t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
```
- `parameters.bin` is the name of the file `train.c` outputs, and contains the parameters of the network.
- `t10k-images-idx3-ubyte` is the path to the test images from MNIST.
- `t10k-labels-idx1-ubyte` is the path to the test labels from MNIST.
