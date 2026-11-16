# Example Implementations with Flint

## ResNet
Loads the ResNet50 ONNX model (downloaded with `download_data.sh`) and classifies an image: `./resnet resnet50-v1-12.onnx tiger.jpg`.

## Gauss
Shows how convolve can be used to implement a (very naive and inefficient) version of the gauss filter.
One implementation is written with the C and the other with the C++ interface. Loads the logo in the root of the project, blurs it and saves the blurred version to a new file.

## MNIST
Showcases the Deep Learning library of the C++ interface by using a sequential model with Convolution and Pooling Layers for learning on the MNIST dataset.
The dataset can be downloaded with the `download.sh` script. `mnist_train` prints the time and loss of every batch and stores the trained model to `model.onnx`.
Options: `--epochs n`, `--batches n` (stop after n batches), `--batch-size n`, `--debug` (logs every kernel with its execution time).
For a live view of the training replace the reporter by a `NetworkMetricReporter` and open `visualization/index.html` in a browser (preferably firefox).

## Building
With CMake:

```
mkdir build
cd build
cmake .. -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make
```
For MNIST don't forget to download the data either manually or with the download script.

## Usage
Gauss just needs the logo file to be 2 folders above and can be run with `./gauss`.
For MNIST after building is done you can run it with:

```
cd ../examples/mnist
bash download.sh
./mnist_train
```
