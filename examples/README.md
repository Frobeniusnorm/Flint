# Example Implementations with Flint

## Gauss
Shows how convolve can be used to implement a (very naive and inefficient) version of the gauss filter.
One implementation is written with the C and the other with the C++ interface. Loads the logo in the root of the project, blurs it and saves the blurred version to a new file.

## MNIST
Showcases the Deep Learning library of the C++ interface by using a sequential model with Convolution and Pooling Layers for learning on the MNIST dataset.
The dataset can be downloaded with the `download_data.sh` script. If training has been succesfull the model should be able to categorize 28x28 images of number characters (0-9). The deep learning library has a small and incomplete visualization front end that is currently used.
Once the `NetworkMetricReporter` is set up, the training status can be seen by opening the `dl/visualization/index.html` in a browser (preferably firefox).

## Building
With CMake:

```
mkdir build
cd build
cmake ..
make
```
For MNIST don't forget to download the data either manually or with the download script.

## Usage
Gauss just needs the logo file to be 2 folders above and can be run with `./gauss`, MNIST consists of a training binary that trains
the MNIST model and saves it and a testing binary that needs the model file and a image as command line input and then executes the model on the image.
After building is done you can run it with:

```
cd ../mnist
bash download_data.sh
./mnist_train
./mnist_test mnist_model.flint <path to image>
```
