<div align="center">
<img src="https://github.com/Frobeniusnorm/Flint/blob/main/flint.png" width="300">
</div>

# Flint DL #
Implementation of often used deep learning algorithm and concepts.
Still a work in progress with no first official version.

## Models ##
- Sequential Model: the output of one layer is passed on as the input for the next

## Layers ##
- Connected: A fully connected layer (matrix multiplication)
- Convolution: A convolution layer with stride, padding and bias support
- Dropout: Sets random activations to 0, reducing overfitting
- Flatten: Flattens the input array to 2 dimensions (batch and channels)
- Relu: Rectified Linear Unit activation function, sets negative values to 0
- SoftMax: Activation function for multiclass classification

## Losses ##
- Cross Entropy Loss: Measures performance for probability estimations in classification tasks

## Optimizers ##
- Adam: first-order gradient-based optimizer for stochastic objective functions based on adaptive estimates of lower-order momentums

