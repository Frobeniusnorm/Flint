# Example Implementations with Flint
## Gauss
A simple Gauss Blur that loads an image to a tensor, blurs it and saves it to the current directory.
Just compile the file by linking Flint and OpenCl against it e.g.
`g++ gauss.cpp -o gauss -L.. -lflint -lOpenCL`

## Mnist
TODO

## Matmul in C
An example showcasing how to use the library in simple C
`gcc matmul.c -o matmul -lOpenCL -lflint -lstdc++ -lm`
