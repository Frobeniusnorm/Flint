# Example Implementations with Flint
## Gauss
A simple Gauss Blur that loads an image to a tensor, blurs it and saves it to the current directoy.
Just compile the file by linking Flint and OpenCl against it e.g.
`g++ gauss.cpp -o gauss -L.. -lflint -lOpenCL`
