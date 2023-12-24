### Linker can't find OpenCL (cannot link against -lOpenCL)

The usual `-lOpenCL` linker flag will only search for libraries named `libOpenCL.so`.
Some distributions of linux may only ship a library named `libOpenCL.so.1.0.0`.
You can either specify the linker flag explicity like: `-lOpenCL:1.0.0` or follow the following steps to alleviate the
problem:

1. Find out where the openCL lib is located: `sudo find / -type f -name "libOpenCL.so*"`
2. This should return something like `/usr/lib64/libOpenCL.so.1.0.0`
3. Create a symlink: `cd /usr/lib64 && sudo ln -s libOpenCL.so.1.0.0 libOpenCL.so`

### Compiler cannot find flint

If you include flint as usual (`#include <flint/flint.h>`) the compiler sometimes will not find it in the common include
directories.
It might be necessary to include the header files explicitly.
First make sure the header files are installed globally by running `sudo make install` in the build directory.
Then make sure your compiler includes `/usr/local/include/flint` or include it explicitly using
`g++ -I /usr/local/include/flint/ ...`
