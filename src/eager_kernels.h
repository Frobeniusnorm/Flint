inline const char *eager_kernels = R"(
// Generate Kernels by GPU Backend
__kernel void FADD00(__global int* R, __global const int* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const int* P1
, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
int v1 = P1[index % total_size_P1];
int v2 = P0[index % total_size_P0];
int v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD01(__global long* R, __global const long* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const int* 
P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
int v1 = P1[index % total_size_P1];
long v2 = P0[index % total_size_P0];
long v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD02(__global float* R, __global const float* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const int
* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
int v1 = P1[index % total_size_P1];
float v2 = P0[index % total_size_P0];
float v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD03(__global double* R, __global const double* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const i
nt* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
int v1 = P1[index % total_size_P1];
double v2 = P0[index % total_size_P0];
double v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD10(__global long* R, __global const int* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const long* 
P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
long v1 = P1[index % total_size_P1];
int v2 = P0[index % total_size_P0];
long v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD11(__global long* R, __global const long* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const long*
 P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
long v1 = P1[index % total_size_P1];
long v2 = P0[index % total_size_P0];
long v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD12(__global float* R, __global const float* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const lon
g* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
long v1 = P1[index % total_size_P1];
float v2 = P0[index % total_size_P0];
float v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD13(__global double* R, __global const double* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const l
ong* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
long v1 = P1[index % total_size_P1];
double v2 = P0[index % total_size_P0];
double v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD20(__global float* R, __global const int* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const float
* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
float v1 = P1[index % total_size_P1];
int v2 = P0[index % total_size_P0];
float v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD21(__global float* R, __global const long* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const floa
t* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
float v1 = P1[index % total_size_P1];
long v2 = P0[index % total_size_P0];
float v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD22(__global float* R, __global const float* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const flo
at* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
float v1 = P1[index % total_size_P1];
float v2 = P0[index % total_size_P0];
float v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD23(__global double* R, __global const double* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const f
loat* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
float v1 = P1[index % total_size_P1];
double v2 = P0[index % total_size_P0];
double v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD30(__global double* R, __global const int* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const doub
le* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
double v1 = P1[index % total_size_P1];
int v2 = P0[index % total_size_P0];
double v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD31(__global double* R, __global const long* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const dou
ble* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
double v1 = P1[index % total_size_P1];
long v2 = P0[index % total_size_P0];
double v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD32(__global double* R, __global const float* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const do
uble* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
double v1 = P1[index % total_size_P1];
float v2 = P0[index % total_size_P0];
double v0 = v1 + v2;
R[index] = v0;
}

__kernel void FADD33(__global double* R, __global const double* P0, const size_t total_size_P0, __global const size_t* shape_P0, const int dimensions_P0, __global const d
ouble* P1, const size_t total_size_P1, __global const size_t* shape_P1, const int dimensions_P1){
int index = get_global_id(0);
double v1 = P1[index % total_size_P1];
double v2 = P0[index % total_size_P0];
double v0 = v1 + v2;
R[index] = v0;
}
)";
