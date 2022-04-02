void kernel int_add(global const int* A, global const int* B, global int* C){
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)]; 
}
void kernel float_add(global const float* A, global const float* B, global float* C){
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)]; 
}
void kernel double_add(global const double* A, global const double* B, global double* C){
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)]; 
}
void kernel long_add(global const long* A, global const long* B, global long* C){
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)]; 
}