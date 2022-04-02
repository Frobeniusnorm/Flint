#include "core.hpp"
#include "logger.hpp"
#include <CL/cl.h>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>


static const std::string kernel_add = R"(
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
)";
inline cl::Program GPUBackend::buildProgram(std::string code){
    cl::Program::Sources sources;
    sources.push_back({code.c_str(), code.length()});
    cl::Program program(*context, sources);
    if(program.build({device})!=CL_SUCCESS){
        log(ERROR, "Could not build a program! Build log: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
        exit(-1);
    }
    return program;
}
inline GPUBackend::GPUBackend(){
    using std::vector;
    Flint::useGPU = true;
    vector<cl::Platform> platforms;
     
    cl::Platform::get(&platforms);
    if(platforms.empty()){
        log(WARNING, "No OpenCL platform found! Make sure you have configured your GPU right!");
        log(VERBOSE, "Defaulting to CPU implementation.");
        Flint::useGPU = false;
        return;
    }
    //TODO: allow configuration
   
    cl::Platform& platform = platforms[0];
    
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.empty()){
        log(WARNING, "No OpenCL device for platform found! Make sure you have configured your GPU right!");
        log(VERBOSE, "Defaulting to CPU implementation.");
        Flint::useGPU = false;
        exit(-1);
    }
    log(VERBOSE, "Chose a Device");
    device = devices[0];
    log(INFO, "Chosen Device: " + device.getInfo<CL_DEVICE_NAME>() + " on platform: " + platform.getInfo<CL_PLATFORM_NAME>());
    context = new cl::Context({device});
    queue = cl::CommandQueue(*context, device);
    cl::Program add = buildProgram(kernel_add);
    func_add_int = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(add, "int_add"));
    func_add_float = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(add, "float_add"));
    func_add_long = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(add, "long_add"));
    func_add_double = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(add, "double_add"));
}

template <typename T, unsigned int n>
inline void GPUBackend::add(Tensor<T, n>& a, Tensor<T, n>& b, Tensor<T, n>& dest){
    std::vector<T>& fdA = a.flatData();
    if(a.vaultID < 0) updateTensor(a);
    if(b.vaultID < 0) updateTensor(b);
    cl::Buffer* bA = dataVault[a.vaultID];
    cl::Buffer* bB = dataVault[b.vaultID];
    if(dest.vaultID < 0){
        dest.vaultID = dataVault.size();
        dataVault.push_back(nullptr);
    }
    if(dest.sizes != a.sizes){
        if(dataVault[dest.vaultID])
            delete dataVault[dest.vaultID];
        dataVault[dest.vaultID] = new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(T)*fdA.size());
    }
    cl::Buffer* bC = dataVault[dest.vaultID];
    const cl::EnqueueArgs ea = cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(fdA.size()), cl::NullRange);
    if(std::is_same<T, int>())
        (*func_add_int)(ea, *bA, *bB, *bC).wait();
    else if(std::is_same<T, float>())
        (*func_add_float)(ea, *bA, *bB, *bC).wait();
    else if(std::is_same<T, long>())
        (*func_add_long)(ea, *bA, *bB, *bC).wait();
     else if(std::is_same<T, double>())
        (*func_add_long)(ea, *bA, *bB, *bC).wait();
    //TODO maybe it's smarter to read memory lazy on demand and mark cpu data as outdated here
    std::vector<T> C(fdA.size());
    queue.enqueueReadBuffer(*bC, true, 0, sizeof(T)*fdA.size(), &C[0]);
    dest.sizes = a.sizes;
    dest.data = C;
}
inline GPUBackend::~GPUBackend(){
    delete func_add_int;
    delete func_add_float;
}
void GPUBackend::deleteTensor(int vaultID){
    if(vaultID >= 0){
        delete dataVault[vaultID];
    }
}
template <typename T, unsigned int n>
void GPUBackend::updateTensor(Tensor<T, n>& a){
    //check if a vault id needs to be assigned
    if(a.vaultID < 0){
        a.vaultID = dataVault.size();
        dataVault.push_back(nullptr);
    }else{
        delete (dataVault[a.vaultID]);
        dataVault[a.vaultID] = nullptr;
    }
    //create new buffer, because buffer may not be resized
    std::vector<T>& fdA = a.flatData();
    dataVault[a.vaultID] = new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(T)*fdA.size());
    queue.enqueueWriteBuffer(*(dataVault[a.vaultID]), true, 0, sizeof(T)*fdA.size(), fdA.data());
}

inline void Flint::init(){
    log(VERBOSE, "Initializing Flint");
    Flint::initialized = true;
    Flint::gpuBackend = new GPUBackend();
    if(!Flint::gpuBackend){
        //init cpu backend
    }
}