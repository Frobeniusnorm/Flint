#ifndef CORE_HPP
#define CORE_HPP
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <array>
#include <cstddef>
#include <vector>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include "logger.hpp"


class GPUBackend;
namespace Flint{
    inline bool initialized = false;
    inline bool useGPU;
    inline GPUBackend* gpuBackend;
    void init();
};

template <typename T>
struct BaseTensor{
    BaseTensor(){
        if(!Flint::initialized){
            Flint::init();
        }
    }
};

template <typename T, unsigned int dimensions = 1>
struct Tensor;

class GPUBackend{
private:
    cl::Device device;
    cl::Context* context;
    cl::CommandQueue queue;
    std::vector<cl::Buffer*> dataVault;
    cl::Program buildProgram(std::string code);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> *func_add_int, *func_add_long, *func_add_float, *func_add_double;
public:
    GPUBackend();
    ~GPUBackend();
    template <typename T, unsigned int n>
    void updateTensor(Tensor<T, n>& a);
    void deleteTensor(int vaultID);
    template <typename T, unsigned int n>
    void add(Tensor<T, n>& a, Tensor<T, n>& b, Tensor<T, n>& dest);
};

template <typename T, unsigned int n>
static void updateTensor(Tensor<T, n>& a){
    if(Flint::useGPU){
        Flint::gpuBackend->updateTensor(a);
    }
}
template <typename T, unsigned int n>
static void add(Tensor<T, n>& a, Tensor<T, n>& b, Tensor<T, n>& dest){
    if(Flint::useGPU){
        Flint::gpuBackend->add(a,b,dest);
    }
}

template <typename T>
struct Tensor<T, 1> : public BaseTensor<T>{
    friend class GPUBackend; 
protected:
    int vaultID = -1;
    typedef std::vector<T> type;
    std::vector<T> data;
    size_t sizes;
public:
    Tensor(){
        static_assert(std::is_same<T, int>() || std::is_same<T, float>());
    }
    Tensor(std::vector<T> data):data(data){
        static_assert(std::is_same<T, int>() || std::is_same<T, float>());
        updateTensor(*this);
    }
    ~Tensor(){
        if(vaultID >= 0 && Flint::gpuBackend) Flint::gpuBackend->deleteTensor(vaultID); 
        vaultID = -1;
    }
    Tensor(const Tensor<T, 1>& other){
        data = other.data;
        sizes = other.sizes; //no memory management, since vaultID is -1, so we implicitly copy the data on the gpu
    }
    Tensor(Tensor<T, 1>&& other){
        vaultID = other.vaultID;
        data = other.data;
        sizes = other.sizes;
        other.vaultID = -1;
    }
    Tensor<T, 1>& operator=(const Tensor<T, 1>& other){
        data = other.data;
        sizes = other.sizes;
        return *this;
    }
    Tensor<T, 1>& operator=(Tensor<T, 1>&& other){
        if(vaultID >= 0) Flint::gpuBackend->deleteTensor(vaultID);
        vaultID = other.vaultID;
        data = other.data;
        sizes = other.sizes;
        other.vaultID = -1;
        return *this;
    }
    std::vector<T>& flatData(){
        return data;
    }
    std::vector<T> operator*(){
        return data;
    }
    T operator[](unsigned index){
        return data[index];
    }
    void operator=(std::vector<T> asgn){
        data = asgn;
        sizes = asgn.size();
        updateTensor(*this);
    }
    void operator+=(Tensor<T, 1>& other){
        add(*this, other, *this);
    }
    Tensor<T, 1> operator+(Tensor<T, 1>& other){
        Tensor<T, 1> result;
        result.sizes = sizes;
        add(*this, other, result);
        return result;
    }
};

template <typename T, unsigned int n>
struct Tensor : public BaseTensor<T>{
    friend class GPUBackend; 
protected:
    int vaultID = -1;
    typedef typename Tensor<T, n-1>::type rektype;
    std::array<size_t, n> sizes;
    std::vector<T> data;
private:
    int assignSizes(std::vector<T>& vec){
        sizes[sizes.size() - 1] = vec.size();
        return 1;
    }
    template<typename E>
    int assignSizes(std::vector<std::vector<E>>& vec){
        size_t subv = vec[0].size();
        for(int i = 1; i < vec.size(); i++){
            if(subv != vec[i].size())
                log(ERROR, "incompatible Tensor sizes!");
        }
        subv = assignSizes(vec[0]) + 1;
        sizes[sizes.size() - subv] = vec.size();
        return subv;
    }

public:
    Tensor(){}
    Tensor(std::vector<T> data):data(data){
        updateTensor(*this);
    }
    ~Tensor(){
        if(vaultID >= 0 && Flint::gpuBackend) Flint::gpuBackend->deleteTensor(vaultID); 
        vaultID = -1;
    }
    Tensor(const Tensor<T, n>& other){
        data = other.data;
        sizes = other.sizes; //no memory management, since vaultID is -1, so we implicitly copy the data on the gpu
    }
    Tensor(Tensor<T, n>&& other){
        vaultID = other.vaultID;
        data = other.data;
        sizes = other.sizes;
        other.vaultID = -1;
    }
    Tensor<T, n>& operator=(const Tensor<T, 1>& other){
        data = other.data;
        sizes = other.sizes;
        return *this;
    }
    Tensor<T, n>& operator=(Tensor<T, 1>&& other){
        if(vaultID >= 0) Flint::gpuBackend->deleteTensor(vaultID);
        vaultID = other.vaultID;
        data = other.data;
        sizes = other.sizes;
        other.vaultID = -1;
        return *this;
    }
    std::vector<T>& flatData(){
        return data;
    }
    std::vector<rektype> operator*(){
        return std::vector<rektype>(); //TODO
    }
    Tensor<T, n-1> operator[](unsigned index){
        Tensor<T, n-1> foo;
        size_t sum = 0;
        for(int i = 0; i < n - 1; i++){
            foo.sizes[i] = sizes[i + 1];
            sum += sizes[i + 1];
        }
        foo.data = std::vector<T>(data.data[index], sum);
        return foo;
    }
    void operator=(std::vector<rektype> asgn){
        assignSizes(asgn);
        data = asgn;
        updateTensor(*this);
    }
    void operator+=(Tensor<T, n>& other){
        add(*this, other, *this);
    }
    Tensor<T, n> operator+(Tensor<T, n>& other){
        Tensor<T, n> result;
        result.sizes = sizes;
        add(*this, other, result);
        return result;
    }
};

#endif