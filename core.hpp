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
static constexpr void checkTypes(){
    static_assert(std::is_same<T, int>() || std::is_same<T, float>(), "Only integer and floating-point Tensor types are allowed");
}
template <typename T>
struct Tensor<T, 1> : public BaseTensor<T>{
    friend class GPUBackend;
protected:
    int vaultID = -1;
    std::vector<T> data;
    size_t sizes;
public:
    typedef std::vector<T> type;
    Tensor(){
        checkTypes<T>();
    }
    Tensor(std::vector<T> data):data(data){
        checkTypes<T>();
        sizes = data.size();
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
    std::string to_string() const{
        std::string res = "[";
        bool first = true;
        for(const T& el : data){
          if(!first) res += ", ";
          else first = false;
          res += std::to_string(el);
        }
        res += "]";
        return res;
    }
    operator std::string() const{
      return to_string();
    }
    std::vector<T>& flatData(){
        return data;
    }
    std::vector<T> operator*() const{
        return data;
    }
    T operator[](unsigned index) const{
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
    Tensor<T, 1> operator+(Tensor<T, 1>& other) const{
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
    size_t calculateSize(std::vector<T>& vec) const{
        return vec.size();
    }
    template<typename E>
    size_t calculateSize(std::vector<std::vector<E>>& vec) const{
        if(vec.size() == 0) log(ERROR, "A tensor does not allow empty vectors!");
        return vec.size() * calculateSize(vec[0]);
    }
    int assignData(std::vector<T>& vec, int index){
        for(int i = 0; i < vec.size(); i++)
            data[index + i] = vec[i];
        return vec.size();
    }
    template<typename E>
    int assignData(std::vector<std::vector<E>>& vec, int index){
        int curridx = index;
        for(std::vector<E>& r : vec){
            curridx += assignData(r, curridx);
        }
        return curridx - index;
    }
    template<typename E>
    void assignRekVector(std::vector<E>& vec){
        assignSizes(vec);
        //calculate overall size
        size_t dataSize = calculateSize(vec);
        data = std::vector<T>(dataSize);
        assignData(vec, 0);
    }
    int fillVector(std::vector<T>& vec, int index, int depth) const{
        vec.resize(sizes[depth]);
        for(int i = 0; i < vec.size(); i++)
            vec[i] = data[index + i];
        return vec.size();
    }
    template<typename E>
    int fillVector(std::vector<std::vector<E>>& vec, int index, int depth) const{
        vec.resize(sizes[depth]);
        int curridx = index;
        for(std::vector<E>& r : vec){
            curridx += fillVector(r, curridx, depth+1);
        }
        return curridx - index;
    }
    std::string toString(int dimension = n-1, int* start = nullptr) const{
      int startbf = 0;
      if(!start) start = &startbf;

      std::string res = "[";
      bool first = true;
      size_t size = sizes[dimension];
      if(dimension == 0){
        for(int i = 0; i < size; i++){
          if(!first) res += ", ";
          else first = false;
          res += std::to_string(data[*start + i]);
        }
        *start += size;
      }else{
        for(int i = 0; i < size; i++){
          if(!first) res += ", ";
          else first = false;
          res += toString(dimension - 1, start);
        }
      }
      res += "]";
      return res;
    }
public:
    typedef typename Tensor<T, n-1>::type rektype;
    typedef std::vector<rektype> type;
    Tensor(){
        checkTypes<T>();
    }
    Tensor(std::vector<T> data, std::array<size_t, n> sizes):data(data), sizes(sizes){
        checkTypes<T>();
        updateTensor(*this);
    }
    Tensor(std::vector<rektype> data){
        checkTypes<T>();
        assignRekVector(data);
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
    Tensor<T, n>& operator=(const Tensor<T, n>& other){
        data = other.data;
        sizes = other.sizes;
        return *this;
    }
    Tensor<T, n>& operator=(Tensor<T, n>&& other){
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
    std::vector<rektype> operator*() const{
        std::vector<rektype> foo = std::vector<rektype>();
        fillVector(foo, 0, 0);
        return foo;
    }
    Tensor<T, n-1> operator[](unsigned index) const{
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
        assignRekVector(asgn);
        updateTensor(*this);
    }
    void operator+=(Tensor<T, n>& other){
        add(*this, other, *this);
    }
    Tensor<T, n> operator+(Tensor<T, n>& other) const{
        Tensor<T, n> result;
        result.sizes = sizes;
        add(*this, other, result);
        return result;
    }
    std::string to_string() const{
        return toString();
    }
    operator std::string() const{
      return to_string();
    }
};

#endif
