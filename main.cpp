#include <iostream>
#include <vector>
#include "flint.hpp"

int main(){
    Tensor<float, 2> a({{0,1}, {2.5f,3}, {4.7f, 5}});
    Tensor<float, 2> b({{4,3.2f}, {2.5f,11.2f}, {7, 1}});
    a += b;
    std::cout << "[";
    for(std::vector<float> f: *a){
        std::cout << "[";
        for(float d : f){
            std::cout << d << " ";
        }
        std::cout << "] ";
    }
    std::cout<< "]" << std::endl;
}