#include <iostream>
#include <vector>
#include "flint.hpp"

int main(){
    Tensor<float, 1> a;
    a = {0,1, 2.5f,3,4.7f};
    Tensor<float, 1> b;
    b = {4,3.2f,2.5f,11.2f,7};
    a += b;
    for(float f: *a){
        std::cout << f << " ";
    }
    std::cout<< std::endl;
    a += a;
    for(int i = 0; i < 5; i++){
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}