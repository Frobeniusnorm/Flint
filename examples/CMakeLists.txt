cmake_minimum_required(VERSION 3.9)
project(flint_examples VERSION 0.0.1)
find_package(OpenCL REQUIRED)

add_executable(gauss gauss/gauss.cpp)
set_property(TARGET gauss PROPERTY CXX_STANDARD 20)
target_link_libraries(gauss -lflint)
target_link_libraries(gauss OpenCL::OpenCL)
add_custom_command(TARGET gauss 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gauss> ${PROJECT_SOURCE_DIR}/gauss/gauss)

add_executable(mnist mnist/mnist.cpp)
set_property(TARGET mnist PROPERTY CXX_STANDARD 20)
target_link_libraries(mnist -lflint)
target_link_libraries(mnist OpenCL::OpenCL)
add_custom_command(TARGET mnist
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mnist> ${PROJECT_SOURCE_DIR}/mnist/mnist)
