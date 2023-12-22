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

add_executable(mnist_train mnist/mnist_train.cpp)
set_property(TARGET mnist_train PROPERTY CXX_STANDARD 20)
target_link_libraries(mnist_train -lflint)
target_link_libraries(mnist_train OpenCL::OpenCL)
add_custom_command(TARGET mnist_train
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mnist_train> ${PROJECT_SOURCE_DIR}/mnist/mnist_train)

add_executable(mnist_test mnist/mnist_test.cpp)
set_property(TARGET mnist_test PROPERTY CXX_STANDARD 20)
target_link_libraries(mnist_test -lflint)
target_link_libraries(mnist_test OpenCL::OpenCL)
add_custom_command(TARGET mnist_test
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mnist_test> ${PROJECT_SOURCE_DIR}/mnist/mnist_test)
