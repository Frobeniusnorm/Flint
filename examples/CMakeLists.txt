cmake_minimum_required(VERSION 3.9)
project(flint_examples VERSION 0.0.1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../") # to include FindOpenCL.cmake

include(GNUInstallDirs)
find_package(OpenCL REQUIRED)
find_package(PkgConfig)
pkg_search_module(Protobuf REQUIRED protobuf)
find_package(absl REQUIRED)

add_executable(gauss gauss/gauss.cpp)
set_property(TARGET gauss PROPERTY CXX_STANDARD 20)
target_link_libraries(gauss -lflint)
target_link_libraries(gauss ${Protobuf_LIBRARIES})
target_link_libraries(gauss OpenCL::OpenCL)
add_custom_command(TARGET gauss 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gauss> ${PROJECT_SOURCE_DIR}/gauss/gauss)

include_directories(${Protobuf_INCLUDE_DIRS})
add_executable(resnet resnet/resnet.cpp)
set_property(TARGET resnet PROPERTY CXX_STANDARD 20)
target_link_libraries(resnet -lflint)
target_link_libraries(resnet ${Protobuf_LIBRARIES})
target_link_libraries(resnet OpenCL::OpenCL)
target_link_libraries(resnet absl::base)
target_link_libraries(resnet absl::strings)
add_link_options("-Wl,--copy-dt-needed-entries")
add_custom_command(TARGET resnet
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:resnet> ${PROJECT_SOURCE_DIR}/resnet/resnet)
