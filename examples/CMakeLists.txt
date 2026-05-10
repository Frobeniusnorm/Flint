cmake_minimum_required(VERSION 3.9)
project(flint_examples VERSION 0.0.1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../") # to include FindOpenCL.cmake

include(GNUInstallDirs)
find_package(OpenCL REQUIRED)

if(NOT TARGET flint)
	message(FATAL_ERROR "examples must be built from the Flint root project (target 'flint' not found)")
endif()

set(FLINT_BUILD_INCLUDE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/include")
set(FLINT_BUILD_INCLUDE_DIR "${FLINT_BUILD_INCLUDE_ROOT}/flint")
file(MAKE_DIRECTORY "${FLINT_BUILD_INCLUDE_DIR}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../flint.h" "${FLINT_BUILD_INCLUDE_DIR}/flint.h" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../flint.hpp" "${FLINT_BUILD_INCLUDE_DIR}/flint.hpp" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../flint_helper.hpp" "${FLINT_BUILD_INCLUDE_DIR}/flint_helper.hpp" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../flint_1.hpp" "${FLINT_BUILD_INCLUDE_DIR}/flint_1.hpp" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../flint_n.hpp" "${FLINT_BUILD_INCLUDE_DIR}/flint_n.hpp" COPYONLY)

add_executable(gauss gauss/gauss.cpp)
set_property(TARGET gauss PROPERTY CXX_STANDARD 20)
target_link_libraries(gauss PRIVATE flint OpenCL::OpenCL)
target_include_directories(gauss PRIVATE "${FLINT_BUILD_INCLUDE_ROOT}" "${CMAKE_BINARY_DIR}")
add_dependencies(gauss flint)
add_custom_command(TARGET gauss 
                   POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gauss> ${PROJECT_SOURCE_DIR}/gauss/gauss)

if(${DEEP_LEARNING})
	find_package(Protobuf REQUIRED)
	find_package(absl REQUIRED)

	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../src/dl/model.hpp" "${FLINT_BUILD_INCLUDE_DIR}/model.hpp" COPYONLY)
	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../src/dl/layers.hpp" "${FLINT_BUILD_INCLUDE_DIR}/layers.hpp" COPYONLY)
	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../src/dl/trainer.hpp" "${FLINT_BUILD_INCLUDE_DIR}/trainer.hpp" COPYONLY)

	add_executable(resnet resnet/resnet.cpp)
	set_property(TARGET resnet PROPERTY CXX_STANDARD 20)
	target_link_libraries(resnet PRIVATE flint ${Protobuf_LIBRARIES} OpenCL::OpenCL absl::base absl::strings)
	target_include_directories(resnet PRIVATE "${FLINT_BUILD_INCLUDE_ROOT}" "${CMAKE_BINARY_DIR}")
	target_link_options(resnet PRIVATE "-Wl,--copy-dt-needed-entries")
	add_dependencies(resnet flint)
	add_custom_command(TARGET resnet
					   POST_BUILD
					   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:resnet> ${PROJECT_SOURCE_DIR}/resnet/resnet)

	add_executable(mnist_train mnist/mnist_train.cpp)
	set_property(TARGET mnist_train PROPERTY CXX_STANDARD 20)
	target_link_libraries(mnist_train PRIVATE flint ${Protobuf_LIBRARIES} OpenCL::OpenCL absl::base absl::strings)
	target_include_directories(mnist_train PRIVATE "${FLINT_BUILD_INCLUDE_ROOT}" "${CMAKE_BINARY_DIR}")
	target_link_options(mnist_train PRIVATE "-Wl,--copy-dt-needed-entries")
	add_dependencies(mnist_train flint)
	add_custom_command(TARGET mnist_train
					   POST_BUILD
					   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mnist_train> ${PROJECT_SOURCE_DIR}/mnist/mnist_train)
else()
	message(STATUS "DEEP_LEARNING=OFF: skipping mnist and resnet examples")
endif()
