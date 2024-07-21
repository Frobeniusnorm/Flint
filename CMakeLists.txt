cmake_minimum_required(VERSION 3.9)
project(flint VERSION 0.0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}") # to include FindOpenCL.cmake

option(BUILD_TEST "Enable building unit tests" ON)
option(BUILD_GRADIENT_TEST "Enable building gradient tests" ON)
option(BUILD_BENCHMARK "Enable building benchmark program" ON)
option(C_COMPATIBLE "Enable C-compatibility (no exceptions but erno)" OFF)
option(DEEP_LEARNING "Enable building and installing the header-only c++ deep learning library" ON)

include(GNUInstallDirs)

find_package(OpenCL REQUIRED)

FILE(GLOB SRCFILES src/*.cpp src/backend_cpu/*.cpp src/backend_ocl/*.cpp src/operations/*.cpp)
SET(HEADERFILES "flint.h;flint.hpp;flint_helper.hpp;flint_1.hpp;flint_n.hpp")
SET(INCLFILES)
if(${DEEP_LEARNING})
	find_package(Protobuf REQUIRED)
	FILE(GLOB_RECURSE SRCFILESDL src/dl/*.cpp)
	list(APPEND SRCFILES ${SRCFILESDL})
	protobuf_generate_cpp(PROTO_SRCFILES INCLFILES onnx.proto3)
	list(APPEND SRCFILES build/onnx.proto3.pb.cc)
	list(APPEND HEADERFILES "src/dl/model.hpp;build/onnx.proto3.pb.h;src/dl/layers.hpp")
	find_package(absl REQUIRED)
endif()
add_library(flint STATIC ${SRCFILES} ${INCLFILES})

set_property(TARGET flint PROPERTY CXX_STANDARD 20)
target_include_directories(flint PRIVATE .)
target_include_directories(flint PRIVATE build)
target_link_libraries(flint OpenCL::OpenCL)
if(${DEEP_LEARNING})
	target_link_libraries(flint ${Protobuf_LIBRARIES})
	target_link_libraries(flint absl::base)
	target_link_libraries(flint absl::strings)
	add_link_options("-Wl,--copy-dt-needed-entries")
endif()
set_target_properties(flint PROPERTIES VERSION ${PROJECT_VERSION} PUBLIC_HEADER "${HEADERFILES}")

if(C_COMPATIBLE)
  add_compile_definitions("C_COMPATIBILITY")
endif()

install(TARGETS flint 
    EXPORT flintTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flint
    )


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    flintConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT flintTargets
        FILE flintTargets.cmake
        NAMESPACE flint::
        DESTINATION lib/cmake/flint)
configure_file(flintConfig.cmake @ONLY)
install(FILES "flintConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/flintConfigVersion.cmake"
        DESTINATION lib/cmake/flint)

if(${BUILD_TEST})
  project(test)
  add_executable(test test/test.cpp)
  target_link_libraries(test OpenCL::OpenCL)
  target_link_libraries(test flint)
endif()

if(${BUILD_GRADIENT_TEST})
  project(test_gradients)
  add_executable(test_gradients test/test_gradients.cpp)
  target_link_libraries(test_gradients OpenCL::OpenCL)
  target_link_libraries(test_gradients flint)
endif()

if(${BUILD_BENCHMARK})
  project(benchmark)
  add_executable(benchmark test/benchmark.cpp)
  target_link_libraries(benchmark OpenCL::OpenCL)
  target_link_libraries(benchmark flint)
endif()
