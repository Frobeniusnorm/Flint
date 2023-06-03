cmake_minimum_required(VERSION 3.9)
project(flint_dl VERSION 0.0.1)
FILE(GLOB SRCFILES_DL src/*.cpp)
add_library(flint_dl STATIC ${SRCFILES_DL})
set_property(TARGET flint_dl PROPERTY CXX_STANDARD 20)
target_include_directories(flint_dl PRIVATE .)
find_package(OpenCL REQUIRED)
find_package(flint REQUIRED)
target_link_libraries(flint_dl OpenCL::OpenCL)
target_link_libraries(flint_dl flint::flint)
set_target_properties(flint_dl PROPERTIES VERSION ${PROJECT_VERSION} PUBLIC_HEADER "layers.hpp;losses.hpp;models.hpp;optimizers.hpp;flint_dl.hpp")
install(TARGETS flint_dl 
    EXPORT flint_dl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flint_dl)
