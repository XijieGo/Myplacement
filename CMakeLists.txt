cmake_minimum_required(VERSION 3.22)

project(MyPlacement VERSION 1.0.0 LANGUAGES CXX)

option(MYPLACEMENT_BUILD_TESTS "Build the test suite" ON)
option(MYPLACEMENT_ENABLE_SANITIZERS "Enable Address/Undefined sanitizers" OFF)
option(MYPLACEMENT_ENABLE_CUDA "Build the optional CUDA global-placement backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_package(Threads REQUIRED)

set(MYPLACEMENT_SOURCES
    src/model/PlacementDatabase.cpp
    src/io/BookshelfParser.cpp
    src/metrics/Metrics.cpp
    src/placement/InitialPlacer.cpp
    src/placement/ElectrostaticField.cpp
    src/placement/GlobalPlacementSupport.cpp
    src/placement/AdaptiveGlobalPlacer.cpp
    src/placement/LegacyGlobalPlacer.cpp
    src/placement/GlobalPlacer.cpp
    src/placement/CudaPlacementBackendStub.cpp
    src/placement/Legalizer.cpp
    src/export/Renderer.cpp
    src/export/GdsWriter.cpp
)

if(MYPLACEMENT_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(REMOVE_ITEM MYPLACEMENT_SOURCES src/placement/CudaPlacementBackendStub.cpp)
    list(APPEND MYPLACEMENT_SOURCES src/placement/cuda/CudaPlacementBackend.cu)
endif()

add_library(myplacement ${MYPLACEMENT_SOURCES})

target_include_directories(myplacement
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(myplacement SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/CImg)

target_link_libraries(myplacement
    PUBLIC
        Eigen3::Eigen
        PkgConfig::FFTW3
        Threads::Threads
)

target_compile_options(myplacement PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>)

if(MYPLACEMENT_ENABLE_CUDA)
    set_target_properties(myplacement PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES 80)
    target_compile_definitions(myplacement PRIVATE MYPLACEMENT_HAS_CUDA=1)
    target_link_libraries(myplacement PUBLIC CUDA::cudart CUDA::cufft)
endif()

if(MYPLACEMENT_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(myplacement PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(myplacement PUBLIC -fsanitize=address,undefined)
endif()

add_executable(myplace src/app/main.cpp)
target_link_libraries(myplace PRIVATE myplacement)

if(MYPLACEMENT_BUILD_TESTS)
    enable_testing()
    add_executable(myplacement_tests
        tests/test_main.cpp
        tests/DensityFieldTests.cpp
        tests/DensityModelTests.cpp
        tests/PlacementFlowTests.cpp
    )
    target_link_libraries(myplacement_tests PRIVATE myplacement)
    target_compile_definitions(myplacement_tests PRIVATE
        MYPLACEMENT_TEST_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures")
    add_test(NAME myplacement_tests COMMAND myplacement_tests)
endif()
