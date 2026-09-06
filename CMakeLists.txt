cmake_minimum_required(VERSION 3.14)

project(inference
    VERSION 0.1.0
    DESCRIPTION "VoxCPM AudioVAE Inference with GGML (CPU only)"
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Threads REQUIRED)

# =============================================================================
# GGML (CPU backend only)
# =============================================================================

set(GGML_DIR ${CMAKE_SOURCE_DIR}/third_party/ggml)
set(GGML_CUDA OFF CACHE BOOL "ggml: use CUDA" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "ggml: use Vulkan" FORCE)
set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)

add_subdirectory(${GGML_DIR} ggml)

# =============================================================================
# VoxCPM AudioVAE Library（仅 AudioVAE encoder/decoder + 基础设施）
# =============================================================================

add_library(voxcpm STATIC
    src/audio-vae.cpp
    src/backend.cpp
    src/context.cpp
    src/weight-store.cpp
)

target_include_directories(voxcpm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${GGML_DIR}/include
        ${GGML_DIR}/src
)

target_link_libraries(voxcpm
    PUBLIC
        ggml
        ggml-base
        Threads::Threads
)

target_compile_definitions(voxcpm
    PRIVATE
        $<$<CONFIG:Debug>:VOXCPM_DEBUG>
)

# Keep the historical output path ./build/examples/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)

# =============================================================================
# voxcpm-vae-decode-mock: AudioVAE 独立解码重放工具（mock 捕获 → WAV）
# =============================================================================

add_executable(voxcpm-vae-decode-mock vae_decode_mock.cpp)
target_link_libraries(voxcpm-vae-decode-mock PRIVATE voxcpm)

# =============================================================================
# voxcpm-vae-dump-tensor-check: GGUF 张量加载校验小工具
# =============================================================================

add_executable(voxcpm-vae-dump-tensor-check dump_tensor_check.cpp)
target_link_libraries(voxcpm-vae-dump-tensor-check PRIVATE voxcpm)
