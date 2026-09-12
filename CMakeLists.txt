cmake_minimum_required(VERSION 3.14)

project(inference
    VERSION 0.1.0
    DESCRIPTION "VoxCPM-0.5B TTS Inference with GGML (CPU only)"
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
# vendored 子树已剪裁 examples/ tests/ 目录，显式关掉对应开关
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "ggml: build examples" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "ggml: build tests"    FORCE)

# -----------------------------------------------------------------------------
# Qualcomm Hexagon NPU backend (可选，-DGGML_HEXAGON=ON 启用)
# GGML_HEXAGON_USE_MEMPOOL: ON = FastRPC/mempool 实现, OFF = 官方 dspqueue 实现
# -----------------------------------------------------------------------------
option(GGML_HEXAGON             "ggml: enable Hexagon NPU backend" OFF)
option(GGML_HEXAGON_USE_MEMPOOL "ggml-hexagon: use FastRPC/mempool AP implementation" OFF)

if(GGML_HEXAGON)
    # HEXAGON_SDK_ROOT 解析顺序：-D 传入 > 环境变量 > ~/hexagon-sdk* 最新版本
    if(NOT HEXAGON_SDK_ROOT)
        if(DEFINED ENV{HEXAGON_SDK_ROOT})
            set(HEXAGON_SDK_ROOT "$ENV{HEXAGON_SDK_ROOT}" CACHE PATH "Hexagon SDK root")
        else()
            file(GLOB _HEX_SDK_CANDS "$ENV{HOME}/hexagon-sdk*")
            if(_HEX_SDK_CANDS)
                list(SORT _HEX_SDK_CANDS ORDER DESCENDING)
                list(GET _HEX_SDK_CANDS 0 _HEX_SDK_DEFAULT)
                set(HEXAGON_SDK_ROOT "${_HEX_SDK_DEFAULT}" CACHE PATH "Hexagon SDK root")
            endif()
        endif()
    endif()
    if(NOT HEXAGON_SDK_ROOT)
        message(FATAL_ERROR "GGML_HEXAGON=ON 需要 HEXAGON_SDK_ROOT（-D 传入或 export HEXAGON_SDK_ROOT=...）")
    endif()
    # SDK 6.6 的 hexagon_fun.cmake 在 include 时无条件执行 string(FIND ${PREBUILT_LIB_DIR} ...)，
    # 而 ggml-hexagon 只在 DSP ExternalProject 子构建中传该变量。父作用域给出占位值避免配置失败；
    # 仅影响 SDK 脚本的 toolvXX 探测分支，实际编译用的是子构建传入的 toolv19_${V_ARCH}
    if(NOT DEFINED PREBUILT_LIB_DIR)
        set(PREBUILT_LIB_DIR "toolv19_v79" CACHE STRING "hexagon_fun.cmake include-time placeholder")
    endif()
    # HEXAGON_TOOLS_ROOT 留空即可：ggml-hexagon/CMakeLists.txt 会从 ${HEXAGON_SDK_ROOT}/hexagon_sdk.json 推导
    message(STATUS "voxcpm: HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}")
endif()

add_subdirectory(${GGML_DIR} ggml)

# x86_64 宿主编译器（gcc<12 / clang 需 AVX512FP16）不支持 _Float16，而 AP 侧
# ggml-hexagon 经 htp/unary-ops.h 仅用到 sizeof(_Float16)（==2），用 uint16_t
# 等宽替换即可在宿主机完成编译冒烟；DSP 侧经 hexagon-clang 交叉编译不受影响，
# aarch64 目标编译器原生支持 _Float16，此定义不会生效
if(TARGET ggml-hexagon AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_definitions(ggml-hexagon PRIVATE _Float16=uint16_t)
    message(STATUS "voxcpm: x86_64 宿主编译 ggml-hexagon，注入 _Float16=uint16_t 等宽替换")
endif()

# mempool 变体的 ggml-hexagon-fastrpc.cpp 使用 std::filesystem 但未 include
# <filesystem>（上游依赖其 SDK 头文件传递包含，完整版 SDK 6.6 头不传递），
# 用 -include 预包含修复，保持 hexagon 目录源码 verbatim
if(TARGET ggml-hexagon AND GGML_HEXAGON_USE_MEMPOOL)
    target_compile_options(ggml-hexagon PRIVATE -include filesystem)
    message(STATUS "voxcpm: mempool 变体注入 -include filesystem 预包含")
endif()

# =============================================================================
# VoxCPM Library（完整 TTS 引擎：AudioVAE + MiniCPM LM + LocEnc/LocDiT + CFM/FSQ
# + tokenizer + runtime；自 VoxCPM.cpp 迁移，仅 CPU 路径）
# =============================================================================

add_library(voxcpm STATIC
    third_party/miniaudio/miniaudio.c
    src/audio-vae.cpp
    src/audio_io.cpp
    src/backend.cpp
    src/components.cpp
    src/context.cpp
    src/fsq.cpp
    src/localenc.cpp
    src/locdit.cpp
    src/minicpm.cpp
    src/output.cpp
    src/state.cpp
    src/tokenizer.cpp
    src/unified_cfm.cpp
    src/voxcpm.cpp
    src/weight-store.cpp
)

target_include_directories(voxcpm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${GGML_DIR}/include
        ${GGML_DIR}/src
        ${CMAKE_SOURCE_DIR}/third_party/miniaudio
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
        VOXCPM_ENABLE_MP3=1
        VOXCPM_ENABLE_OPUS=0
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

# =============================================================================
# voxcpm_tts: VoxCPM-0.5B 文本转语音 CLI（自 VoxCPM.cpp examples/voxcpm_tts.cpp
# 迁移，--backend 仅接受 cpu）
# =============================================================================

add_executable(voxcpm_tts voxcpm_tts.cpp)
target_link_libraries(voxcpm_tts PRIVATE voxcpm)
