# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.26)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _version_lines)
foreach(_line IN LISTS _version_lines)
    if(_line MATCHES "^NEMO_SPEECH_VERSION:[ \t]*([0-9]+\\.[0-9]+\\.[0-9]+)([-+][0-9A-Za-z.-]+)?[ \t]*$")
        set(NEMO_SPEECH_VERSION "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        set(NEMO_SPEECH_PROJECT_VERSION "${CMAKE_MATCH_1}")
    endif()
endforeach()
if(NOT NEMO_SPEECH_VERSION OR NOT NEMO_SPEECH_PROJECT_VERSION)
    message(FATAL_ERROR "could not parse NEMO_SPEECH_VERSION from VERSION")
endif()

project(nemo_speech VERSION ${NEMO_SPEECH_PROJECT_VERSION} LANGUAGES C CXX)
add_compile_definitions(NEMO_SPEECH_VERSION_STR="${NEMO_SPEECH_VERSION}")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(WIN32)
    if(NOT DEFINED GGML_OPENMP OR GGML_OPENMP)
        set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)
    endif()
    include(InstallRequiredSystemLibraries)
    add_compile_definitions(NOMINMAX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_TESTING "Build diarization and shared audio tests" OFF)
option(NEMO_SPEECH_GGML_PATCHED "Apply the required ggml patch series" ON)
option(GGML_VULKAN "Enable the ggml Vulkan backend" OFF)
option(GGML_METAL "Enable the ggml Metal backend" OFF)
if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Enable ggml CUDA Graphs")
endif()

set(GGML_DEPENDENCIES ggml ggml-base ggml-cpu)
if(GGML_CUDA)
    list(APPEND GGML_DEPENDENCIES ggml-cuda)
endif()
if(GGML_VULKAN)
    list(APPEND GGML_DEPENDENCIES ggml-vulkan)
endif()
if(GGML_METAL)
    list(APPEND GGML_DEPENDENCIES ggml-metal)
endif()
set(GGML_LLAMAFILE ON CACHE BOOL "Use ggml llamafile CPU matrix kernels")

include(cmake/llama_cpp.cmake)
add_subdirectory("${NEMO_SPEECH_LLAMA_CPP_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
set_property(TARGET ggml PROPERTY PUBLIC_HEADER "")
install(TARGETS ggml ggml-base ggml-cpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
foreach(_backend ggml-cuda ggml-vulkan ggml-metal)
    if(TARGET ${_backend})
        install(TARGETS ${_backend}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
endforeach()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra>")
endif()

add_subdirectory(src/runtime/ggml)
add_subdirectory(src/common)
add_subdirectory(src/audio_frontend)
add_subdirectory(src/diar)
add_subdirectory(app)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

install(FILES
    include/nemo_speech/common.h
    include/nemo_speech/diar.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nemo_speech)
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/nemo-speech")
install(FILES README.md CONTRIBUTING.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nemo-speech")
install(DIRECTORY docs/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nemo-speech/docs")
install(DIRECTORY config/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nemo-speech/config")
install(FILES llama.cpp/LICENSE
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/nemo-speech/third_party/ggml")

configure_package_config_file(
    cmake/NeMoSpeechConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NeMoSpeech")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NeMoSpeechConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NeMoSpeech")
