# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generic configuration and JSON helpers shared by the pipelines.
add_library(nemo_speech_common STATIC
    json.cpp
    parameter_parser.cpp
)
target_include_directories(nemo_speech_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(nemo_speech_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Audio I/O and resampling are shared by ASR, diarization, and TTS. Keeping
# them in one library avoids compiling the frontend separately into pipelines.
add_library(nemo_speech_audio SHARED
    audio_file.cpp
    audio_resampler.cpp
)
target_include_directories(nemo_speech_audio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32)
    set_target_properties(nemo_speech_audio PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
install(TARGETS nemo_speech_audio
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# The streaming decoder is currently used only by its focused test. It is
# excluded from the default build and is pulled in when that test is enabled.
if(NEMO_SPEECH_BUILD_ASR)
    add_library(nemo_speech_audio_decoder STATIC EXCLUDE_FROM_ALL audio_decoder.cpp)
    target_include_directories(nemo_speech_audio_decoder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(nemo_speech_audio_decoder PUBLIC nemo_speech_audio)
endif()

# Subtitle formatting belongs to the ASR/live transcript path, not diarization.
if(NEMO_SPEECH_BUILD_ASR)
    add_library(nemo_speech_subtitles STATIC EXCLUDE_FROM_ALL subtitles.cpp)
    target_include_directories(nemo_speech_subtitles PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Shared WFST grammar runtime for ASR ITN and TTS TN. Keep this separate from
# nemo_speech_common so builds that do not request text normalization retain
# no OpenFST/Sparrowhawk dependency. It is a shared library of its own so that
# its protobuf runtime stays out of the ASR library, which links SentencePiece's
# bundled protobuf, and so ASR and TTS share one copy.
if(NEMO_SPEECH_WITH_NORM)
    add_library(nemo_speech_text_normalization SHARED
        text_normalization/fst_normalizer.cpp)
    target_compile_definitions(
        nemo_speech_text_normalization PUBLIC NEMO_SPEECH_WITH_NORM=1)

    set(_NEMO_SPEECH_ITN_PREFIX "${NEMO_SPEECH_DEPENDENCY_PREFIX}/itn")
    find_library(SPARROWHAWK_LIB sparrowhawk HINTS "${_NEMO_SPEECH_ITN_PREFIX}/lib" REQUIRED)
    find_library(FSTFAR_LIB fstfar HINTS "${_NEMO_SPEECH_ITN_PREFIX}/lib" REQUIRED)
    find_library(FST_LIB fst HINTS "${_NEMO_SPEECH_ITN_PREFIX}/lib" REQUIRED)
    find_library(RE2_LIB re2 HINTS "${_NEMO_SPEECH_ITN_PREFIX}/lib" REQUIRED)
    find_path(SPARROWHAWK_INCLUDE_DIR sparrowhawk/normalizer.h
        HINTS "${_NEMO_SPEECH_ITN_PREFIX}/include" REQUIRED)
    find_path(OPENFST_INCLUDE_DIR fst/fst.h HINTS "${_NEMO_SPEECH_ITN_PREFIX}/include" REQUIRED)
    find_package(Threads REQUIRED)

    # scripts/build_itn_deps.sh with STATIC=1 installs protobuf and RE2 into
    # the same prefix; otherwise use the system protobuf.
    find_library(ITN_PROTOBUF_LIB protobuf
        PATHS "${_NEMO_SPEECH_ITN_PREFIX}/lib" NO_DEFAULT_PATH)
    set(ABSL_LIBS)
    if(ITN_PROTOBUF_LIB)
        set(ITN_PROTOBUF_LIBS ${ITN_PROTOBUF_LIB})
    else()
        find_package(Protobuf REQUIRED)
        find_library(ITN_PROTOBUF_LIB
            NAMES libprotobuf.so.25.3.0 libprotobuf.so.25
            PATHS /lib /lib/x86_64-linux-gnu /usr/lib /usr/lib/x86_64-linux-gnu
            NO_DEFAULT_PATH)
        if(NOT ITN_PROTOBUF_LIB)
            if(TARGET protobuf::libprotobuf)
                set(ITN_PROTOBUF_LIB protobuf::libprotobuf)
            else()
                set(ITN_PROTOBUF_LIB protobuf)
            endif()
        endif()
        find_library(UTF8_RANGE_LIB NAMES utf8_range_lib PATHS /usr/lib /lib)
        file(GLOB ABSL_LIBS /usr/lib/libabsl_*.so /lib/libabsl_*.so)
        set(ITN_PROTOBUF_LIBS ${ITN_PROTOBUF_LIB})
        if(UTF8_RANGE_LIB)
            list(APPEND ITN_PROTOBUF_LIBS ${UTF8_RANGE_LIB})
        endif()
    endif()

    target_include_directories(nemo_speech_text_normalization
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/text_normalization
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/text_normalization/compat)
    target_include_directories(nemo_speech_text_normalization SYSTEM
        PRIVATE
            ${OPENFST_INCLUDE_DIR}
            ${SPARROWHAWK_INCLUDE_DIR})
    # OpenFST registers its FST types from static initializers; keep them when
    # linking the static archive.
    set(_NEMO_SPEECH_FST_LINK ${FST_LIB})
    if(FST_LIB MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
        set(_NEMO_SPEECH_FST_LINK "$<LINK_LIBRARY:WHOLE_ARCHIVE,${FST_LIB}>")
    endif()
    target_link_libraries(nemo_speech_text_normalization PRIVATE
        ${SPARROWHAWK_LIB} ${FSTFAR_LIB} ${_NEMO_SPEECH_FST_LINK}
        ${ITN_PROTOBUF_LIBS})
    if(ABSL_LIBS)
        target_link_libraries(nemo_speech_text_normalization PRIVATE
            -Wl,--as-needed ${ABSL_LIBS} -Wl,--no-as-needed)
    endif()
    target_link_libraries(nemo_speech_text_normalization PRIVATE
        ${RE2_LIB} Threads::Threads ${CMAKE_DL_LIBS})
    if(UNIX AND NOT APPLE)
        # Keep the symbols of statically linked dependencies private.
        target_link_options(nemo_speech_text_normalization PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    install(TARGETS nemo_speech_text_normalization
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

    file(GLOB _NEMO_SPEECH_NORM_RUNTIME_LIBS
        "${_NEMO_SPEECH_ITN_PREFIX}/lib/libsparrowhawk${CMAKE_SHARED_LIBRARY_SUFFIX}*"
        "${_NEMO_SPEECH_ITN_PREFIX}/lib/libfst${CMAKE_SHARED_LIBRARY_SUFFIX}*"
        "${_NEMO_SPEECH_ITN_PREFIX}/lib/libfstfar${CMAKE_SHARED_LIBRARY_SUFFIX}*"
        "${_NEMO_SPEECH_ITN_PREFIX}/lib/libsparrowhawk.*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_NEMO_SPEECH_ITN_PREFIX}/lib/libfst.*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_NEMO_SPEECH_ITN_PREFIX}/lib/libfstfar.*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    install(FILES ${_NEMO_SPEECH_NORM_RUNTIME_LIBS}
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
    set(_NEMO_SPEECH_ITN_LICENSE_DIR
        "${_NEMO_SPEECH_ITN_PREFIX}/share/licenses/nemo-speech/third_party")
    if(EXISTS "${_NEMO_SPEECH_ITN_LICENSE_DIR}")
        install(DIRECTORY "${_NEMO_SPEECH_ITN_LICENSE_DIR}/"
            DESTINATION "${NEMO_SPEECH_LICENSE_INSTALL_DIR}/third_party")
    endif()
endif()
